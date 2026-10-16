"""jevmond: a long-running daemon over the exec pipeline, controlled by a Unix socket.

The daemon owns the shared `Monitor` (source -> context -> judge -> store) and serves the versioned
newline-JSON protocol in `protocol.py` on `$XDG_RUNTIME_DIR/jevmon/jevmond.sock` (mode 0600, in a
0700 directory; a socket under the state dir if there is no runtime dir). A live daemon already on
the socket is refused; a stale socket file is removed. SIGINT/SIGTERM shut down cleanly: the monitor
stops and flushes, the store is closed and the socket is unlinked.

The daemon adds no judgement of its own: it reports the model's verdicts and counters, records
human labels and acks, and can pause the judge while the poller and census keep running.
"""

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import signal
import sys
from pathlib import Path

from jevmon.config import Config, ConfigError, load_config
from jevmon.core import SqliteStore
from jevmon.daemon import protocol
from jevmon.monitor import Monitor
from jevmon.paths import default_socket_path, state_dir
from jevmon.sources.replay import ReplayError


class DaemonError(RuntimeError):
    """The daemon cannot start (socket owned by a live daemon, bad path)."""


class DaemonServer:
    """Serve the control socket for one running monitor."""

    def __init__(
        self,
        monitor: Monitor,
        store: SqliteStore,
        socket_path: Path,
        *,
        poll_interval: float = 0.25,
        stats_interval: float = 2.0,
    ) -> None:
        self.monitor = monitor
        self.store = store
        self.socket_path = Path(socket_path)
        self._poll_interval = poll_interval
        self._stats_interval = stats_interval
        self._shutdown = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.AbstractServer | None = None
        self._tasks: set[asyncio.Task] = set()
        self._lock_fd: int | None = None
        self._owns_socket = False
        self._closed = False

    @property
    def bound(self) -> bool:
        """True once the listening socket exists (used by tests to wait for startup)."""
        return self._server is not None

    # -- lifecycle ------------------------------------------------------------------------

    async def serve(self) -> int:
        self._prepare_socket()
        self._loop = asyncio.get_running_loop()
        self._install_signal_handlers()
        server = await asyncio.start_unix_server(
            self._handle, path=str(self.socket_path), limit=protocol.MAX_LINE
        )
        self._server = server
        os.chmod(self.socket_path, 0o600)
        self._announce(f"listening on {self.socket_path}  store {self.store.path}")
        reporter = asyncio.create_task(self._report())
        try:
            async with server:
                await self._shutdown.wait()
        finally:
            reporter.cancel()
            await self._close()
            self._announce("stopped")
        return 0

    @staticmethod
    def _announce(text: str) -> None:
        print(f"jevmond: {text}", file=sys.stderr, flush=True)

    async def _report(self, heartbeat: float = 60.0, tick: float = 2.0) -> None:
        """Say what the daemon is doing: a silent but healthy daemon looks exactly like a hung one."""
        last_model: str | None = None
        elapsed = heartbeat  # one line right away
        while True:
            status = self.monitor.status()
            model = status.get("model_status")
            if model != last_model:
                last_model = model
                self._announce(f"model: {model}")
            if elapsed >= heartbeat:
                elapsed = 0.0
                self._announce(
                    f"up {status.get('uptime', 0):.0f}s  judged {status.get('judged', 0)}  "
                    f"cached {status.get('cached', 0)}  dropped {status.get('dropped', 0)}  "
                    f"miss {status.get('miss_ratio', 0):.0%}  paused {status.get('paused', False)}"
                )
            await asyncio.sleep(tick)
            elapsed += tick

    def request_shutdown(self) -> None:
        """Thread-safe request to stop serving (used by tests and signal handlers)."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._shutdown.set)

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, self._shutdown.set)

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._remove_socket()
        if self._tasks:
            _done, pending = await asyncio.wait(self._tasks, timeout=2.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.wait(pending, timeout=1.0)
        await asyncio.to_thread(self.monitor.stop)
        await asyncio.to_thread(self.store.close)
        self._release_lock()

    # -- socket ---------------------------------------------------------------------------

    def _prepare_socket(self) -> None:
        path = self.socket_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(path.parent, 0o700)
        self._acquire_lock()
        self._owns_socket = True
        # We hold the lock, so any socket file still here belongs to a dead daemon.
        path.unlink(missing_ok=True)

    def _acquire_lock(self) -> None:
        """Hold an exclusive lock beside the socket; a live daemon owns it."""
        lock_path = self.socket_path.with_name(self.socket_path.name + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise DaemonError(f"another daemon holds {lock_path}") from exc
        self._lock_fd = fd

    def _release_lock(self) -> None:
        if self._lock_fd is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(self._lock_fd)
        self._lock_fd = None

    def _remove_socket(self) -> None:
        if not self._owns_socket:
            return
        with contextlib.suppress(OSError):
            self.socket_path.unlink(missing_ok=True)

    # -- connection handling --------------------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        try:
            while not self._shutdown.is_set():
                try:
                    line = await reader.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    await self._send(writer, protocol.error(None, "request too large"))
                    break
                except (ConnectionError, OSError):
                    break
                if not line:
                    break
                if len(line) > protocol.MAX_LINE:
                    await self._send(writer, protocol.error(None, "request too large"))
                    break
                try:
                    obj = json.loads(line)
                except ValueError as exc:
                    await self._send(writer, protocol.error(None, f"invalid JSON: {exc}"))
                    continue
                try:
                    request = protocol.parse_request(obj)
                except protocol.ProtocolError as exc:
                    await self._send(writer, protocol.error(None, str(exc)))
                    continue
                if request.kind == "subscribe":
                    await self._send(writer, protocol.stream("ready", result={"subscribed": True}))
                    await self._stream(reader, writer, request)
                    return
                try:
                    result = await self._dispatch(request)
                except Exception as exc:  # noqa: BLE001 - one bad request must not kill the server
                    await self._send(writer, protocol.error(request.req_id, _describe(exc)))
                    continue
                await self._send(writer, protocol.ok(request.req_id, result))
                if request.kind == "shutdown":
                    self._shutdown.set()
                    return
        except (ConnectionError, OSError):
            pass  # client vanished mid-request
        finally:
            writer.close()
            with contextlib.suppress(OSError, ConnectionError):
                await writer.wait_closed()

    async def _dispatch(self, request: protocol.Request) -> dict:
        kind = request.kind
        if kind == "status":
            return self.monitor.status()
        if kind == "events":
            return await asyncio.to_thread(self._events, request.params)
        if kind == "event":
            return await asyncio.to_thread(self._one_event, request.params["event_id"])
        if kind == "label":
            await asyncio.to_thread(
                self.store.add_label, request.params["event_id"], request.params["label"], "human"
            )
            return await asyncio.to_thread(self._one_event, request.params["event_id"])
        if kind == "ack":
            await asyncio.to_thread(
                self.store.add_label, request.params["event_id"], "ack", "human"
            )
            return await asyncio.to_thread(self._one_event, request.params["event_id"])
        if kind == "pause":
            self.monitor.pause()
            return self.monitor.status()
        if kind == "resume":
            self.monitor.resume()
            return self.monitor.status()
        if kind == "reload":
            return await asyncio.to_thread(self.monitor.reload)
        if kind == "shutdown":
            return {"shutdown": True}
        raise protocol.ProtocolError(f"unhandled request {kind!r}")

    async def _stream(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, request: protocol.Request
    ) -> None:
        since = request.params.get("since_id")
        last_id = (
            int(since)
            if isinstance(since, int)
            else await asyncio.to_thread(self.store.max_event_id)
        )
        every = max(1, round(self._stats_interval / self._poll_interval))
        closed = asyncio.create_task(reader.read())
        tick = 0
        try:
            while not self._shutdown.is_set():
                rows = await asyncio.to_thread(
                    lambda since=last_id: self.store.query_events(
                        since_id=since, limit=protocol.MAX_LIMIT
                    )
                )
                for event_id, event, labels in rows:
                    last_id = event_id
                    payload = protocol.event_payload(event_id, event, labels)
                    await self._send(writer, protocol.stream("event", event=payload))
                if tick % every == 0:
                    stats = await asyncio.to_thread(self.monitor.status)
                    await self._send(writer, protocol.stream("stats", stats=stats))
                tick += 1
                done, _ = await asyncio.wait({closed}, timeout=self._poll_interval)
                if done:
                    with contextlib.suppress(Exception):
                        closed.result()
                    break  # the reader finished (EOF or reset): the client went away
        except (ConnectionError, OSError):
            pass
        finally:
            closed.cancel()
            with contextlib.suppress(Exception):
                await closed

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, obj: dict) -> None:
        writer.write(protocol.encode(obj))
        await writer.drain()

    # -- request handlers -----------------------------------------------------------------

    def _events(self, params: dict) -> dict:
        limit = int(params.get("limit", protocol.DEFAULT_LIMIT))
        rows = self.store.query_events(
            since_id=int(params.get("since_id", 0)),
            min_severity=int(params.get("min_severity", 0)),
            limit=limit,
            unlabeled=bool(params.get("unlabeled", False)),
            text=params.get("text"),
        )
        events = [
            protocol.event_payload(event_id, event, labels) for event_id, event, labels in rows
        ]
        return {"events": events, "count": len(events), "more": len(events) >= limit}

    def _one_event(self, event_id: int) -> dict:
        row = self.store.event_with_labels(int(event_id))
        if row is None:
            raise protocol.ProtocolError(f"no event {event_id}")
        return {"event": protocol.event_payload(*row)}


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jevmond", description="jevmon decision daemon")
    parser.add_argument("--config", help="path to a config.toml (default: XDG config)")
    parser.add_argument("--db", help="SQLite path (default: state dir)")
    parser.add_argument("--proc-root", help="alternate /proc tree (for tests)")
    parser.add_argument("--socket", help="control socket path (default: XDG runtime dir)")
    return parser


def _open_store(args: argparse.Namespace, config: Config) -> SqliteStore:
    path: str | Path = args.db or (state_dir() / "events.db")
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(path, retention_days=config.store.retention_days)
    store.prune()
    return store


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        sys.stderr.write(f"jevmond: {exc}\n")
        return 2
    socket_path = Path(args.socket) if args.socket else default_socket_path()
    store = _open_store(args, config)
    monitor = Monitor(config, store, proc_root=args.proc_root, config_path=args.config)
    server = DaemonServer(monitor, store, socket_path)
    try:
        monitor.start()
        return asyncio.run(server.serve())
    except DaemonError as exc:
        sys.stderr.write(f"jevmond: {exc}\n")
        return 2
    except ReplayError as exc:  # a bad [exec] recording: fail before serving anything
        sys.stderr.write(f"jevmond: {exc}\n")
        return 2
    finally:
        server._remove_socket()
        server._release_lock()
        monitor.stop()
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())

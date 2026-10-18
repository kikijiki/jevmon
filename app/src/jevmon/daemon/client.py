"""Async client for the jevmond control socket, plus the `jevmon ctl` entry point.

`DaemonClient` speaks the versioned newline-JSON protocol (see `server.py`). It auto-connects on
the first request, so a client built before the daemon starts reconnects once it is up. A client
used for `subscribe()` is a one-way stream: do not mix requests onto it, use a second client.

`run_ctl` backs `jevmon ctl status|events|label|ack|pause|resume|shutdown`, printing JSON so the
daemon is scriptable without the TUI.
"""

import argparse
import asyncio
import contextlib
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Self

from jevmon.daemon import protocol
from jevmon.paths import default_socket_path


class DaemonError(RuntimeError):
    """The daemon is unreachable, closed the connection or answered with an error."""


class DaemonClient:
    """One control connection. Reconnect by calling `close()` and issuing another request."""

    def __init__(self, socket_path: str | Path | None = None) -> None:
        self.socket_path = Path(socket_path) if socket_path else default_socket_path()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._next_id = 1

    @property
    def connected(self) -> bool:
        return self._writer is not None

    async def connect(self) -> None:
        if self._writer is not None:
            return
        try:
            reader, writer = await asyncio.open_unix_connection(
                str(self.socket_path), limit=protocol.MAX_REPLY
            )
        except OSError as exc:
            raise DaemonError(f"cannot connect to {self.socket_path}: {exc}") from exc
        self._reader, self._writer = reader, writer

    async def close(self) -> None:
        writer, self._writer = self._writer, None
        self._reader = None
        if writer is not None:
            writer.close()
            with contextlib.suppress(OSError, ConnectionError):
                await writer.wait_closed()

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- protocol -------------------------------------------------------------------------

    async def request(self, kind: str, **params: object) -> dict:
        await self.connect()
        req_id = self._next_id
        self._next_id += 1
        await self._send({"v": protocol.PROTOCOL_VERSION, "id": req_id, "type": kind, **params})
        msg = await self._read_json()
        if not msg.get("ok", False):
            raise DaemonError(str(msg.get("error", "request failed")))
        result = msg.get("result")
        return result if isinstance(result, dict) else {}

    async def subscribe(self, since_id: int | None = None) -> AsyncIterator[dict]:
        """Yield stream messages (`event`, `stats`, `ready`) until the daemon goes away."""
        await self.connect()
        request: dict = {"v": protocol.PROTOCOL_VERSION, "type": "subscribe"}
        if since_id is not None:
            request["since_id"] = since_id
        await self._send(request)
        while True:
            yield await self._read_json()

    async def _send(self, obj: dict) -> None:
        assert self._writer is not None
        self._writer.write(protocol.encode(obj))
        await self._writer.drain()

    async def _read_json(self) -> dict:
        assert self._reader is not None
        try:
            line = await self._reader.readline()
        except (ValueError, asyncio.LimitOverrunError) as exc:
            await self.close()
            raise DaemonError("reply too large") from exc
        if not line:
            await self.close()
            raise DaemonError("daemon closed the connection")
        try:
            msg = json.loads(line)
        except ValueError as exc:
            raise DaemonError(f"invalid reply: {exc}") from exc
        if not isinstance(msg, dict):
            raise DaemonError("invalid reply: not an object")
        return msg

    # -- convenience ----------------------------------------------------------------------

    async def status(self) -> dict:
        return await self.request("status")

    async def events(
        self,
        *,
        since_id: int = 0,
        min_severity: int = 0,
        limit: int = protocol.DEFAULT_LIMIT,
        unlabeled: bool = False,
        text: str | None = None,
    ) -> dict:
        params: dict[str, object] = {
            "since_id": since_id,
            "min_severity": min_severity,
            "limit": limit,
            "unlabeled": unlabeled,
        }
        if text:
            params["text"] = text
        return await self.request("events", **params)

    async def event(self, event_id: int) -> dict:
        return await self.request("event", event_id=event_id)

    async def label(self, event_id: int, label: str) -> dict:
        return await self.request("label", event_id=event_id, label=label)

    async def ack(self, event_id: int) -> dict:
        return await self.request("ack", event_id=event_id)

    async def pause(self) -> dict:
        return await self.request("pause")

    async def resume(self) -> dict:
        return await self.request("resume")

    async def reload(self) -> dict:
        return await self.request("reload")

    async def allow_list(self) -> dict:
        return await self.request("allow_list")

    async def allow_add(self, rule: str) -> dict:
        return await self.request("allow_add", rule=rule)

    async def allow_remove(self, rule: str) -> dict:
        return await self.request("allow_remove", rule=rule)

    async def shutdown(self) -> dict:
        return await self.request("shutdown")


# -- jevmon ctl ----------------------------------------------------------------------------


async def _run_ctl(args: argparse.Namespace) -> dict:
    client = DaemonClient(args.socket)
    try:
        command = args.ctl_command
        if command == "status":
            return await client.status()
        if command == "events":
            return await client.events(
                since_id=args.since_id,
                min_severity=args.min_severity,
                limit=args.limit,
                unlabeled=args.unlabeled,
                text=args.text,
            )
        if command == "label":
            return await client.label(args.event_id, args.label)
        if command == "ack":
            return await client.ack(args.event_id)
        if command == "pause":
            return await client.pause()
        if command == "resume":
            return await client.resume()
        if command == "allow":
            if args.allow_command == "list":
                return await client.allow_list()
            if args.allow_command == "add":
                return await client.allow_add(args.rule)
            return await client.allow_remove(args.rule)
        if command == "shutdown":
            return await client.shutdown()
        raise DaemonError(f"unknown ctl command {command!r}")
    finally:
        await client.close()


def run_ctl(args: argparse.Namespace) -> int:
    try:
        result = asyncio.run(_run_ctl(args))
    except DaemonError as exc:
        sys.stderr.write(f"jevmon: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
    return 0


__all__ = ["DaemonClient", "DaemonError", "run_ctl"]

"""eBPF exec source: read the privileged helper's exec stream from its unix socket.

`jevmon-exec` (app/helper/) runs as root, sees every successful execve and writes one JSON line
per exec to a socket that only this user may connect to (SO_PEERCRED). This source connects,
turns each line into the same `Event` the `/proc` poller produces, and reports exact counters:
execs observed, events the helper lost (ring buffer full), events it could not deliver, and
per-second rates.

Wire format (version 1), newline-delimited JSON objects with a `t` key:

- `hello`: `v`, `argv_max`, `args_max`, `exe_source` ("kernel" or "none"), `version`.
- `exec`: `ts` (wall), `mono`, `start`, `pid`, `ppid`, `pstart`, `uid`, `euid`, `gid`, `cgroup`
  (id), `comm`, `pcomm`, `exe` (resolved path of the binary that ran), `filename` (the raw
  execve argument), `argv` (list), `argc`, `trunc` (argv cut at the helper's limits), `flags`.
- `hb`: once a second: `seen`, `emitted`, `lost`, `undelivered`, `client_dropped` totals.

If the helper is not there the source says so (`status()["exec_error"]`) and retries with
backoff; it never falls back to polling. Lines that do not parse are counted and skipped.

The parent's command comes from a bounded cache of recent execs keyed by pid and checked
against the parent's start time, so a reused pid cannot be mistaken for the parent that exec'd
earlier. A parent that exec'd before the helper started is not in the cache; its `comm` from the
kernel is used and the exe is empty (unknown), never guessed.
"""

import contextlib
import json
import os
import socket
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass

from jevmon.events import Event, Severity
from jevmon.redact import redact_argv_text
from jevmon.sanitize import clean, shorten
from jevmon.sources.exec import KIND_EXEC, KIND_SUMMARY, ExecSettings
from jevmon.sources.facts import FactReader
from jevmon.sources.proc import RealProcReader

PROTOCOL = 1
HEARTBEAT_STALE = 5.0  # no line at all for this long: the helper is gone or wedged
RATE_WINDOW = 60  # seconds of per-second exec counts kept for the rate and peak
CGROUP_RESCAN = 5.0  # at most one cgroup tree walk per this many seconds
CGROUP_MAX = 50_000
HELPER_HINT = "run: jevmon start"
UNAVAILABLE = "exec source unavailable (helper not running)"


@dataclass(slots=True)
class _Parent:
    start: int
    exe: str
    comm: str
    ppid: int
    pstart: int
    argv: str


class CgroupResolver:
    """cgroup2 inode id -> path, from a bounded cache refilled by a rate-limited tree walk."""

    def __init__(self, root: str = "/sys/fs/cgroup", mono: Callable[[], float] = time.monotonic):
        self.root = root
        self._mono = mono
        self._paths: dict[int, str] = {}
        self._last_scan = -1e9

    def path(self, cgroup_id: int) -> str:
        found = self._paths.get(cgroup_id)
        if found is not None:
            return found
        now = self._mono()
        if now - self._last_scan >= CGROUP_RESCAN:
            self._last_scan = now
            self._scan()
            found = self._paths.get(cgroup_id)
        return found or ""

    def _scan(self) -> None:
        paths: dict[int, str] = {}
        stack = [("", self.root)]
        while stack and len(paths) < CGROUP_MAX:
            rel, directory = stack.pop()
            try:
                st = os.stat(directory)
                paths[st.st_ino] = rel or "/"
                with os.scandir(directory) as it:
                    stack += [
                        (f"{rel}/{e.name}", e.path) for e in it if e.is_dir(follow_symlinks=False)
                    ]
            except OSError:
                continue
        self._paths = paths


class EbpfSource:
    """Read the helper socket; `emit` receives every event the source produces."""

    def __init__(
        self,
        settings: ExecSettings | None = None,
        emit: Callable[[Event], None] | None = None,
        clock: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
        cgroup_root: str = "/sys/fs/cgroup",
        cpu_clock: Callable[[], float] = time.thread_time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or ExecSettings(source="ebpf")
        self._emit = emit if emit is not None else (lambda event: None)
        self._clock = clock
        self._mono = mono
        self._cpu_clock = cpu_clock
        self._sleep = sleep
        self._cpu_mark = (mono(), cpu_clock())
        self.throttled_seconds = 0.0
        self._cgroups = CgroupResolver(cgroup_root)
        self._recent: OrderedDict[int, _Parent] = OrderedDict()
        self._facts = FactReader(RealProcReader(self.settings.proc_root), self.settings.facts)
        self._lock = threading.Lock()
        # connection state
        self.connected = False
        self.error = f"{UNAVAILABLE}; {HELPER_HINT}"
        self.helper_version = ""
        self.exe_source = ""
        self.protocol_error = ""
        self._retry_in = 0.0
        # counters
        self.observed_total = 0  # execs received from the helper
        self.malformed = 0
        self.exe_unresolved = 0
        self.truncated = 0
        self.helper_seen = 0  # helper totals, latest heartbeat (reset on reconnect)
        self.helper_lost = 0
        self.helper_undelivered = 0
        self.helper_client_dropped = 0
        self._base_lost = 0  # totals from earlier connections of this run
        self._base_dropped = 0
        self._base_seen = 0
        self._per_second: deque[tuple[int, int]] = deque(maxlen=RATE_WINDOW)
        self._cur_second = 0
        self._cur_count = 0
        self._last_summary: float | None = None
        self._sum_observed = 0
        self._sum_lost = 0

    # -- status ----------------------------------------------------------------------------

    @property
    def effective_interval(self) -> float:
        return 0.0

    @property
    def tracked(self) -> int:
        return len(self._recent)

    @property
    def missed_total(self) -> int:
        """Events that existed but never reached the daemon: ring buffer losses plus events the
        helper could not queue for this client. Exact counts from the helper."""
        return self._base_lost + self.helper_lost + self._base_dropped + self.helper_client_dropped

    @property
    def miss_ratio(self) -> float:
        total = self.observed_total + self.missed_total
        return self.missed_total / total if total else 0.0

    def apply_settings(self, settings: ExecSettings) -> None:
        self.settings = settings
        self._facts = FactReader(RealProcReader(settings.proc_root), settings.facts)

    def rates(self) -> tuple[float, int]:
        """(average execs per second over the last RATE_WINDOW seconds, busiest second)."""
        with self._lock:
            done = list(self._per_second)
        if not done:
            return 0.0, 0
        return sum(n for _, n in done) / len(done), max(n for _, n in done)

    def status(self) -> dict:
        per_s, peak = self.rates()
        notes = ""
        if self.connected and self.exe_source != "kernel":
            notes = "helper could not resolve exe paths; using the execve filename"
        return {
            "exec_source": "ebpf",
            "exec_connected": self.connected,
            "exec_error": "" if self.connected else self.error,
            "exec_notes": notes,
            "exec_seen": self.helper_seen + self._base_seen,
            "exec_observed": self.observed_total,
            "exec_lost": self._base_lost + self.helper_lost,
            "exec_undelivered": self.helper_undelivered,
            "exec_client_dropped": self._base_dropped + self.helper_client_dropped,
            "exec_per_s": round(per_s, 1),
            "exec_peak_per_s": peak,
            "exec_truncated": self.truncated,
            "exec_malformed": self.malformed,
            "exec_exe_unresolved": self.exe_unresolved,
            "exec_throttled_s": round(self.throttled_seconds, 1),
        }

    # -- run loop --------------------------------------------------------------------------

    def run(self, stop: threading.Event) -> None:
        backoff = self.settings.helper_retry_min
        while not stop.is_set():
            sock = self._connect()
            if sock is None:
                stop.wait(backoff)
                backoff = min(self.settings.helper_retry_max, backoff * 2)
                continue
            backoff = self.settings.helper_retry_min
            try:
                self._serve(sock, stop)
            finally:
                self._disconnect(sock)
            if not stop.is_set():
                stop.wait(backoff)

    def _connect(self) -> socket.socket | None:
        path = self.settings.helper_socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(2.0)
            sock.connect(path)
        except OSError as exc:
            sock.close()
            if isinstance(exc, FileNotFoundError | ConnectionRefusedError):
                self.error = f"{UNAVAILABLE}; {HELPER_HINT}"
            elif isinstance(exc, PermissionError):
                self.error = f"exec source unavailable (helper refused this user: {exc})"
            else:
                self.error = f"{UNAVAILABLE} ({exc.strerror or exc}); {HELPER_HINT}"
            self.connected = False
            return None
        sock.settimeout(1.0)
        return sock

    def _disconnect(self, sock: socket.socket) -> None:
        with contextlib.suppress(OSError):
            sock.close()
        if self.connected:
            self.error = f"{UNAVAILABLE}: connection to the helper was lost; {HELPER_HINT}"
        self.connected = False
        self._base_lost += self.helper_lost
        self._base_dropped += self.helper_client_dropped
        self._base_seen += self.helper_seen
        self.helper_lost = self.helper_client_dropped = self.helper_seen = 0

    def _serve(self, sock: socket.socket, stop: threading.Event) -> None:
        buf = b""
        last_data = self._mono()
        hello = False
        while not stop.is_set():
            try:
                chunk = sock.recv(262_144)
            except TimeoutError:
                if self._mono() - last_data > HEARTBEAT_STALE:
                    self.error = f"{UNAVAILABLE}: no data from the helper; {HELPER_HINT}"
                    return
                continue
            except OSError:
                return
            if not chunk:
                return
            last_data = self._mono()
            buf += chunk
            *lines, buf = buf.split(b"\n")
            if len(buf) > 4_000_000:  # a line this long is not ours
                buf = b""
                self.malformed += 1
            for line in lines:
                if not line:
                    continue
                if not hello:
                    hello = self._hello(line)
                    if not hello:
                        return
                    continue
                self.handle_line(line)
            self._throttle()

    def _throttle(self) -> None:
        """Hold the reader's own CPU (thread time) to `helper_cpu_budget` of a core.

        When the feed outruns the budget the source stops reading for a while; the helper's
        per-client buffer then fills and it drops and counts events (`exec_client_dropped`),
        so overload shows up as exact counts instead of a pegged core.
        """
        now, cpu = self._mono(), self._cpu_clock()
        then, cpu_then = self._cpu_mark
        wall = now - then
        if wall < 0.25:
            return
        used = cpu - cpu_then
        budget = self.settings.helper_cpu_budget
        self._cpu_mark = (now, cpu)
        if budget > 0 and used > budget * wall:
            pause = min(1.0, used / budget - wall)
            self.throttled_seconds += pause
            self._sleep(pause)
            self._cpu_mark = (self._mono(), self._cpu_clock())

    def _hello(self, line: bytes) -> bool:
        try:
            obj = json.loads(line)
            if obj.get("t") != "hello":
                raise ValueError("not a hello")
            version = int(obj["v"])
        except (ValueError, KeyError, TypeError, AttributeError):
            self.error = f"{UNAVAILABLE}: the peer did not speak the helper protocol"
            return False
        if version != PROTOCOL:
            self.error = (
                f"exec source unavailable (helper protocol v{version}, expected v{PROTOCOL})"
            )
            return False
        self.helper_version = str(obj.get("version", ""))
        self.exe_source = str(obj.get("exe_source", ""))
        self.connected = True
        self.error = ""
        return True

    # -- lines -----------------------------------------------------------------------------

    def handle_line(self, line: bytes | str) -> None:
        """Process one wire line. Public so tests and the replay bench can feed lines directly."""
        try:
            obj = json.loads(line)
            kind = obj["t"]
        except (ValueError, KeyError, TypeError):
            self.malformed += 1
            return
        if kind == "exec":
            try:
                event = self._exec_event(obj)
            except (KeyError, ValueError, TypeError, AttributeError):
                self.malformed += 1
                return
            self._count(event.time)
            self._emit(event)
        elif kind == "hb":
            try:
                self.helper_seen = int(obj["seen"])
                self.helper_lost = int(obj["lost"])
                self.helper_undelivered = int(obj.get("undelivered", 0))
                self.helper_client_dropped = int(obj.get("client_dropped", 0))
            except (KeyError, ValueError, TypeError):
                self.malformed += 1
                return
            self._flush_second(self._clock())
            summary = self._maybe_summary(self._clock())
            if summary is not None:
                self._emit(summary)
        elif kind == "hello":
            pass
        else:
            self.malformed += 1

    def _count(self, when: float) -> None:
        self.observed_total += 1
        self._sum_observed += 1
        second = int(self._mono())
        with self._lock:
            if second != self._cur_second:
                if self._cur_second:
                    self._per_second.append((self._cur_second, self._cur_count))
                self._cur_second, self._cur_count = second, 0
            self._cur_count += 1

    def _flush_second(self, now: float) -> None:
        second = int(self._mono())
        with self._lock:
            if self._cur_second and second != self._cur_second:
                self._per_second.append((self._cur_second, self._cur_count))
                self._cur_second, self._cur_count = second, 0

    def _maybe_summary(self, now: float) -> Event | None:
        if (
            self._last_summary is not None
            and now - self._last_summary < self.settings.summary_seconds
        ):
            return None
        interval = now - self._last_summary if self._last_summary is not None else 0.0
        self._last_summary = now
        lost = self.missed_total
        observed, self._sum_observed = self._sum_observed, 0
        lost_delta, self._sum_lost = lost - self._sum_lost, lost
        per_s, _ = self.rates()
        ratio = lost_delta / (observed + lost_delta) if observed + lost_delta else 0.0
        return Event(
            time=now,
            source="exec",
            subject="exec",
            kind=KIND_SUMMARY,
            severity=Severity.SAFE,
            message="exec summary",
            fields={
                "observed": observed,
                "lost": lost_delta,
                "miss_ratio": round(ratio, 4),
                "interval": round(interval, 1),
                # execs per second, in the slot the poller fills with its fork rate
                "forks_per_s": round(per_s, 2),
                "tracked": len(self._recent),
            },
        )

    def _exec_event(self, obj: dict) -> Event:
        pid = int(obj["pid"])
        ppid = int(obj["ppid"])
        start = int(obj.get("start", 0))
        pstart = int(obj.get("pstart", 0))
        raw_argv = [str(a) for a in obj.get("argv", [])]
        filename = str(obj.get("filename", ""))
        exe = str(obj.get("exe", "")) or filename
        if not obj.get("exe"):
            self.exe_unresolved += 1
        if obj.get("trunc"):
            self.truncated += 1
        comm = str(obj.get("comm", ""))
        argv_text = redact_argv_text(raw_argv)
        exe_text = redact_argv_text([exe]) if exe else ""

        parent = self._recent.get(ppid)
        if parent is not None and parent.start != pstart:
            parent = None  # a reused pid, not the process that exec'd before
        parent_comm = parent.comm if parent is not None else str(obj.get("pcomm", ""))
        parent_exe = parent.exe if parent is not None else ""
        parent_argv = parent.argv if parent is not None and self.settings.parent_argv else ""
        parents = self._chain(parent, str(obj.get("pcomm", "")))

        self._remember(
            pid,
            _Parent(
                start=start,
                exe=exe_text,
                comm=clean(comm, 64),
                ppid=ppid,
                pstart=pstart,
                argv=argv_text if self.settings.parent_argv else "",
            ),
        )

        argv0 = clean(os.path.basename(raw_argv[0]), 256) if raw_argv else ""
        path0 = clean(raw_argv[0], 512) if raw_argv else ""
        facts = (
            {f"fact_{k}": v for k, v in self._facts.read(pid, raw_argv[0], exe).items()}
            if self.settings.facts and raw_argv
            else {}
        )
        return Event(
            time=float(obj["ts"]),
            source="exec",
            subject=str(pid),
            kind=KIND_EXEC,
            severity=Severity.SAFE,
            message=f"exec {shorten(clean(comm), 64)}",
            fields={
                "pid": pid,
                "ppid": ppid,
                "uid": int(obj.get("uid", -1)),
                "comm": clean(comm, 128),
                "starttime": start,
                "state": "",
                "exe": exe_text,
                "filename": clean(filename, 256),
                "argv0": argv0,
                "path0": path0,
                "argv": argv_text,
                "cwd": "",
                "cgroup": self._cgroups.path(int(obj.get("cgroup", 0))),
                "parent_comm": parent_comm,
                "parent_exe": parent_exe,
                "parents": parents,
                "parent_argv": parent_argv,
                "argv_truncated": int(bool(obj.get("trunc"))),
                **facts,
            },
        )

    def _chain(self, parent: _Parent | None, pcomm: str, depth: int = 3) -> str:
        """Ancestor labels (exe basename, else comm), nearest first, from the exec cache."""
        labels: list[str] = []
        node = parent
        if node is None:
            return clean(pcomm, 64)
        for _ in range(depth):
            if node is None:
                break
            labels.append(os.path.basename(node.exe) if node.exe else node.comm)
            nxt = self._recent.get(node.ppid)
            node = nxt if nxt is not None and nxt.start == node.pstart else None
        return " <- ".join(labels)

    def _remember(self, pid: int, entry: _Parent) -> None:
        self._recent[pid] = entry
        self._recent.move_to_end(pid)
        while len(self._recent) > self.settings.recent_exec_max:
            self._recent.popitem(last=False)


__all__ = ["HELPER_HINT", "UNAVAILABLE", "CgroupResolver", "EbpfSource"]

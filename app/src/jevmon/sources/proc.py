"""Read process facts from a `/proc`-like tree, tolerating races, permissions and odd bytes.

`RealProcReader` is the real implementation; the constructor takes `proc_root` so tests can point
it at a fake tree under a temp directory. Parsing lives in module-level pure functions so it can be
tested without touching a filesystem.

Robustness: any of `stat`, `cmdline`, `status`, `cgroup` may be missing (the process vanished) or
unreadable (permission); each accessor returns None then. `cmdline` bytes are kept raw here and
decoded by the caller, so invalid UTF-8 never raises.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ProcStat:
    pid: int
    comm: str
    state: str
    ppid: int
    starttime: int


@dataclass(frozen=True, slots=True)
class ProcInfo:
    pid: int
    comm: str
    state: str
    ppid: int
    starttime: int
    uid: int | None
    argv: tuple[bytes, ...]
    exe: str | None
    cwd: str | None
    cgroup: str | None


class ProcReader(Protocol):
    """Read-only view of a process table. Every accessor may return None on a race."""

    def pids(self) -> list[int]: ...

    def stat(self, pid: int) -> ProcStat | None: ...

    def cmdline(self, pid: int) -> tuple[bytes, ...] | None: ...

    def uid(self, pid: int) -> int | None: ...

    def cgroup(self, pid: int) -> str | None: ...

    def link(self, pid: int, name: str) -> str | None: ...

    def read_file(self, pid: int, name: str) -> bytes | None: ...

    def stat_path(self, pid: int, name: str) -> os.stat_result | None: ...

    def fork_count(self) -> int | None: ...


def parse_stat(data: bytes) -> ProcStat | None:
    """Parse `/proc/<pid>/stat`. `comm` may contain spaces and parens; split from the last ')'."""
    text = data.decode("ascii", "replace")
    lp = text.find("(")
    rp = text.rfind(")")
    if lp < 0 or rp <= lp:
        return None
    fields = text[rp + 1 :].split()
    if len(fields) < 20:
        return None
    try:
        return ProcStat(
            pid=int(text[:lp].strip()),
            comm=text[lp + 1 : rp],
            state=fields[0],
            ppid=int(fields[1]),
            starttime=int(fields[19]),
        )
    except ValueError:
        return None


def parse_uid(status: bytes) -> int | None:
    """First field of the `Uid:` line in `/proc/<pid>/status` (the real uid)."""
    for line in status.splitlines():
        if line.startswith(b"Uid:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
            return None
    return None


def parse_cmdline(data: bytes) -> tuple[bytes, ...]:
    """NUL-separated argv. Drops the single trailing empty element left by the terminating NUL."""
    parts = data.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    return tuple(parts)


def unit_from_cgroup(text: str) -> str | None:
    """Map a cgroup path to a systemd unit/scope name.

    A rootless podman container's processes sit under `libpod-<id>.scope/container/...`, where
    the inner components (`init.scope`, `nginx.service`) are container-local names - so the
    `libpod-...scope` component (the real host-level unit) wins when present. Otherwise the
    innermost `.service`/`.scope` component (scanning from the end) is returned.
    """
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        path = parts[2].strip()
        if not path or path == "/":
            continue
        comps = path.rstrip("/").split("/")
        for comp in comps:
            if comp.startswith("libpod-") and comp.endswith(".scope"):
                return comp
        for comp in reversed(comps):
            if comp.endswith((".service", ".scope")):
                return comp
    return None


def parse_fork_count(data: bytes) -> int | None:
    """The cumulative `processes` counter from `/proc/stat` (forks since boot)."""
    for line in data.splitlines():
        if line.startswith(b"processes"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


class RealProcReader:
    """Read a real or fake `/proc` tree. All failures degrade to None rather than raising."""

    def __init__(self, proc_root: str | Path = "/proc") -> None:
        self.root = Path(proc_root)

    def pids(self) -> list[int]:
        out: list[int] = []
        try:
            scan = os.scandir(self.root)
        except OSError:
            return out
        with scan:
            for entry in scan:
                if entry.name.isdigit():
                    out.append(int(entry.name))
        return out

    def _read(self, pid: int, name: str) -> bytes | None:
        try:
            return (self.root / str(pid) / name).read_bytes()
        except (OSError, ValueError):
            return None

    def stat(self, pid: int) -> ProcStat | None:
        data = self._read(pid, "stat")
        return parse_stat(data) if data is not None else None

    def cmdline(self, pid: int) -> tuple[bytes, ...] | None:
        data = self._read(pid, "cmdline")
        return parse_cmdline(data) if data is not None else None

    def uid(self, pid: int) -> int | None:
        data = self._read(pid, "status")
        return parse_uid(data) if data is not None else None

    def cgroup(self, pid: int) -> str | None:
        data = self._read(pid, "cgroup")
        return unit_from_cgroup(data.decode("utf-8", "replace")) if data is not None else None

    def link(self, pid: int, name: str) -> str | None:
        try:
            return os.readlink(self.root / str(pid) / name)
        except OSError:
            return None

    def read_file(self, pid: int, name: str) -> bytes | None:
        """Raw bytes of `/proc/<pid>/<name>` (status, stat, loginuid, uid_map, ...) or None."""
        return self._read(pid, name)

    def stat_path(self, pid: int, name: str) -> os.stat_result | None:
        """`os.stat` through a magic link such as `exe`, None when it vanished or is denied."""
        try:
            return os.stat(self.root / str(pid) / name)
        except OSError:
            return None

    def fork_count(self) -> int | None:
        data = self._read_top("stat")
        return parse_fork_count(data) if data is not None else None

    def _read_top(self, name: str) -> bytes | None:
        try:
            return (self.root / name).read_bytes()
        except (OSError, ValueError):
            return None

    def info(self, pid: int, stat: ProcStat | None = None) -> ProcInfo | None:
        """Read everything about a pid in one pass; None if it has vanished."""
        found = stat if stat is not None else self.stat(pid)
        if found is None:
            return None
        argv = self.cmdline(pid) or ()
        return ProcInfo(
            pid=found.pid,
            comm=found.comm,
            state=found.state,
            ppid=found.ppid,
            starttime=found.starttime,
            uid=self.uid(pid),
            argv=argv,
            exe=self.link(pid, "exe"),
            cwd=self.link(pid, "cwd"),
            cgroup=self.cgroup(pid),
        )

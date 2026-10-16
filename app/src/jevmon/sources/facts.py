"""Launch-time per-process facts: short `key: value` texts read from `/proc/<pid>` in one pass.

These are FACTS for the model to read, never verdicts: no names, paths or values are looked up
in a list and nothing here decides severity. Each fact is a short string; an unreadable or
vanished process gives `unknown`, never an exception. `[exec] facts = [...]` picks which are
read (empty = none, the default) and `[context] order` picks which are shown to the model.

| fact           | text                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| exe_state      | `normal`, `deleted` (exe link ends with "(deleted)"), `memfd`, `unknown`      |
| argv0_vs_exe   | `match` / `differs` by basename of argv[0] and of the exe link                |
| stdio          | type of fd 0, 1, 2: `tty`, `pipe`, `socket`, `file`, `null`, `other`          |
| session        | `tty=yes sid_leader=no loginuid=set`                                          |
| workdir        | the cwd link (redacted like any path)                                         |
| cgroup_path    | the tail of the cgroup path with long ids shortened                           |
| privileges     | `uid=1000 euid=0 capeff=0000000000000000 nnp=0 seccomp=0`                     |
| namespaces     | combined state, per-namespace `same`/`differs`, uid_map and NSpid hints       |
| exe_file       | owner uid, mode, setuid and world-writable bits of the executable file        |
"""

import os
import re
import stat as statmod
from collections.abc import Callable, Sequence

from jevmon.redact import redact_argv_text
from jevmon.sources.proc import ProcReader

FACT_NAMES = (
    "exe_state",
    "argv0_vs_exe",
    "stdio",
    "session",
    "workdir",
    "cgroup_path",
    "privileges",
    "namespaces",
    "exe_file",
)
NAMESPACES = ("mnt", "pid", "net", "user", "uts", "ipc", "cgroup", "time")
UNKNOWN = "unknown"
_LOGINUID_UNSET = 4294967295
_ID = re.compile(r"[0-9a-fA-F]{12,}")
_TRIVIAL_MAP = ("0", "0", "4294967295")


def validate(names: Sequence[str]) -> None:
    unknown = set(names) - set(FACT_NAMES)
    if unknown:
        raise ValueError(f"unknown fact(s): {', '.join(sorted(unknown))}")


def _field(stat_text: bytes, index: int) -> str | None:
    """Field `index` (0-based after the comm) of /proc/<pid>/stat: 0 state, 1 ppid, 2 pgrp,
    3 session, 4 tty_nr."""
    text = stat_text.decode("ascii", "replace")
    rp = text.rfind(")")
    parts = text[rp + 1 :].split() if rp >= 0 else []
    return parts[index] if index < len(parts) else None


def _status_value(status: bytes, key: bytes) -> list[bytes] | None:
    for line in status.splitlines():
        if line.startswith(key + b":"):
            return line.split()[1:]
    return None


def _exe_state(exe: str | None) -> str:
    if exe is None:
        return UNKNOWN
    if exe.startswith("/memfd:"):
        return "memfd"
    if exe.endswith(" (deleted)"):
        return "deleted"
    return "normal"


def _base(path: str) -> str:
    return os.path.basename(path.removesuffix(" (deleted)"))


def _fd_type(target: str | None) -> str:
    if target is None:
        return UNKNOWN
    if target == "/dev/null":
        return "null"
    if target.startswith("pipe:"):
        return "pipe"
    if target.startswith("socket:"):
        return "socket"
    if target.startswith(("/dev/pts/", "/dev/tty", "/dev/console")):
        return "tty"
    if target.startswith(("anon_inode:", "/memfd:")):
        return "other"
    if target.startswith("/"):
        return "file"
    return "other"


def _cgroup_path(data: bytes | None) -> str:
    if data is None:
        return UNKNOWN
    for line in data.decode("utf-8", "replace").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[2].strip() not in ("", "/"):
            comps = [c for c in parts[2].strip().split("/") if c]
            tail = "/".join(comps[-3:])
            return _ID.sub("<id>", tail)
    return "/"


def _privileges(status: bytes | None) -> str:
    if status is None:
        return UNKNOWN
    uid = _status_value(status, b"Uid") or []
    cap = _status_value(status, b"CapEff") or [b"?"]
    nnp = _status_value(status, b"NoNewPrivs") or [b"?"]
    sec = _status_value(status, b"Seccomp") or [b"?"]
    real = uid[0].decode() if uid else "?"
    eff = uid[1].decode() if len(uid) > 1 else "?"
    return (
        f"uid={real} euid={eff} capeff={cap[0].decode()} nnp={nnp[0].decode()} "
        f"seccomp={sec[0].decode()}"
    )


def _exe_file(st: os.stat_result | None) -> str:
    if st is None:
        return UNKNOWN
    mode = statmod.S_IMODE(st.st_mode)
    return (
        f"owner={st.st_uid} mode={mode:04o} "
        f"setuid={'yes' if mode & statmod.S_ISUID else 'no'} "
        f"world_writable={'yes' if mode & statmod.S_IWOTH else 'no'}"
    )


class FactReader:
    """Reads the enabled facts of one pid. One instance per source; holds the baselines."""

    def __init__(
        self,
        reader: ProcReader,
        names: Sequence[str],
        self_pid: int | None = None,
        redactor: Callable[[list[str]], str] = redact_argv_text,
    ) -> None:
        validate(names)
        self.reader = reader
        self.names = tuple(n for n in FACT_NAMES if n in names)
        self.self_pid = self_pid if self_pid is not None else os.getpid()
        self._redact = redactor
        self._baseline: dict[str, tuple[str, str | None]] | None = None

    def _ns_baseline(self) -> dict[str, tuple[str, str | None]]:
        """Namespace link of init, else of this process, per namespace (read once)."""
        if self._baseline is None:
            base: dict[str, tuple[str, str | None]] = {}
            for ns in NAMESPACES:
                init = self.reader.link(1, f"ns/{ns}")
                if init is not None:
                    base[ns] = ("init", init)
                else:
                    base[ns] = ("daemon", self.reader.link(self.self_pid, f"ns/{ns}"))
            self._baseline = base
        return self._baseline

    def read(self, pid: int, argv0: str, exe: str | None) -> dict[str, str]:
        """The enabled facts for `pid`; `argv0` and `exe` are what the poller already read."""
        out: dict[str, str] = {}
        r = self.reader
        names = self.names
        status = r.read_file(pid, "status") if {"privileges", "namespaces"} & set(names) else None
        for name in names:
            out[name] = self._one(name, pid, argv0, exe, status)
        return out

    def _one(self, name: str, pid: int, argv0: str, exe: str | None, status: bytes | None) -> str:
        r = self.reader
        try:
            if name == "exe_state":
                return _exe_state(exe)
            if name == "argv0_vs_exe":
                if not argv0 or exe is None:
                    return UNKNOWN
                return "match" if os.path.basename(argv0) == _base(exe) else "differs"
            if name == "stdio":
                return " ".join(f"{fd}={_fd_type(r.link(pid, f'fd/{fd}'))}" for fd in (0, 1, 2))
            if name == "session":
                return self._session(pid)
            if name == "workdir":
                cwd = r.link(pid, "cwd")
                return self._redact([cwd]).strip() if cwd else UNKNOWN
            if name == "cgroup_path":
                return _cgroup_path(r.read_file(pid, "cgroup"))
            if name == "privileges":
                return _privileges(status)
            if name == "namespaces":
                return self._namespaces(pid, status)
            if name == "exe_file":
                return _exe_file(r.stat_path(pid, "exe"))
        except (OSError, ValueError, IndexError):
            return UNKNOWN
        return UNKNOWN

    def _session(self, pid: int) -> str:
        r = self.reader
        data = r.read_file(pid, "stat")
        if data is None:
            return UNKNOWN
        sid, tty = _field(data, 3), _field(data, 4)
        if sid is None or tty is None:
            return UNKNOWN
        login = r.read_file(pid, "loginuid")
        if login is None:
            login_text = UNKNOWN
        else:
            try:
                login_text = "unset" if int(login.strip()) == _LOGINUID_UNSET else "set"
            except ValueError:
                login_text = UNKNOWN
        return (
            f"tty={'yes' if tty != '0' else 'no'} "
            f"sid_leader={'yes' if sid == str(pid) else 'no'} loginuid={login_text}"
        )

    def _namespaces(self, pid: int, status: bytes | None) -> str:
        r = self.reader
        base = self._ns_baseline()
        per: dict[str, str] = {}
        for ns in NAMESPACES:
            here = r.link(pid, f"ns/{ns}")
            _who, there = base[ns]
            per[ns] = (
                UNKNOWN
                if here is None or there is None
                else ("same" if here == there else "differs")
            )
        differing = [ns for ns, v in per.items() if v == "differs"]
        known = [v for v in per.values() if v != UNKNOWN]
        if not known:
            return UNKNOWN
        combined = f"differs({','.join(differing)})" if differing else "host"
        parts = [combined]
        parts.append("[" + " ".join(f"{ns}={v}" for ns, v in per.items()) + "]")
        uid_map = r.read_file(pid, "uid_map")
        if uid_map is not None:
            fields = uid_map.split()
            trivial = tuple(f.decode() for f in fields[:3]) == _TRIVIAL_MAP
            parts.append("uid_map=" + ("identity" if trivial else "remapped"))
        if status is not None:
            nspid = _status_value(status, b"NSpid")
            if nspid and len(nspid) > 1:
                parts.append("nspid=" + ">".join(p.decode() for p in nspid))
        return " ".join(parts)

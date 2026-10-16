"""Structural facts read from the running system (no names): file ownership, mode, mtime, mount.

`StructuralFacts` is the small record the context builder renders. `RealSystemFacts` reads it with
`os.stat` and `/proc/self/mountinfo`; tests inject their own `SystemFacts`. The mount parsing is a
pure function so it can be tested without a filesystem.
"""

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Mount:
    point: str
    fstype: str
    options: frozenset[str]


@dataclass(frozen=True, slots=True)
class StructuralFacts:
    readable: bool = False
    writable_dir: bool = False
    owned_by_nonroot: bool = False
    recently_written: bool = False
    mount_type: str = "unknown"
    on_tmpfs_or_noexec: bool = False


class SystemFacts(Protocol):
    def inspect(self, exe: str, now: float, recent_seconds: float) -> StructuralFacts: ...


def parse_mounts(text: str) -> list[Mount]:
    """Parse `/proc/self/mountinfo` lines into mount point, fstype and the combined options."""
    mounts: list[Mount] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            sep = fields.index("-")
        except ValueError:
            continue
        if sep + 1 >= len(fields):
            continue
        fstype = fields[sep + 1]
        options = set(fields[5].split(","))
        if sep + 3 < len(fields):
            options.update(fields[sep + 3].split(","))
        mounts.append(Mount(fields[4], fstype, frozenset(options)))
    return mounts


def mount_for_path(path: str, mounts: list[Mount]) -> Mount | None:
    """The longest mount point that contains `path` (mount points nest)."""
    best: Mount | None = None
    for mount in mounts:
        point = mount.point.rstrip("/") or "/"
        contains = path == point or path.startswith(point + "/") or point == "/"
        if contains and (best is None or len(mount.point) > len(best.point)):
            best = mount
    return best


def _writable(mode: int) -> bool:
    return bool(mode & (stat.S_IWOTH | stat.S_IWGRP))


class RealSystemFacts:
    """Read structural facts from the real filesystem and mount table, cached by mtime."""

    def __init__(self, mountinfo: str | Path = "/proc/self/mountinfo") -> None:
        self.mountinfo = Path(mountinfo)
        self._mounts: list[Mount] | None = None
        self._mounts_mtime: float = -1.0

    def _load_mounts(self) -> list[Mount]:
        try:
            mtime = self.mountinfo.stat().st_mtime
        except OSError:
            return self._mounts or []
        if self._mounts is None or mtime != self._mounts_mtime:
            try:
                text = self.mountinfo.read_text(encoding="utf-8", errors="replace")
                self._mounts = parse_mounts(text)
                self._mounts_mtime = mtime
            except OSError:
                return self._mounts or []
        return self._mounts

    def inspect(self, exe: str, now: float, recent_seconds: float) -> StructuralFacts:
        if not exe:
            return StructuralFacts()
        try:
            st = os.stat(exe)
            parent = os.stat(os.path.dirname(exe) or "/")
        except OSError:
            return StructuralFacts()
        mount = mount_for_path(exe, self._load_mounts())
        fstype = mount.fstype if mount else "unknown"
        on_tmpfs = bool(mount and (mount.fstype == "tmpfs" or "noexec" in mount.options))
        written = max(st.st_mtime, st.st_ctime)
        return StructuralFacts(
            readable=True,
            writable_dir=_writable(parent.st_mode),
            owned_by_nonroot=st.st_uid != 0,
            recently_written=(now - written) <= recent_seconds,
            mount_type=fstype,
            on_tmpfs_or_noexec=on_tmpfs,
        )

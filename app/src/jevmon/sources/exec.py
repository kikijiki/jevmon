"""Tier 0/0+ exec source: poll a `/proc` tree, diff pids, and estimate missed execs.

Every scan reads the pid list and, for each pid, its `stat`, `cmdline` and `exe` link. A pid is a
new exec when:

- it is unseen, or
- it was seen with a different start time (pid reuse), or
- it was seen with the same start time but its cmdline or exe changed (an exec in place).

Tier 0+ reads the cumulative `processes` counter in `/proc/stat`: `forks_delta - new_pids_observed`
(floored at 0) estimates forks that were never observed. The estimate is an upper bound — the
counter counts forks, not execs, and a fork without exec is invisible — and on the first scan the
delta is treated as 0 so the pre-existing process table is not reported as a backlog.

A hard CPU budget widens the poll interval when a scan costs more than `cpu_budget` of one core
(measured with `thread_time`: the poller's own thread, not the model's). Work per scan is bounded: at most `max_pids` are inspected and at
most `max_tracked` pids are remembered (least-recently-seen evicted). Nothing is persisted here;
the source only emits `Event`s. `scan_once` returns a `ScanResult` for tests and callers.

Two clocks, on purpose. Event timestamps (`clock`, wall time) are persisted and later compared
with file mtimes, retention cutoffs and counts written by earlier runs - even from an earlier
boot - so they must be epoch time. Durations and cadence (`mono`, monotonic) - scan length, the
fork-rate delta, the summary throttle - must not jump when the wall clock is stepped. Stamping
events with the monotonic clock mixed the two domains: `first_seen_age` went negative by hours
after a reboot (schema 2 stores, fixed in schema 3).
"""

import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass

from jevmon.events import Event, Severity
from jevmon.redact import redact_argv_text
from jevmon.sanitize import clean, shorten
from jevmon.sources.facts import FACT_NAMES, FactReader
from jevmon.sources.proc import ProcReader, ProcStat, RealProcReader

KIND_EXEC = "exec"
KIND_SUMMARY = "summary"


SOURCES = ("ebpf", "proc", "replay")


@dataclass(frozen=True, slots=True)
class ExecSettings:
    """Source knobs. Defaults are the shipped `[exec]` values; see `config.py`.

    `source` picks the event source: `ebpf` reads the privileged helper's exec stream (the
    shipped default, `sources/ebpf.py`), `proc` polls a `/proc` tree without privileges,
    `replay` re-emits a recorded JSON-lines stream from `replay_file` at `interval` seconds per
    event. Replay keeps the rest of the pipeline identical, which is what makes demos, tests and
    screenshot runs reproducible without a real process table.
    """

    proc_root: str = "/proc"
    interval: float = 0.05
    min_interval: float = 0.05
    max_interval: float = 5.0
    cpu_budget: float = 0.03
    max_pids: int = 20_000
    max_tracked: int = 5_000
    summary_seconds: float = 30.0
    source: str = "ebpf"
    replay_file: str = ""
    # Also read each parent's command line (one more /proc read per exec). Off by default: the
    # `parent_cmd` context segment did not help the judge.
    parent_argv: bool = False
    # Launch-time per-process facts to read (names in `jevmon.sources.facts.FACT_NAMES`); empty =
    # none. They are facts for the model to read, never rules. One extra pass of /proc reads per
    # new exec; the context shows the ones named in `[context] order`.
    facts: tuple[str, ...] = ()
    # ebpf source only (ignored by the others): the helper's socket, the reconnect backoff bounds
    # in seconds, and how many recent execs are remembered to name a launch's parent.
    helper_socket: str = "/run/jevmon/exec.sock"
    helper_retry_min: float = 0.5
    helper_retry_max: float = 10.0
    recent_exec_max: int = 4096
    # ebpf source: the reader's CPU ceiling as a fraction of one core (thread time); above it the
    # source stops reading and the helper counts what it could not deliver. 0 = no ceiling.
    helper_cpu_budget: float = 0.5
    # Where `jevmon start` finds the helper binary; empty = `jevmon-exec` on PATH, else the build
    # under app/helper/result.
    helper_path: str = ""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {', '.join(SOURCES)}, got {self.source!r}")
        unknown = set(self.facts) - set(FACT_NAMES)
        if unknown:
            raise ValueError(f"unknown fact(s): {', '.join(sorted(unknown))}")
        if not self.helper_socket:
            raise ValueError("helper_socket must not be empty")
        if not 0 < self.helper_retry_min <= self.helper_retry_max:
            raise ValueError("need 0 < helper_retry_min <= helper_retry_max")
        if not 0 <= self.helper_cpu_budget <= 1:
            raise ValueError("helper_cpu_budget must be between 0 and 1")
        if self.recent_exec_max < 1:
            raise ValueError("recent_exec_max must be positive")
        if self.source == "replay" and not self.replay_file:
            raise ValueError("source = 'replay' needs replay_file")


@dataclass(frozen=True, slots=True)
class ScanResult:
    time: float
    events: tuple[Event, ...]
    observed: int
    forks_delta: int
    estimated_missed: int
    miss_ratio: float
    effective_interval: float
    scan_seconds: float
    tracked: int
    forks_per_s: float
    summary: Event | None


@dataclass(slots=True)
class _Tracked:
    starttime: int
    argv_key: int
    exe: str | None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


class ExecSource:
    """Poll a `ProcReader`. `emit` receives every event the source produces."""

    def __init__(
        self,
        reader: ProcReader | None = None,
        settings: ExecSettings | None = None,
        emit: Callable[[Event], None] | None = None,
        clock: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
        cpu_clock: Callable[[], float] = time.thread_time,
    ) -> None:
        self.settings = settings or ExecSettings()
        self.reader = reader if reader is not None else RealProcReader(self.settings.proc_root)
        self._emit = emit if emit is not None else (lambda event: None)
        self._clock = clock
        self._mono = mono
        self._cpu_clock = cpu_clock
        self._base_interval = self.settings.interval
        self._interval = self.settings.interval
        self._facts = FactReader(self.reader, self.settings.facts)
        self._tracked: OrderedDict[int, _Tracked] = OrderedDict()
        self._last_forks: int | None = None
        self._last_scan_tick: float | None = None
        self._scan_times: deque[float] = deque(maxlen=200)
        self._last_summary_tick: float | None = None
        self._sum_observed = 0
        self._sum_missed = 0
        self.observed_total = 0
        self.missed_total = 0

    @property
    def effective_interval(self) -> float:
        return self._interval

    @property
    def tracked(self) -> int:
        return len(self._tracked)

    @property
    def miss_ratio(self) -> float:
        """Cumulative missed / (observed + missed); 0.0 before the first baseline scan."""
        denominator = self.observed_total + self.missed_total
        return self.missed_total / denominator if denominator else 0.0

    def apply_settings(self, settings: ExecSettings) -> None:
        """Adopt new source knobs, clamping the current interval into the new bounds."""
        self.settings = settings
        self._facts = FactReader(self.reader, settings.facts)
        self._base_interval = settings.interval
        self._interval = min(max(self._interval, settings.min_interval), settings.max_interval)

    def scan_once(self) -> ScanResult:
        now = self._clock()  # wall: the stamp carried by every event this scan
        tick = self._mono()  # monotonic: durations and cadence only
        cpu0 = self._cpu_clock()
        forks = self.reader.fork_count()
        pids = sorted(self.reader.pids())
        if len(pids) > self.settings.max_pids:
            pids = pids[: self.settings.max_pids]

        events: list[Event] = []
        observed = 0
        for pid in pids:
            stat = self.reader.stat(pid)
            if stat is None:
                continue
            argv = self.reader.cmdline(pid)
            if argv is None:  # vanished between stat and cmdline
                continue
            exe = self.reader.link(pid, "exe")
            argv_key = hash(argv)
            prev = self._tracked.get(pid)
            new_pid = prev is None or prev.starttime != stat.starttime
            changed = not new_pid and (prev.argv_key != argv_key or prev.exe != exe)
            if new_pid or changed:
                self._tracked[pid] = _Tracked(stat.starttime, argv_key, exe)
                self._tracked.move_to_end(pid)
                if new_pid:
                    observed += 1
                if stat.state != "Z" and argv:
                    events.append(self._make_event(now, stat, argv, exe))
            else:
                self._tracked.move_to_end(pid)
        self._evict()

        cpu_seconds = self._cpu_clock() - cpu0
        now_tick = self._mono()
        scan_seconds = now_tick - tick
        self._scan_times.append(scan_seconds)
        self._adapt_interval(cpu_seconds, scan_seconds)

        forks_delta, missed = self._miss(observed, forks)
        dt = now_tick - self._last_scan_tick if self._last_scan_tick is not None else 0.0
        self._last_scan_tick = now_tick
        forks_per_s = (forks_delta / dt) if dt > 0 else 0.0
        self._sum_observed += observed
        self._sum_missed += missed
        self.observed_total += observed
        self.missed_total += missed
        denominator = self._sum_observed + self._sum_missed
        ratio = (self._sum_missed / denominator) if denominator else 0.0
        summary = self._maybe_summary(now, now_tick, forks_per_s, ratio)
        return ScanResult(
            time=now,
            events=tuple(events),
            observed=observed,
            forks_delta=forks_delta,
            estimated_missed=missed,
            miss_ratio=ratio,
            effective_interval=self._interval,
            scan_seconds=scan_seconds,
            tracked=len(self._tracked),
            forks_per_s=forks_per_s,
            summary=summary,
        )

    def run(self, stop: threading.Event) -> None:
        """Poll until `stop` is set, emitting events and summaries."""
        while not stop.is_set():
            result = self.scan_once()
            for event in result.events:
                self._emit(event)
            if result.summary is not None:
                self._emit(result.summary)
            stop.wait(self._interval)

    # -- internals -------------------------------------------------------------------------

    def _evict(self) -> None:
        while len(self._tracked) > self.settings.max_tracked:
            self._tracked.popitem(last=False)

    def _miss(self, observed: int, forks: int | None) -> tuple[int, int]:
        if forks is None:
            return 0, 0
        if self._last_forks is None:
            self._last_forks = forks
            return 0, 0  # first scan: no baseline, no spurious backlog
        delta = max(0, forks - self._last_forks)  # tolerate a counter reset/counter read race
        self._last_forks = forks
        return delta, max(0, delta - observed)

    def _adapt_interval(self, cpu_seconds: float, scan_seconds: float) -> None:
        # Budget is CPU over the whole polling cycle (sleep + scan), not over the scan alone;
        # otherwise a CPU-bound scan always looks 100% busy and the interval pins at the maximum.
        cycle = max(self._interval + scan_seconds, 1e-6)
        utilization = cpu_seconds / cycle
        budget = self.settings.cpu_budget
        if utilization > budget > 0:
            ratio = utilization / budget
            self._interval = min(
                self.settings.max_interval, max(self._interval * ratio, self._interval + 0.005)
            )
        elif utilization < budget * 0.5:
            self._interval = max(self._base_interval, self._interval * 0.9)

    def _maybe_summary(
        self, now: float, tick: float, forks_per_s: float, miss_ratio: float
    ) -> Event | None:
        if (
            self._last_summary_tick is not None
            and tick - self._last_summary_tick < self.settings.summary_seconds
        ):
            return None
        self._last_summary_tick = tick
        durations = list(self._scan_times)
        summary = Event(
            time=now,
            source="exec",
            subject="exec",
            kind=KIND_SUMMARY,
            severity=Severity.SAFE,
            message="exec scan summary",
            fields={
                "observed": self._sum_observed,
                "estimated_missed": self._sum_missed,
                "miss_ratio": round(miss_ratio, 4),
                "effective_interval": round(self._interval, 4),
                "scan_p50": round(_percentile(durations, 0.50), 6),
                "scan_p95": round(_percentile(durations, 0.95), 6),
                "tracked": len(self._tracked),
                "forks_per_s": round(forks_per_s, 2),
            },
        )
        self._sum_observed = 0
        self._sum_missed = 0
        return summary

    def _parent(self, ppid: int) -> tuple[str, str]:
        if ppid <= 0:
            return "", ""
        stat = self.reader.stat(ppid)
        comm = clean(stat.comm, 64) if stat is not None else ""
        exe = self.reader.link(ppid, "exe") or ""
        return comm, exe

    def _parent_argv(self, ppid: int) -> str:
        """The parent's redacted command line (data, so the model can read what spawned this)."""
        if ppid <= 0:
            return ""
        raw = self.reader.cmdline(ppid)
        if not raw:
            return ""
        return redact_argv_text([os.fsdecode(arg) for arg in raw])

    def _parent_chain(self, ppid: int, depth: int = 3) -> str:
        """A few ancestor labels (basename of exe, else comm), nearest first. Data, not a rule."""
        labels: list[str] = []
        current = ppid
        for _ in range(depth):
            if current <= 0:
                break
            stat = self.reader.stat(current)
            if stat is None:
                break
            exe = self.reader.link(current, "exe")
            labels.append(os.path.basename(exe) if exe else clean(stat.comm, 64))
            current = stat.ppid
        return " <- ".join(labels)

    def _make_event(
        self, now: float, stat: ProcStat, argv: tuple[bytes, ...], exe: str | None
    ) -> Event:
        decoded = [os.fsdecode(arg) for arg in argv]
        uid = self.reader.uid(stat.pid)
        parent_comm, parent_exe = self._parent(stat.ppid)
        parents = self._parent_chain(stat.ppid)
        parent_argv = self._parent_argv(stat.ppid) if self.settings.parent_argv else ""
        facts = (
            {f"fact_{k}": v for k, v in self._facts.read(stat.pid, decoded[0], exe).items()}
            if self.settings.facts and decoded
            else {}
        )
        argv0 = clean(os.path.basename(decoded[0]), 256) if decoded else ""
        path0 = clean(decoded[0], 512) if decoded else ""
        return Event(
            time=now,
            source="exec",
            subject=str(stat.pid),
            kind=KIND_EXEC,
            severity=Severity.SAFE,
            message=f"exec {shorten(clean(stat.comm), 64)}",
            fields={
                "pid": stat.pid,
                "ppid": stat.ppid,
                "uid": uid if uid is not None else -1,
                "comm": clean(stat.comm, 128),
                "starttime": stat.starttime,
                "state": stat.state,
                "exe": redact_argv_text([exe]) if exe else "",
                "argv0": argv0,
                "path0": path0,
                "argv": redact_argv_text(decoded),
                "cwd": redact_argv_text([self.reader.link(stat.pid, "cwd") or ""]).strip(),
                "cgroup": self.reader.cgroup(stat.pid) or "",
                "parent_comm": parent_comm,
                "parent_exe": parent_exe,
                "parents": parents,
                "parent_argv": parent_argv,
                **facts,
            },
        )

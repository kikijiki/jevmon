"""ExecSource against a fake /proc tree under tmp_path: new pids, reuse, exec-in-place, races."""

import os
import time
from collections import deque
from pathlib import Path

from jevmon.sources.exec import ExecSettings, ExecSource
from jevmon.sources.proc import RealProcReader


def stat_line(pid: int, comm: str, state: str, ppid: int, starttime: int) -> str:
    rest = [state, str(ppid), *(["0"] * 17), str(starttime)]
    return f"{pid} ({comm}) " + " ".join(rest) + "\n"


class FakeProc:
    """Builds and mutates a fake /proc tree."""

    def __init__(self, root: Path, forks: int = 1000) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.set_forks(forks)

    def set_forks(self, n: int) -> None:
        (self.root / "stat").write_text(f"cpu  1 2 3\nprocesses {n}\nprocs_running 2\n")

    def add(
        self,
        pid: int,
        argv: list[str] | None = None,
        comm: str = "proc",
        ppid: int = 1,
        starttime: int = 100,
        uid: int = 1000,
        state: str = "S",
        cgroup: str | None = None,
        exe: str | None = None,
    ) -> Path:
        d = self.root / str(pid)
        d.mkdir(exist_ok=True)
        (d / "stat").write_text(stat_line(pid, comm, state, ppid, starttime))
        data = b"".join(a.encode() + b"\0" for a in (argv or []))
        (d / "cmdline").write_bytes(data)
        (d / "status").write_text(f"Name:\t{comm}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
        if cgroup:
            (d / "cgroup").write_text(cgroup)
        if exe:
            os.symlink(exe, d / "exe")
        return d

    def remove(self, pid: int) -> None:
        for child in (self.root / str(pid)).iterdir():
            child.unlink()
        (self.root / str(pid)).rmdir()


def source(proc: FakeProc, **overrides) -> ExecSource:
    overrides.setdefault("summary_seconds", 3600)
    settings = ExecSettings(**overrides)
    return ExecSource(RealProcReader(proc.root), settings)


def test_new_pid_emits_exec_event_with_fields(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    cg = "0::/user.slice/user-1000.slice/agent.scope\n"
    proc.add(100, ["/usr/bin/ls", "-la"], comm="ls", uid=1000, cgroup=cg, exe="/usr/bin/ls")
    res = source(proc).scan_once()
    assert res.observed == 1 and res.estimated_missed == 0  # first scan: no backlog
    (event,) = res.events
    assert event.kind == "exec" and event.subject == "100"
    f = event.fields
    assert f["comm"] == "ls" and f["uid"] == 1000 and f["cgroup"] == "agent.scope"
    assert f["exe"] == "/usr/bin/ls" and f["argv"] == "/usr/bin/ls -la"


def test_first_scan_with_existing_table_is_not_a_backlog(tmp_path):
    proc = FakeProc(tmp_path / "proc", forks=999_999)
    for pid in range(200, 260):
        proc.add(pid, [f"/bin/p{pid}"])
    res = source(proc).scan_once()
    assert res.observed == 60 and res.estimated_missed == 0 and res.miss_ratio == 0.0


def test_miss_estimate_after_first_scan(tmp_path):
    proc = FakeProc(tmp_path / "proc", forks=1000)
    proc.add(100, ["/bin/a"])
    src = source(proc)
    src.scan_once()
    proc.add(101, ["/bin/b"])
    proc.set_forks(1005)  # one observed, five forks: four missed
    res = src.scan_once()
    assert res.observed == 1 and res.forks_delta == 5 and res.estimated_missed == 4
    assert res.miss_ratio == 4 / 5


def test_pid_reuse_detected_by_starttime(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    proc.add(100, ["/bin/old"], starttime=100)
    src = source(proc)
    src.scan_once()
    proc.add(100, ["/bin/new"], comm="new", starttime=200)
    res = src.scan_once()
    assert res.observed == 1
    assert [e.fields["argv"] for e in res.events] == ["/bin/new"]


def test_exec_in_place_when_cmdline_changes(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    proc.add(100, ["/bin/sleep", "10"], starttime=100)
    src = source(proc)
    src.scan_once()
    proc.add(100, ["/bin/bash", "-c", "id"], comm="bash", starttime=100)  # same starttime
    res = src.scan_once()
    assert res.observed == 0 and len(res.events) == 1
    assert res.events[0].fields["argv"] == "/bin/bash -c id"


def test_no_event_when_nothing_changes(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    proc.add(100, ["/bin/sleep", "10"])
    src = source(proc)
    src.scan_once()
    res = src.scan_once()
    assert res.events == () and res.observed == 0


def test_vanishing_process_is_skipped(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    d = proc.add(100, ["/bin/x"])
    (d / "cmdline").unlink()  # stat ok, cmdline gone: race
    res = source(proc).scan_once()
    assert res.events == ()


def test_kernel_thread_and_zombie_are_tracked_not_emitted(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    proc.add(2, [], comm="kthreadd", ppid=0)  # empty cmdline
    proc.add(200, [], comm="defunct", ppid=1, state="Z")  # zombie
    res = source(proc).scan_once()
    assert res.events == ()
    assert res.tracked == 2
    assert res.observed == 2


def test_tracked_is_bounded(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    for pid in range(10, 20):
        proc.add(pid, [f"/bin/p{pid}"])
    res = source(proc, max_tracked=3).scan_once()
    assert res.tracked == 3


def test_cpu_budget_widens_interval(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    proc.add(100, ["/bin/x"])
    mono = deque([10.0, 10.01])  # 10 ms scan on the monotonic clock
    cpu = deque([1.0, 1.10])  # 100 ms of CPU -> 10x over a 3% budget
    src = ExecSource(
        RealProcReader(proc.root),
        ExecSettings(summary_seconds=3600, cpu_budget=0.03),
        mono=lambda: mono.popleft(),
        cpu_clock=lambda: cpu.popleft(),
    )
    res = src.scan_once()
    assert abs(res.scan_seconds - 0.01) < 1e-9
    assert res.effective_interval > 0.05


def test_events_and_summary_are_stamped_with_wall_clock(tmp_path):
    """Event times are epoch seconds, not the monotonic boot clock: they are persisted, shown
    to the model (hour, first_seen_age) and compared with file mtimes and earlier runs."""
    proc = FakeProc(tmp_path / "proc")
    proc.add(100, ["/usr/bin/ls", "-la"], exe="/usr/bin/ls")
    src = source(proc, summary_seconds=0)
    before = time.time()
    res = src.scan_once()
    after = time.time()
    (event,) = res.events
    assert before <= event.time <= after
    assert res.time == event.time
    assert res.summary is not None and before <= res.summary.time <= after


def test_summary_event_is_emitted_and_carries_the_numbers(tmp_path):
    proc = FakeProc(tmp_path / "proc", forks=1000)
    proc.add(100, ["/bin/a"])
    src = source(proc, summary_seconds=0)  # a summary every scan
    res = src.scan_once()
    assert res.summary is not None
    f = res.summary.fields
    assert res.summary.kind == "summary"
    assert f["observed"] == 1 and f["estimated_missed"] == 0
    assert f["tracked"] == 1 and "scan_p95" in f and "effective_interval" in f

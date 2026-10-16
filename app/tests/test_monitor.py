"""Monitor: the shared runtime drives source -> context -> judge -> store and honours pause."""

import time
from pathlib import Path

from jevmon.config import load_config
from jevmon.core import SqliteStore
from jevmon.laya.decider import Decision
from jevmon.monitor import Monitor


class FakeDecider:
    def __init__(self, label: str = "crit", confidence: float = 0.95) -> None:
        self.label = label
        self.confidence = confidence
        self.loaded = False
        self.calls = 0

    def load(self) -> None:
        self.loaded = True

    @property
    def ready(self) -> bool:
        return True

    @property
    def status(self) -> str:
        return "fake-ready"

    def decide(self, state: str) -> Decision:
        self.calls += 1
        time.sleep(0.002)  # keep ms_per_context above the status rounding
        return Decision(self.label, self.confidence, 0.0, False, "fake verdict")


def stat_line(pid: int, comm: str, state: str, ppid: int, starttime: int) -> str:
    rest = [state, str(ppid), *(["0"] * 17), str(starttime)]
    return f"{pid} ({comm}) " + " ".join(rest) + "\n"


def write_proc(root: Path, pid: int, argv: list[str], *, starttime: int = 100) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "stat").write_text("cpu 1 2 3\nprocesses 1000\n")
    d = root / str(pid)
    d.mkdir(exist_ok=True)
    (d / "stat").write_text(stat_line(pid, "proc", "S", 1, starttime))
    (d / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))
    (d / "status").write_text("Name:\tproc\nUid:\t1000\t1000\t1000\t1000\n")


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _paused_event(store: SqliteStore) -> bool:
    rows = store.query_events(limit=100)
    return any(int(event.fields.get("judged", -1)) == 0 for _, event, _ in rows)


def test_monitor_judges_then_records_paused_contexts(tmp_path):
    root = tmp_path / "proc"
    write_proc(root, 100, ["/bin/sleep", "9"])
    config = load_config()
    store = SqliteStore(tmp_path / "e.db")
    decider = FakeDecider()
    monitor = Monitor(config, store, proc_root=str(root), decider=decider)
    monitor.start()
    try:
        assert wait_for(lambda: store.count_events() >= 1), "no verdict stored"
        assert decider.loaded and decider.calls >= 1
        status = monitor.status()
        assert status["judged"] >= 1 and status["paused"] is False and status["running"] is True
        assert status["avg_batch"] >= 1.0 and status["ms_per_context"] > 0.0

        monitor.pause()
        assert monitor.status()["paused"] is True
        write_proc(root, 100, ["/bin/bash", "-c", "id"])  # exec in place: same starttime
        assert wait_for(lambda: _paused_event(store)), "no paused context recorded"

        monitor.resume()
        assert monitor.status()["paused"] is False
    finally:
        monitor.stop()
        store.close()


def test_monitor_status_before_start_is_safe(tmp_path):
    store = SqliteStore(tmp_path / "e.db")
    try:
        monitor = Monitor(load_config(), store, decider=FakeDecider())
        status = monitor.status()
        assert status["running"] is False and status["judged"] == 0
        assert status["model_status"] == "fake-ready"
        assert status["avg_batch"] == 0.0 and status["ms_per_context"] == 0.0
        assert status["queue_age"] == 0.0
    finally:
        store.close()


def test_monitor_status_reports_judge_device_and_dtype(tmp_path):
    store = SqliteStore(tmp_path / "e.db")
    try:
        monitor = Monitor(load_config(), store, decider=FakeDecider())
        status = monitor.status()
        assert status["judge_device"] == "" and status["judge_dtype"] == ""  # nothing loaded
    finally:
        store.close()


def test_build_decider_wires_device_and_dtype_from_judge_config(tmp_path):
    from jevmon.monitor import build_decider

    default = build_decider(load_config())
    assert (default.config.device, default.config.dtype) == ("auto", "auto")
    assert default.config.threads_gpu == 2 and default.config.threads == 2
    path = tmp_path / "config.toml"
    path.write_text("[judge]\ndevice = 'gpu'\ndtype = 'fp16'\n")
    decider = build_decider(load_config(path))
    assert (decider.config.device, decider.config.dtype) == ("gpu", "fp16")

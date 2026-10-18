"""The ebpf exec source against a fake helper on a unix socket. Nothing here needs root."""

import contextlib
import json
import os
import socket
import threading
import time
from dataclasses import replace

import pytest

from jevmon.config import ConfigError, load_config
from jevmon.core import BufferedCounts, SqliteStore, StoreSink
from jevmon.detect.fake import FakeDecider
from jevmon.detect.judge import JudgeSettings
from jevmon.events import Event, Severity
from jevmon.monitor import Monitor
from jevmon.sources.ebpf import HELPER_HINT, UNAVAILABLE, CgroupResolver, EbpfSource
from jevmon.sources.exec import ExecSettings

HELLO = {"t": "hello", "v": 1, "helper": "fake", "version": "0", "exe_source": "kernel"}


def exec_line(pid=100, ppid=50, argv=("/bin/ls", "-l"), **over) -> dict:
    obj = {
        "t": "exec", "seq": pid, "ts": 1_700_000_000.0 + pid, "mono": pid, "start": pid * 10,
        "pid": pid, "ppid": ppid, "pstart": ppid * 10, "uid": 1000, "euid": 1000, "gid": 1000,
        "cgroup": 7, "argc": len(argv), "flags": 0, "comm": "ls", "pcomm": "bash",
        "exe": "/nix/store/x-coreutils/bin/ls", "filename": "ls", "argv": list(argv), "trunc": 0,
    }  # fmt: skip
    obj.update(over)
    return obj


class FakeHelper:
    """Serves scripted lines to whoever connects; can be stopped and restarted."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.server: socket.socket | None = None
        self.conn: socket.socket | None = None
        self.accepted = threading.Event()

    def listen(self) -> None:
        self.server = socket.socket(socket.AF_UNIX)
        self.server.bind(self.path)
        self.server.listen(1)
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        try:
            self.conn, _ = self.server.accept()
        except OSError:
            return
        self.accepted.set()

    def send(self, *objs, raw: bytes = b"") -> None:
        assert self.accepted.wait(5), "source never connected"
        data = b"".join(json.dumps(o).encode() + b"\n" for o in objs) + raw
        self.conn.sendall(data)

    def drop(self) -> None:
        for s in (self.conn, self.server):
            if s is not None:
                with contextlib.suppress(OSError):
                    s.close()
        self.conn = self.server = None
        self.accepted.clear()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.path)


class Harness:
    def __init__(self, tmp_path, **settings) -> None:
        self.path = str(tmp_path / "exec.sock")
        self.events: list[Event] = []
        opts = {"helper_retry_min": 0.05, "helper_retry_max": 0.2, "helper_socket": self.path}
        opts.update(settings)
        self.source = EbpfSource(
            replace(ExecSettings(), source="ebpf", **opts),
            emit=self.events.append,
            cgroup_root=str(tmp_path / "cgroup"),
        )
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.source.run, args=(self.stop,), daemon=True)
        self.helper = FakeHelper(self.path)

    def start(self) -> None:
        self.thread.start()

    def wait(self, predicate, seconds=5.0) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("condition not reached")

    def close(self) -> None:
        self.stop.set()
        self.helper.drop()
        self.thread.join(5)


@pytest.fixture
def harness(tmp_path):
    h = Harness(tmp_path)
    yield h
    h.close()


def test_default_source_is_ebpf():
    assert ExecSettings().source == "ebpf"
    assert load_config().exec.source == "ebpf"


def test_events_carry_the_fields_the_context_builder_needs(harness):
    harness.helper.listen()
    harness.start()
    harness.helper.send(HELLO, exec_line(argv=["/bin/ls", "-l", "/tmp"]))
    harness.wait(lambda: harness.events)
    e = harness.events[0]
    assert e.kind == "exec" and e.subject == "100"
    f = e.fields
    assert f["exe"] == "/nix/store/x-coreutils/bin/ls" and f["filename"] == "ls"
    assert f["argv"] == "/bin/ls -l /tmp" and f["argv0"] == "ls" and f["ppid"] == 50
    assert f["uid"] == 1000 and f["comm"] == "ls" and f["parent_comm"] == "bash"
    assert f["argv_truncated"] == 0
    assert harness.source.connected and harness.source.status()["exec_error"] == ""


def test_truncation_flag_and_unresolved_exe(harness):
    harness.helper.listen()
    harness.start()
    harness.helper.send(HELLO, exec_line(trunc=1, exe="", filename="/bin/odd"))
    harness.wait(lambda: harness.events)
    f = harness.events[0].fields
    assert f["argv_truncated"] == 1 and f["exe"] == "/bin/odd"
    st = harness.source.status()
    assert st["exec_truncated"] == 1 and st["exec_exe_unresolved"] == 1


def test_lost_counter_comes_from_the_heartbeat(harness):
    harness.helper.listen()
    harness.start()
    harness.helper.send(
        HELLO,
        exec_line(1),
        exec_line(2),
        {"t": "hb", "seen": 10, "emitted": 2, "lost": 7, "undelivered": 0, "client_dropped": 1},
    )
    harness.wait(lambda: harness.source.helper_lost == 7)
    st = harness.source.status()
    assert st["exec_lost"] == 7 and st["exec_client_dropped"] == 1 and st["exec_seen"] == 10
    assert harness.source.missed_total == 8
    assert harness.source.miss_ratio == pytest.approx(8 / 10)


def test_malformed_lines_are_counted_and_skipped(harness):
    harness.helper.listen()
    harness.start()
    harness.helper.send(HELLO, raw=b"not json\n{broken\n[1,2]\n")
    harness.helper.send({"t": "exec", "pid": "x"}, {"t": "who"}, exec_line(5))
    harness.wait(lambda: harness.events)
    assert len(harness.events) == 1
    assert harness.source.malformed == 5


def test_unavailable_helper_is_reported_not_replaced_by_polling(harness):
    harness.start()
    harness.wait(lambda: harness.source.error)
    st = harness.source.status()
    assert st["exec_connected"] is False
    assert st["exec_error"].startswith(UNAVAILABLE) and HELPER_HINT in st["exec_error"]
    assert harness.events == []


def test_reconnects_with_backoff_after_the_helper_comes_and_goes(harness):
    harness.start()
    time.sleep(0.2)  # a few refused attempts first
    harness.helper.listen()
    harness.helper.send(HELLO, exec_line(1, argv=["/bin/a"]))
    harness.wait(lambda: len(harness.events) == 1)
    harness.helper.drop()
    harness.wait(lambda: not harness.source.connected)
    assert "connection to the helper was lost" in harness.source.status()["exec_error"]
    harness.helper.listen()
    harness.helper.send(HELLO, exec_line(2, argv=["/bin/b"]), {"t": "hb", "seen": 1, "lost": 3})
    harness.wait(lambda: len([e for e in harness.events if e.kind == "exec"]) == 2)
    harness.wait(lambda: harness.source.helper_lost == 3)
    assert harness.source.connected
    assert harness.source.status()["exec_lost"] == 3


def test_wrong_protocol_version_is_unavailable(harness):
    harness.helper.listen()
    harness.start()
    harness.helper.send({"t": "hello", "v": 99})
    harness.wait(lambda: "protocol v99" in harness.source.error)
    assert not harness.source.connected


def test_parent_cache_lookup(tmp_path):
    events: list[Event] = []
    s = EbpfSource(ExecSettings(), emit=events.append, cgroup_root=str(tmp_path))
    s.handle_line(json.dumps(exec_line(50, 1, exe="/bin/bash", comm="bash", argv=["bash"])))
    s.handle_line(json.dumps(exec_line(60, 50, pstart=500)))
    s.handle_line(json.dumps(exec_line(61, 50, pstart=999)))
    _, child, reused = events
    assert child.fields["parent_exe"] == "/bin/bash" and child.fields["parents"] == "bash"
    assert reused.fields["parent_exe"] == "" and reused.fields["parent_comm"] == "bash"


def test_cache_is_bounded(tmp_path):
    s = EbpfSource(replace(ExecSettings(), recent_exec_max=10), cgroup_root=str(tmp_path))
    for pid in range(100, 200):
        s.handle_line(json.dumps(exec_line(pid)))
    assert s.tracked == 10


def test_cgroup_resolver_maps_ids_to_paths(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    r = CgroupResolver(str(tmp_path), mono=iter(range(0, 1000, 10)).__next__)
    ino = (tmp_path / "a" / "b").stat().st_ino
    assert r.path(ino) == "/a/b"
    assert r.path(123456789) == ""


def test_hello_must_come_first(harness):
    harness.helper.listen()
    harness.start()
    harness.helper.send(exec_line(1))
    harness.wait(lambda: "did not speak" in harness.source.error)
    assert harness.events == []


def test_monitor_reports_unavailable_helper_in_status(tmp_path):
    cfg = load_config()
    cfg = replace(
        cfg,
        exec=replace(cfg.exec, helper_socket=str(tmp_path / "none.sock"), helper_retry_min=0.05),
        judge=JudgeSettings(mode="fake"),
    )
    store = SqliteStore(":memory:")
    monitor = Monitor(cfg, store, decider=FakeDecider(cfg.judge))
    monitor.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not monitor.status().get("exec_error"):
            time.sleep(0.02)
        st = monitor.status()
        assert st["exec_source"] == "ebpf" and st["exec_connected"] is False
        assert "exec source unavailable (helper not running)" in st["exec_error"]
        assert "jevmon start" in st["exec_error"]
    finally:
        monitor.stop()
        store.close()


def test_config_ebpf_keys(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        '[exec]\nsource = "proc"\nhelper_socket = "/x/y.sock"\nhelper_retry_min = 1\n'
        'helper_retry_max = 2\nrecent_exec_max = 9\nhelper_path = "/p/jevmon-exec"\n'
    )
    c = load_config(path)
    assert c.exec.source == "proc" and c.exec.helper_socket == "/x/y.sock"
    assert c.exec.recent_exec_max == 9 and c.exec.helper_retry_max == 2.0
    path.write_text('[exec]\nhelper_sock = "x"\n')
    with pytest.raises(ConfigError):
        load_config(path)
    path.write_text("[exec]\nhelper_retry_min = 5\nhelper_retry_max = 1\n")
    with pytest.raises(ConfigError):
        load_config(path)
    path.write_text('[exec]\nsource = "polling"\n')
    with pytest.raises(ConfigError):
        load_config(path)


def test_buffered_counts_match_the_store_after_flush():
    store = SqliteStore(":memory:")
    c = BufferedCounts(store)
    assert c.bump("exe", "a", 1.0) == 0 and c.bump("exe", "a", 2.0) == 1
    c.bump("exe", "b", 3.0)
    assert c.count("exe", "a") == 2 and c.total("exe") == 3 and c.distinct("exe") == 2
    assert c.first_time("exe", "a") == 1.0 and c.first_time("exe", "zz") is None
    c.flush()
    assert store.count("exe", "a") == 2 and store.total("exe") == 3
    # a new instance starts from what was stored
    again = BufferedCounts(store)
    assert again.count("exe", "a") == 2 and again.total("exe") == 3
    for v in (1.0, 2.0, 3.0, 4.0):
        c.add_sample("rarity", v, now=10.0)
    assert c.rank("rarity", 0.0, 2.5) == 50.0
    c2 = BufferedCounts(store)
    assert c2.rank("rarity", 0.0, 2.5) is None  # samples are in memory only
    store.close()


def test_store_sink_coalesces_repeated_cached_verdicts():
    store = SqliteStore(":memory:")
    sink = StoreSink(store, False, repeat_seconds=60.0)

    def verdict(t, cached):
        return Event(
            time=t, source="exec", subject="1", kind="verdict", severity=Severity.SAFE,
            message="m", fields={"signature": "s", "cached": int(cached)},
        )  # fmt: skip

    sink.emit(verdict(0, False))
    for t in (1, 2, 3):
        sink.emit(verdict(t, True))  # first cached is stored, the rest in the window are not
    sink.emit(verdict(4, True))
    sink.emit(verdict(100, True))  # window over: stored, carries the skipped count
    sink.flush()
    rows = store.recent_events(10)
    assert len(rows) == 3 and sink.coalesced == 3
    assert rows[0].fields["repeats"] == 3
    store.close()


def test_reader_is_throttled_when_it_exceeds_its_cpu_budget(tmp_path):
    clock = {"t": 0.0, "cpu": 0.0}
    slept: list[float] = []
    s = EbpfSource(
        replace(ExecSettings(), helper_cpu_budget=0.5),
        mono=lambda: clock["t"],
        cpu_clock=lambda: clock["cpu"],
        sleep=slept.append,
        cgroup_root=str(tmp_path),
    )
    clock.update(t=1.0, cpu=0.4)  # 40% of a core: under budget
    s._throttle()
    assert slept == []
    clock.update(t=2.0, cpu=1.4)  # a full core in one second: well over
    s._throttle()
    assert len(slept) == 1 and 0.9 < slept[0] <= 1.0 and s.status()["exec_throttled_s"] > 0

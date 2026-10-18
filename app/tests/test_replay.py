"""The replay source, the fictional demo recording and the config hooks that select them."""

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from jevmon.config import ConfigError, load_config
from jevmon.core import SqliteStore
from jevmon.detect.fake import STATUS, FakeDecider, fake_verdict
from jevmon.events import Event, severity_of
from jevmon.monitor import Monitor, build_decider, build_source
from jevmon.sources.demo import DEMO_EPOCH, demo_events
from jevmon.sources.exec import KIND_EXEC, KIND_SUMMARY, ExecSettings
from jevmon.sources.replay import ReplayError, ReplaySource, load_events, write_events


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def recording(tmp_path: Path, count: int = 6) -> Path:
    path = tmp_path / "events.jsonl"
    write_events(path, demo_events(count=count))
    return path


def replay_config(tmp_path: Path, path: Path) -> Path:
    return write_config(
        tmp_path,
        "[exec]\n"
        'source = "replay"\n'
        f'replay_file = "{path}"\n'
        "interval = 0.0\n"
        "\n"
        "[judge]\n"
        'mode = "fake"\n',
    )


def replay_settings(path: Path) -> ExecSettings:
    return ExecSettings(source="replay", replay_file=str(path), interval=0.0)


def drain(settings: ExecSettings, timeout: float = 5.0) -> tuple[ReplaySource, list[Event]]:
    seen: list[Event] = []
    source = ReplaySource(settings, emit=seen.append)
    stop = threading.Event()
    thread = threading.Thread(target=source.run, args=(stop,), daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "the replay did not finish"
    return source, seen


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# -- the source --------------------------------------------------------------------------------


def test_replay_emits_the_recording_in_order_with_plausible_counters(tmp_path):
    path = recording(tmp_path)
    recorded = load_events(path)
    source, seen = drain(replay_settings(path))

    assert [event.to_json() for event in seen] == [event.to_json() for event in recorded]
    assert source.observed_total == sum(1 for e in recorded if e.kind == KIND_EXEC)
    assert source.missed_total == sum(
        int(e.fields.get("estimated_missed", 0)) for e in recorded if e.kind == KIND_SUMMARY
    )
    assert source.tracked == source.observed_total  # every demo pid is distinct
    assert source.effective_interval == 0.0
    assert 0.0 <= source.miss_ratio < 1.0
    assert source.apply_settings(replay_settings(path)) is None  # pacing is hot-reloadable


def test_replay_stops_on_the_stop_event_without_draining_the_recording(tmp_path):
    seen: list[Event] = []
    source = ReplaySource(replay_settings(recording(tmp_path, count=10)), emit=seen.append)
    stop = threading.Event()
    stop.set()
    source.run(stop)
    assert seen == [], "a stopped source must not emit"


def test_replay_file_errors_are_reported_as_replay_error(tmp_path):
    with pytest.raises(ReplayError, match="cannot read replay file"):
        load_events(tmp_path / "nope.jsonl")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ReplayError, match="no events recorded"):
        load_events(empty)

    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"time": 1}\n', encoding="utf-8")
    with pytest.raises(ReplayError, match="missing field"):
        load_events(broken)

    with pytest.raises(ReplayError, match="nothing to replay"):
        ReplaySource(replay_settings(recording(tmp_path)), events=[])


def test_replay_without_a_file_is_rejected_at_construction():
    with pytest.raises(ValueError, match="replay_file"):
        ExecSettings(source="replay")
    with pytest.raises(ValueError, match="source must be one of"):
        ExecSettings(source="telepathy")


# -- the fictional recording -------------------------------------------------------------------


def test_the_demo_recording_is_byte_identical_for_a_fixed_seed():
    first = demo_events()
    assert [e.to_json() for e in first] == [e.to_json() for e in demo_events()]
    assert first[0].time == DEMO_EPOCH
    assert demo_events(seed=1) != first  # ... and a different seed moves it


def test_the_demo_recording_is_entirely_fictional():
    text = "\n".join(event.to_json() for event in demo_events())
    for needle in (os.environ.get("USER"), os.environ.get("HOME"), socket.gethostname()):
        if needle and len(needle) >= 3:
            assert needle not in text, f"the demo recording leaked {needle!r}"
    assert "/srv/demo/" in text and "demo-gw-01" in text and "demo-session.service" in text
    assert "/home/" not in text  # invented uids and paths, never a real home


def test_the_demo_recording_has_a_baseline_summary_and_enough_shape():
    events = demo_events()
    kinds = [event.kind for event in events]
    assert kinds[0] == KIND_SUMMARY, "the recording starts with the poller's baseline scan"
    assert kinds.count(KIND_EXEC) == 64 and kinds.count(KIND_SUMMARY) == 8
    assert all(event.source == "exec" for event in events)
    assert demo_events(count=3)[0].kind == KIND_SUMMARY
    with pytest.raises(ValueError, match="count must be >= 1"):
        demo_events(count=0)


def test_a_recorded_event_survives_a_json_round_trip_untouched():
    for event in demo_events(count=5):
        assert Event(**json.loads(event.to_json())) == event
        assert all(isinstance(value, str | int | float) for value in event.fields.values())


# -- config hooks -----------------------------------------------------------------------------


def test_config_selects_the_replay_source_and_the_fake_judge(tmp_path):
    path = recording(tmp_path)
    cfg = load_config(replay_config(tmp_path, path))
    assert cfg.exec.source == "replay" and cfg.exec.replay_file == str(path)
    assert cfg.judge.mode == "fake"
    assert isinstance(build_source(cfg.exec, emit=lambda event: None), ReplaySource)
    assert isinstance(build_decider(cfg), FakeDecider)

    with pytest.raises(ConfigError, match=r"\[exec\] source must be one of"):
        load_config(write_config(tmp_path, '[exec]\nsource = "magic"\n'))
    with pytest.raises(ConfigError, match=r"\[exec\].*needs replay_file"):
        load_config(write_config(tmp_path, '[exec]\nsource = "replay"\n'))
    with pytest.raises(ConfigError, match=r"\[judge\] mode must be"):
        load_config(write_config(tmp_path, '[judge]\nmode = "oracle"\n'))


def test_the_shipped_defaults_select_the_ebpf_source_and_laya():
    cfg = load_config()
    assert cfg.exec.source == "ebpf" and cfg.judge.mode == "laya"
    assert not isinstance(build_source(cfg.exec, emit=lambda event: None), ReplaySource)
    assert not isinstance(build_decider(cfg), FakeDecider)


# -- the fake judge ---------------------------------------------------------------------------


def test_the_fake_decider_is_deterministic_and_reports_itself_honestly():
    settings = load_config().judge
    decider = FakeDecider(settings)
    decider.load()
    assert decider.ready and decider.status == STATUS and decider.active_device == "fake"

    state = "process launch (tier 0 /proc poller)\nexe: /srv/demo/bin/reportd"
    first = decider.decide(state)
    again = decider.decide(state)
    assert (first.label, first.confidence, first.probabilities) == (
        again.label,
        again.confidence,
        again.probabilities,
    )
    assert first.available and first.summary.startswith("mock: ")
    assert set(first.probabilities) == set(settings.categories)
    assert pytest.approx(sum(first.probabilities.values()), abs=0.02) == 1.0
    assert [d.label for d in decider.decide_batch([state, state])] == [first.label, first.label]
    assert fake_verdict(state, tuple(settings.categories))[:2] == (first.label, first.confidence)

    other = decider.decide(state + "exe: /srv/demo/tmp/stage2")
    assert other.summary.startswith("mock: ") and other.label in settings.categories
    with pytest.raises(ValueError, match="at least one category"):
        fake_verdict(state, ())


def test_the_fake_judge_covers_all_three_classes():
    settings = load_config().judge
    categories = tuple(settings.categories)
    seen = set()
    for index in range(400):
        label, _confidence, _probs = fake_verdict(str(index), categories)
        seen.add(severity_of(label))
    assert seen == {0, 1, 2}, f"the demo severity mix looks wrong: {seen}"


# -- end to end -------------------------------------------------------------------------------


def test_monitor_replays_the_recording_and_judges_it_with_the_fake_decider(tmp_path):
    path = recording(tmp_path, count=24)
    store = SqliteStore(tmp_path / "events.db")
    monitor = Monitor(load_config(replay_config(tmp_path, path)), store)
    try:
        monitor.start()
        assert isinstance(monitor.source, ReplaySource)
        assert wait_for(lambda: store.count_events() >= 6), "no verdicts stored"

        status = monitor.status()
        assert status["model_status"] == STATUS and status["judge_device"] == "fake"
        assert status["running"] is True and status["paused"] is False
        assert status["observed"] >= 6 and status["judged"] >= 1

        rows = store.query_events(limit=50)
        assert rows, "the store is empty"
        for _event_id, event, _labels in rows:
            assert event.kind == "verdict" and event.severity in (0, 1, 2)
            context = str(event.fields.get("context", ""))
            assert "/demo/" in context and "/home/" not in context
    finally:
        monitor.stop()
        store.close()


def test_a_replay_run_ends_on_its_own(tmp_path):
    path = recording(tmp_path, count=3)
    store = SqliteStore(tmp_path / "events.db")
    monitor = Monitor(load_config(replay_config(tmp_path, path)), store)
    try:
        monitor.start()
        assert wait_for(lambda: not monitor.running), "the replay never finished"
        assert monitor.status()["running"] is False
        assert wait_for(lambda: store.count_events() >= 1), "verdicts were dropped on shutdown"
    finally:
        monitor.stop()
        store.close()


def test_a_missing_recording_fails_before_any_thread_starts(tmp_path):
    config = write_config(
        tmp_path, f'[exec]\nsource = "replay"\nreplay_file = "{tmp_path / "gone.jsonl"}"\n'
    )
    store = SqliteStore(tmp_path / "events.db")
    monitor = Monitor(load_config(config), store)
    try:
        with pytest.raises(ReplayError, match="cannot read replay file"):
            monitor.start()
        assert monitor.status()["running"] is False
    finally:
        store.close()


def test_jevmon_once_replays_the_recording_to_stdout(tmp_path, capsys):
    from jevmon.cli import main

    path = recording(tmp_path, count=4)
    config = write_config(tmp_path, f'[exec]\nsource = "replay"\nreplay_file = "{path}"\n')
    assert main(["exec", "--once", "--config", str(config), "--db", str(tmp_path / "e.db")]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines and lines[-1]["kind"] == KIND_SUMMARY
    assert all(line["source"] == "exec" for line in lines)

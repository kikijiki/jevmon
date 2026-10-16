"""SQLite store migration/retention/counts/labels, and the pipeline runner."""

import sqlite3
import time

import pytest

from jevmon.core import (
    KIND_EDGE,
    SCHEMA_VERSION,
    MemoryCounts,
    Pipeline,
    SqliteStore,
    edge_key,
)
from jevmon.events import Event, JsonlSink


def ev(t: float, severity: int = 0, kind: str = "exec") -> Event:
    return Event(time=t, source="exec", subject="1", kind=kind, severity=severity)


def test_fresh_store_migrates_to_current_version(tmp_path):
    store = SqliteStore(tmp_path / "e.db")
    assert store.schema_version == SCHEMA_VERSION
    store.close()


@pytest.mark.parametrize("version", [1, 4, SCHEMA_VERSION + 5])
def test_store_from_another_schema_is_discarded_and_recreated(tmp_path, version, capsys):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, time REAL, source TEXT, subject TEXT, "
        "kind TEXT, severity INT, message TEXT, fields TEXT);"
        f"PRAGMA user_version={version};"
    )
    conn.execute(
        "INSERT INTO events (time, source, subject, kind, severity, message, fields) "
        "VALUES (1.7e9, 'exec', '1', 'verdict', 2, 'judge: crit 0.9', '{}')"
    )
    conn.commit()
    conn.close()
    store = SqliteStore(path)
    assert store.schema_version == SCHEMA_VERSION
    assert store.count_events() == 0
    assert "discarded" in capsys.readouterr().err
    store.add_event(Event(time=1.7e9, source="exec", subject="2", kind="verdict", command="ls"))
    assert [e.command for e in store.recent_events(10)] == ["ls"]
    store.close()


def test_events_roundtrip_and_retention(tmp_path):
    store = SqliteStore(tmp_path / "e.db", retention_days=1)
    now = time.time()
    store.add_event(ev(now - 3 * 86400))
    store.add_event(ev(now, severity=2))
    store.add_sample(KIND_EDGE, 1.0, now=now - 3 * 86400)
    store.add_sample(KIND_EDGE, 2.0, now=now)
    assert store.count_events() == 2
    assert store.prune(now=now) == 1
    assert store.count_events() == 1
    assert store.samples(KIND_EDGE, since=0) == [2.0]
    ids = store.recent_with_ids(10)
    assert ids[0][1].severity == 2
    store.close()


def test_counts_bump_count_total_distinct_and_prefix():
    counts = MemoryCounts()
    assert counts.bump(KIND_EDGE, edge_key("a", "x")) == 0
    assert counts.bump(KIND_EDGE, edge_key("a", "x")) == 1
    assert counts.bump(KIND_EDGE, edge_key("a", "y")) == 0
    assert counts.count(KIND_EDGE, edge_key("a", "x")) == 2
    assert counts.total(KIND_EDGE) == 3
    assert counts.distinct(KIND_EDGE) == 2
    assert counts.distinct_prefix(KIND_EDGE, "a\x1f") == 2
    assert counts.distinct_prefix(KIND_EDGE, "b\x1f") == 0


def test_sqlite_counts_match_memory_semantics(tmp_path):
    store = SqliteStore(tmp_path / "e.db")
    store.bump(KIND_EDGE, edge_key("a", "x"), now=1.0)
    store.bump(KIND_EDGE, edge_key("a", "x"), now=2.0)
    store.bump(KIND_EDGE, edge_key("a", "y"), now=3.0)
    assert store.count(KIND_EDGE, edge_key("a", "x")) == 2
    assert store.total(KIND_EDGE) == 3
    assert store.distinct(KIND_EDGE) == 2
    assert store.distinct_prefix(KIND_EDGE, "a\x1f") == 2
    assert store.first_time(KIND_EDGE, edge_key("a", "x")) == 1.0
    store.close()


def test_labels_roundtrip(tmp_path):
    store = SqliteStore(tmp_path / "e.db")
    eid = store.add_event(ev(time.time(), severity=2, kind="alert"))
    store.add_label(eid, "true-positive", source="human")
    store.add_label(eid, "ack", source="human")
    assert store.labels(eid) == [("true-positive", "human"), ("ack", "human")]
    assert store.label_count() == 2
    rows = list(store.labelled_events())
    assert len(rows) == 2 and rows[0][0] == eid
    store.close()


def test_pipeline_moves_events_to_sink(tmp_path):
    path = tmp_path / "out.jsonl"
    sink = JsonlSink(path=path)
    pipeline = Pipeline([sink], capacity=8)
    pipeline.start()
    for i in range(5):
        pipeline.submit(ev(time.time(), severity=2))
    deadline = time.time() + 5
    while path.read_text().count("\n") < 5 and time.time() < deadline:
        time.sleep(0.01)
    pipeline.stop()
    assert path.read_text().count("\n") == 5
    assert pipeline.stats().delivered == 5


def test_pipeline_survives_broken_sink():
    class Broken:
        def emit(self, event):
            raise RuntimeError("nope")

        def close(self):
            pass

    import io

    sink = JsonlSink(stream=io.StringIO())
    pipeline = Pipeline([Broken(), sink], capacity=4)
    pipeline.start()
    pipeline.submit(ev(time.time()))
    deadline = time.time() + 5
    while pipeline.stats().delivered < 1 and time.time() < deadline:
        time.sleep(0.01)
    pipeline.stop()
    assert pipeline.stats().delivered == 1


def test_command_round_trips_through_the_store(tmp_path):
    store = SqliteStore(tmp_path / "new.db")
    store.add_event(
        Event(time=1.7e9 + 1, source="exec", subject="2", kind="verdict", command="ls -la")
    )
    assert [e.command for e in store.recent_events(10)] == ["ls -la"]
    store.close()

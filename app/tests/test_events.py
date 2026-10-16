"""Event core: serialization, drop-oldest counters, JSONL sink."""

import io
import json

from jevmon.events import BoundedQueue, Event, JsonlSink, Severity


def ev(n: int = 0, **kw) -> Event:
    base = {"time": 1000.0 + n, "source": "exec", "subject": str(n), "kind": "exec"}
    base.update(kw)
    return Event(**base)


def test_event_json_is_ascii_one_line():
    e = ev(message="caf\u00e9 \x1b[31mred", fields={"a": 1, "b": "x\ny"})
    raw = e.to_json()
    assert "\n" not in raw and raw.isascii()
    back = json.loads(raw)
    assert back["severity"] == 0 and back["fields"]["b"] == "x\ny"


def test_queue_drops_oldest_and_counts():
    q = BoundedQueue(capacity=3)
    for i in range(5):
        q.put(ev(i))
    assert len(q) == 3
    st = q.stats()
    assert (st.received, st.dropped) == (5, 2)
    got = [q.get(timeout=0).subject for _ in range(3)]
    assert got == ["2", "3", "4"]  # oldest dropped
    assert q.stats().delivered == 3


def test_queue_close_returns_none():
    q = BoundedQueue(capacity=1)
    q.close()
    assert q.get(timeout=0) is None
    assert q.put(ev()) is False


def test_jsonl_sink_writes_lines_to_stream():
    buf = io.StringIO()
    sink = JsonlSink(stream=buf)
    sink.emit(ev(1, severity=Severity.CRIT, message="boom"))
    sink.emit(ev(2))
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 2
    assert all(line.isascii() for line in lines)
    assert json.loads(lines[0])["severity"] == Severity.CRIT
    sink.close()


def test_jsonl_sink_path_roundtrip(tmp_path):
    path = tmp_path / "out.jsonl"
    sink = JsonlSink(path=path)
    sink.emit(ev(1))
    sink.close()
    assert json.loads(path.read_text().strip())["subject"] == "1"


def test_command_is_serialized_and_defaults_to_empty():
    assert json.loads(ev(command="ls -la").to_json())["command"] == "ls -la"
    assert ev().command == ""

"""Protocol: parsing, validation, limits, encoding and the bounded event payload."""

import json

import pytest

from jevmon.daemon import protocol
from jevmon.events import Event


def test_valid_requests_parse():
    req = protocol.parse_request({"v": 4, "id": 3, "type": "events", "limit": 5, "unlabeled": True})
    assert req.kind == "events" and req.req_id == 3
    assert req.params == {"limit": 5, "unlabeled": True}


def test_version_defaults_to_one():
    assert protocol.parse_request({"type": "status"}).kind == "status"


def test_unknown_type_field_and_version_are_rejected():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"type": "nope"})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"type": "status", "bogus": 1})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"v": 99, "type": "status"})


def test_bad_field_types_and_bounds_are_rejected():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"type": "events", "limit": 0})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"type": "events", "limit": protocol.MAX_LIMIT + 1})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"type": "events", "unlabeled": "yes"})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"type": "events", "text": "x" * (protocol.MAX_TEXT + 1)})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"type": "label", "event_id": 1, "label": "x" * 200})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"type": "event", "event_id": True})


def test_encode_is_one_ascii_line():
    line = protocol.encode({"text": "héllo"})
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert line.decode("ascii")
    assert json.loads(line)["text"] == "héllo"


def test_event_payload_is_bounded_and_carries_probabilities():
    event = Event(
        time=1.0,
        source="exec",
        subject="42",
        kind="verdict",
        severity=2,
        message="m" * 10_000,
        fields={
            "context": "c" * 10_000,
            "verdict": "v" * 10_000,
            "label": "crit",
            "confidence": 0.9,
            "judged": 1,
            "probabilities": json.dumps({"safe": 0.05, "crit": 0.9}),
        },
    )
    payload = protocol.event_payload(7, event, ["ack", "true-positive"])
    assert payload["id"] == 7 and payload["class"] == "crit"
    assert payload["probabilities"] == {"safe": 0.05, "crit": 0.9}
    assert payload["label"] == "true-positive" and payload["acked"] is True
    assert len(payload["message"]) <= protocol.MAX_STRING
    assert len(payload["context"]) <= protocol.MAX_STRING


def test_event_payload_survives_broken_probabilities():
    event = Event(
        time=1.0, source="exec", subject="1", kind="verdict", fields={"probabilities": "{"}
    )
    assert protocol.event_payload(1, event, [])["probabilities"] == {}


def test_error_and_stream_shapes():
    assert protocol.error(2, "boom")["ok"] is False
    assert protocol.stream("stats", stats={})["stream"] == "stats"
    assert protocol.ok(None, {})["id"] is None


def test_event_payload_carries_command_and_class():
    event = Event(
        time=1.0,
        source="exec",
        subject="42",
        kind="verdict",
        severity=1,
        command="curl -s https://demo-gw-01/",
        fields={"label": "warn"},
    )
    payload = protocol.event_payload(1, event, [])
    assert payload["command"] == "curl -s https://demo-gw-01/"
    assert payload["class"] == "warn" and payload["severity"] == 1
    assert "category" not in payload and "category_severity" not in payload

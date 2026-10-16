"""Wire protocol for the jevmon daemon control socket, version 2.

Newline-delimited JSON over a Unix domain socket. One request per line; a line longer than
`MAX_LINE` is rejected. Every request carries `type` (required), an optional correlating `id`
(echoed on the response) and `v` (the protocol version; absent means 3). Unknown request types and
unknown fields are rejected, and every reply is bounded in size.

    request:  {"v":4,"id":7,"type":"status"}
    response: {"v":4,"id":7,"ok":true,"result":{...}}
    error:    {"v":4,"id":7,"ok":false,"error":"..."}

`subscribe` turns the connection into a one-way stream of new events plus periodic stats:

    {"v":4,"stream":"ready","result":{"subscribed":true}}
    {"v":4,"stream":"event","event":{...}}
    {"v":4,"stream":"stats","stats":{...}}

Version 2 (one axis): an event's `severity` is the class (0 safe, 1 warn, 2 crit) and `class`
is its name; the old `category` and `category_severity` fields are gone. There is no
compatibility with version 1 peers.

Requests:

- `status`                  -> status object (see `Monitor.status`).
- `events`  {since_id, min_severity, limit, unlabeled, text} -> {events, count, more}.
- `event`   {event_id}      -> {event}.
- `label`   {event_id,label}-> {event} (the label is human input, never interpreted here).
- `ack`     {event_id}      -> {event}.
- `pause` / `resume`        -> status.
- `reload`                  -> {reloaded, config}.
- `shutdown`                -> {shutdown:true}.
"""

import json
from dataclasses import dataclass

from jevmon.events import Event
from jevmon.sanitize import shorten

PROTOCOL_VERSION = 4
MAX_LINE = 131_072  # bytes of one request line
MAX_REPLY = 8_000_000  # bytes of one reply line the client will read
MAX_LIMIT = 100  # events returned by one query
DEFAULT_LIMIT = 50
MAX_TEXT = 512  # query text filter
MAX_LABEL = 128  # human label
MAX_STRING = 4_000  # any string field in a reply

REQUEST_FIELDS: dict[str, frozenset[str]] = {
    "status": frozenset(),
    "events": frozenset({"since_id", "min_severity", "limit", "unlabeled", "text"}),
    "event": frozenset({"event_id"}),
    "label": frozenset({"event_id", "label"}),
    "ack": frozenset({"event_id"}),
    "pause": frozenset(),
    "resume": frozenset(),
    "reload": frozenset(),
    "shutdown": frozenset(),
    "subscribe": frozenset({"since_id"}),
}
COMMON_FIELDS = frozenset({"v", "id", "type"})


class ProtocolError(ValueError):
    """A request line is malformed, oversized, unknown or has a bad field."""


@dataclass(frozen=True, slots=True)
class Request:
    kind: str
    req_id: int | None
    params: dict[str, object]


def _int(
    value: object, name: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError(f"{name}: expected an integer")
    if minimum is not None and value < minimum:
        raise ProtocolError(f"{name}: must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ProtocolError(f"{name}: must be <= {maximum}")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"{name}: expected a boolean")
    return value


def _str(value: object, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{name}: expected a string")
    if len(value) > limit:
        raise ProtocolError(f"{name}: longer than {limit}")
    return value


def parse_request(obj: object) -> Request:
    """Validate a decoded request object. Raises `ProtocolError` on anything unknown or bad."""
    if not isinstance(obj, dict):
        raise ProtocolError("request must be a JSON object")
    version = obj.get("v", PROTOCOL_VERSION)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version!r}")
    kind = obj.get("type")
    if not isinstance(kind, str):
        raise ProtocolError("missing request type")
    allowed = REQUEST_FIELDS.get(kind)
    if allowed is None:
        raise ProtocolError(f"unknown request type {kind!r}")
    unknown = set(obj) - COMMON_FIELDS - allowed
    if unknown:
        raise ProtocolError(f"unknown field(s): {', '.join(sorted(unknown))}")
    req_id = obj.get("id")
    if req_id is not None:
        req_id = _int(req_id, "id", minimum=0)
    params = _parse_params(kind, obj)
    return Request(kind, req_id, params)


def _parse_params(kind: str, obj: dict) -> dict[str, object]:
    if kind == "events":
        params: dict[str, object] = {}
        if "since_id" in obj:
            params["since_id"] = _int(obj["since_id"], "since_id", minimum=0)
        if "min_severity" in obj:
            params["min_severity"] = _int(obj["min_severity"], "min_severity", minimum=0, maximum=2)
        if "limit" in obj:
            params["limit"] = _int(obj["limit"], "limit", minimum=1, maximum=MAX_LIMIT)
        if "unlabeled" in obj:
            params["unlabeled"] = _bool(obj["unlabeled"], "unlabeled")
        if "text" in obj:
            params["text"] = _str(obj["text"], "text", MAX_TEXT)
        return params
    if kind in {"event", "ack"}:
        return {"event_id": _int(obj.get("event_id"), "event_id", minimum=1)}
    if kind == "label":
        return {
            "event_id": _int(obj.get("event_id"), "event_id", minimum=1),
            "label": _str(obj.get("label"), "label", MAX_LABEL),
        }
    if kind == "subscribe":
        params = {}
        if "since_id" in obj:
            params["since_id"] = _int(obj["since_id"], "since_id", minimum=0)
        return params
    return {}


# -- encoding ------------------------------------------------------------------------------


def encode(obj: dict) -> bytes:
    """One ASCII-only JSON line, newline-terminated."""
    return (json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")


def ok(req_id: int | None, result: dict) -> dict:
    return {"v": PROTOCOL_VERSION, "id": req_id, "ok": True, "result": result}


def error(req_id: int | None, message: str) -> dict:
    return {"v": PROTOCOL_VERSION, "id": req_id, "ok": False, "error": shorten(message, 400)}


def stream(kind: str, **payload: object) -> dict:
    return {"v": PROTOCOL_VERSION, "stream": kind, **payload}


def bounded(value: str, limit: int = MAX_STRING) -> str:
    """Cap a reply string so a hostile field cannot inflate a reply."""
    return shorten(value, limit)


def _number(fields: dict, key: str) -> float | None:
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _probabilities(fields: dict) -> dict[str, float]:
    raw = fields.get("probabilities")
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for name, prob in parsed.items():
        if isinstance(prob, int | float) and not isinstance(prob, bool):
            out[str(name)] = round(float(prob), 4)
    return out


def event_payload(event_id: int, event: Event, labels: list[str]) -> dict:
    """The full event view shown by the TUI/ctl: model context, verdict, probabilities, label."""
    fields = event.fields
    human = [bounded(str(label), MAX_LABEL) for label in labels]
    # `label` is the last human verdict (TP/FP); the ack is reported separately so it cannot mask it.
    human_labels = [label for label in human if label != "ack"]
    return {
        "id": int(event_id),
        "time": round(float(event.time), 3),
        "source": bounded(event.source, 64),
        "subject": bounded(event.subject, 256),
        "kind": bounded(event.kind, 64),
        "severity": int(event.severity),
        "message": bounded(event.message),
        "command": bounded(event.command),
        "context": bounded(str(fields.get("context", ""))),
        "verdict": bounded(str(fields.get("verdict", ""))),
        "class": bounded(str(fields.get("label", "")), MAX_LABEL),
        "confidence": _number(fields, "confidence"),
        "probabilities": _probabilities(fields),
        "judged": bool(fields.get("judged", 0)),
        "cached": bool(fields.get("cached", 0)),
        "available": bool(fields.get("available", 0)),
        "label": human_labels[-1] if human_labels else None,
        "labels": human,
        "acked": "ack" in human,
    }


__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LABEL",
    "MAX_LIMIT",
    "MAX_LINE",
    "MAX_REPLY",
    "MAX_STRING",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "Request",
    "bounded",
    "encode",
    "error",
    "event_payload",
    "ok",
    "parse_request",
    "stream",
]

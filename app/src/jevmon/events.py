"""Neutral event core: the thing sources produce, queues move and sinks consume.

Nothing here assumes logs. An `Event` is a timestamped, severity-tagged statement from a source
about a subject, with a short human message and scalar fields. Text fields are expected to be
already sanitized and redacted by the producer; sinks only serialize.

Components:

- `Event`: frozen, JSON-serializable (ASCII-only) value.
- `BoundedQueue`: thread-safe drop-oldest queue with received/dropped/delivered counters.
- `Sink` / `JsonlSink`: one JSON object per line, to a path or a stream.

The runner that drains a queue into sinks lives in `core.py`.
"""

import json
import sys
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, TextIO

type Scalar = str | int | float


class Severity(IntEnum):
    """The three verdict classes, one axis: 0 safe, 1 warn, 2 crit. A verdict's label is the
    lowercase member name. IntEnum so it serializes as a plain int."""

    SAFE = 0
    WARN = 1
    CRIT = 2


def severity_of(label: str) -> int:
    """The severity of a verdict label (`safe`, `warn`, `crit`); an unknown label reads as safe."""
    try:
        return int(Severity[label.upper()])
    except KeyError:
        return int(Severity.SAFE)


@dataclass(frozen=True, slots=True)
class Event:
    """One observation. `fields` values are scalars so the event is trivially serializable.

    `command` is an optional, already sanitized/bounded/redacted command line for display (the
    producer builds it; sinks only serialize).
    """

    time: float
    source: str
    subject: str
    kind: str
    severity: int = Severity.SAFE
    message: str = ""
    fields: Mapping[str, Scalar] = field(default_factory=dict)
    command: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "time": round(self.time, 6),
            "source": self.source,
            "subject": self.subject,
            "kind": self.kind,
            "severity": int(self.severity),
            "message": self.message,
            "fields": dict(self.fields),
            "command": self.command,
        }

    def to_json(self) -> str:
        """One line, ASCII-only (ensure_ascii), no embedded newlines."""
        return json.dumps(self.as_dict(), ensure_ascii=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class QueueStats:
    received: int
    dropped: int
    delivered: int
    size: int
    capacity: int


class BoundedQueue:
    """Drop-oldest queue. `put` never blocks; `get` blocks until an item or close/timeout."""

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._cap = capacity
        self._q: deque[Event] = deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._received = 0
        self._dropped = 0
        self._delivered = 0
        self._closed = False

    def put(self, event: Event) -> bool:
        """Enqueue `event`. Returns False if the queue is closed; oldest item is dropped if full."""
        with self._cond:
            if self._closed:
                return False
            self._received += 1
            if len(self._q) >= self._cap:
                self._q.popleft()
                self._dropped += 1
            self._q.append(event)
            self._cond.notify()
            return True

    def get(self, timeout: float | None = None) -> Event | None:
        """Return the oldest event, or None on timeout/closed-and-empty."""
        with self._cond:
            while not self._q and not self._closed:
                if not self._cond.wait(timeout):
                    return None
            if self._q:
                self._delivered += 1
                return self._q.popleft()
            return None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)

    def stats(self) -> QueueStats:
        with self._lock:
            return QueueStats(
                self._received, self._dropped, self._delivered, len(self._q), self._cap
            )


class Sink(Protocol):
    """Anything events can be delivered to. `close` must be idempotent."""

    def emit(self, event: Event) -> None: ...

    def close(self) -> None: ...


class JsonlSink:
    """Write one ASCII-only JSON object per event. Path mode opens append; stream mode flushes."""

    def __init__(self, path=None, stream: TextIO | None = None) -> None:
        if path is not None and stream is not None:
            raise ValueError("pass either path or stream, not both")
        self._lock = threading.Lock()
        self._own = path is not None
        if path is not None:
            self._stream: TextIO | None = open(path, "a", encoding="utf-8")  # noqa: SIM115
        else:
            self._stream = stream if stream is not None else sys.stdout

    def emit(self, event: Event) -> None:
        line = event.to_json()
        with self._lock:
            if self._stream is None:
                return
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if self._own and self._stream is not None:
                self._stream.close()
                self._stream = None


class NullSink:
    def emit(self, event: Event) -> None:
        pass

    def close(self) -> None:
        pass


class MinSeveritySink:
    """Forward only events at or above `min_severity` to an inner sink (e.g. alerts to stdout)."""

    def __init__(self, inner: Sink, min_severity: int = Severity.CRIT) -> None:
        self.inner = inner
        self.min_severity = int(min_severity)

    def emit(self, event: Event) -> None:
        if int(event.severity) >= self.min_severity:
            self.inner.emit(event)

    def close(self) -> None:
        self.inner.close()

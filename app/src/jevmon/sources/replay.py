"""Replay a recorded event stream instead of polling `/proc`.

The pipeline does not care where events come from: a source emits `Event`s and the context
builder, judge and store run unchanged. `ReplaySource` reads a JSON-lines recording of those
events (see `jevmon.sources.demo` for a fictional one) and re-emits them at a fixed wall-clock
pace, so demos, tests and screenshot runs are reproducible without touching a real process table.

Select it in config (`[exec]`):

    source = "replay"
    replay_file = "/path/to/events.jsonl"
    interval = 0.1   # wall seconds between replayed events

`interval` is the only pacing knob; the event timestamps are whatever the recording says.
"""

import json
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

from jevmon.events import Event, Scalar
from jevmon.sources.exec import KIND_EXEC, ExecSettings

_REQUIRED = ("time", "source", "subject", "kind")


class ReplayError(ValueError):
    """The recording is unreadable, empty or contains a malformed line."""


def _event(obj: object, where: str) -> Event:
    if not isinstance(obj, dict):
        raise ReplayError(f"{where}: event must be a JSON object")
    missing = [key for key in _REQUIRED if key not in obj]
    if missing:
        raise ReplayError(f"{where}: missing field(s): {', '.join(missing)}")
    fields = obj.get("fields", {})
    if not isinstance(fields, dict):
        raise ReplayError(f"{where}: fields must be a JSON object")
    clean_fields: dict[str, Scalar] = {}
    for key, value in fields.items():
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            raise ReplayError(f"{where}: field {key!r} is not a scalar")
        clean_fields[str(key)] = value
    severity = obj.get("severity", 0)
    if isinstance(severity, bool) or not isinstance(severity, int):
        raise ReplayError(f"{where}: severity must be an integer")
    try:
        return Event(
            time=float(obj["time"]),
            source=str(obj["source"]),
            subject=str(obj["subject"]),
            kind=str(obj["kind"]),
            severity=severity,
            message=str(obj.get("message", "")),
            fields=clean_fields,
            command=str(obj.get("command", "")),
        )
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"{where}: {exc}") from exc


def load_events(path: str | Path) -> list[Event]:
    """Read a JSON-lines recording into `Event`s, in file order."""
    recording = Path(path)
    try:
        text = recording.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReplayError(f"cannot read replay file {recording}: {exc}") from exc
    events: list[Event] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError as exc:
            raise ReplayError(f"{recording}:{lineno}: {exc}") from exc
        events.append(_event(obj, f"{recording}:{lineno}"))
    if not events:
        raise ReplayError(f"{recording}: no events recorded")
    return events


def write_events(path: str | Path, events: Iterable[Event]) -> int:
    """Write `events` as JSON lines (the recording format `load_events` reads back)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(event.to_json() + "\n")
            count += 1
    return count


class ReplaySource:
    """Emit a recorded event stream; `settings.interval` wall seconds before each event.

    The status counters the daemon reports keep their `/proc` meaning: `observed_total` counts
    replayed exec events, `tracked` counts distinct subjects, and there is nothing missed.
    """

    def __init__(
        self,
        settings: ExecSettings,
        emit: Callable[[Event], None] | None = None,
        *,
        events: list[Event] | None = None,
    ) -> None:
        self.settings = settings
        self._events = events if events is not None else load_events(settings.replay_file)
        if not self._events:
            raise ReplayError("nothing to replay")
        self._emit = emit if emit is not None else (lambda event: None)
        self.observed_total = 0
        self.missed_total = 0
        self._tracked: set[str] = set()

    @property
    def effective_interval(self) -> float:
        return self.settings.interval

    @property
    def tracked(self) -> int:
        return len(self._tracked)

    @property
    def miss_ratio(self) -> float:
        """Cumulative missed / (observed + missed), counted from the recording's summaries."""
        total = self.observed_total + self.missed_total
        return self.missed_total / total if total else 0.0

    @property
    def recording(self) -> tuple[Event, ...]:
        """The whole recording, in order (the demo/screenshot tooling reads it back)."""
        return tuple(self._events)

    def apply_settings(self, settings: ExecSettings) -> None:
        """Adopt new pacing; the recording itself is read once, at construction."""
        self.settings = settings

    def drain(self) -> tuple[Event, ...]:
        """Every recorded event at once (the `--once` path has no pacing)."""
        return tuple(self._events)

    def run(self, stop: threading.Event) -> None:
        for event in self._events:
            if stop.wait(self.settings.interval):
                return
            self._emit(event)
            if event.kind == KIND_EXEC:
                self.observed_total += 1
                self._tracked.add(event.subject)


__all__ = ["ReplayError", "ReplaySource", "load_events", "write_events"]

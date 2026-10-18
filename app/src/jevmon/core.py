"""Storage and the runner: a minimal SQLite store and a thread that drains a queue into sinks.

`SqliteStore` holds events, the learned counts used to build model context (exe/edge/template
frequencies, recently observed surprisal values) and human labels for later fine-tuning. WAL mode,
a schema version in `PRAGMA user_version`, day-based retention, safe from several threads.

`MemoryCounts` is the in-process counterpart of the count tables, for runs with no database.

`Pipeline` is the neutral runner: it owns a `BoundedQueue`, an ordered list of sinks and one worker
thread. Producers call `submit`; the worker moves each event to every sink.
"""

import bisect
import json
import sqlite3
import sys
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from jevmon.events import BoundedQueue, Event, QueueStats, Sink

SCHEMA_VERSION = 7  # any other version on disk is discarded and recreated (no migration)
DAY = 86_400.0

KIND_EXE = "exe"
KIND_EDGE = "edge"
KIND_TEMPLATE = "template"
KIND_RARITY = "rarity"
# Unit separator: unique enough inside names, and unlike NUL it survives SQLite text functions.
KEY_SEP = "\x1f"
# Separator used when aggregating multiple human labels into one column.
LABEL_SEP = "\x1f"

EVENT_KIND_VERDICT = "verdict"
EVENT_KIND_EXEC = "exec"


def _event_from_row(row: sqlite3.Row) -> Event:
    return Event(
        time=float(row["time"]),
        source=str(row["source"]),
        subject=str(row["subject"]),
        kind=str(row["kind"]),
        severity=int(row["severity"]),
        message=str(row["message"]),
        fields=json.loads(row["fields"]),
        command=str(row["command"]),
    )


def edge_key(parent: str, child: str) -> str:
    return f"{parent}{KEY_SEP}{child}"


class Counts(Protocol):
    """Persisted (or in-memory) frequencies and recent samples used to compute statistics."""

    def bump(self, kind: str, key: str, now: float | None = None) -> int: ...

    def count(self, kind: str, key: str) -> int: ...

    def first_time(self, kind: str, key: str) -> float | None: ...

    def total(self, kind: str) -> int: ...

    def distinct(self, kind: str) -> int: ...

    def distinct_prefix(self, kind: str, prefix: str) -> int: ...

    def add_sample(self, kind: str, value: float, now: float | None = None) -> None: ...

    def samples(self, kind: str, since: float) -> list[float]: ...


class MemoryCounts:
    """In-process `Counts`; the CLI uses it when no database is configured."""

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = defaultdict(dict)
        self._first: dict[str, dict[str, float]] = defaultdict(dict)
        self._totals: dict[str, int] = defaultdict(int)
        self._samples: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def bump(self, kind: str, key: str, now: float | None = None) -> int:
        prior = self._counts[kind].get(key, 0)
        if prior == 0:
            self._first[kind][key] = now if now is not None else time.time()
        self._counts[kind][key] = prior + 1
        self._totals[kind] += 1
        return prior

    def count(self, kind: str, key: str) -> int:
        return self._counts.get(kind, {}).get(key, 0)

    def first_time(self, kind: str, key: str) -> float | None:
        return self._first.get(kind, {}).get(key)

    def total(self, kind: str) -> int:
        return self._totals.get(kind, 0)

    def distinct(self, kind: str) -> int:
        return len(self._counts.get(kind, {}))

    def distinct_prefix(self, kind: str, prefix: str) -> int:
        return sum(1 for key in self._counts.get(kind, {}) if key.startswith(prefix))

    def add_sample(self, kind: str, value: float, now: float | None = None) -> None:
        self._samples[kind].append((now if now is not None else time.time(), value))

    def samples(self, kind: str, since: float) -> list[float]:
        return [value for stamp, value in self._samples.get(kind, []) if stamp >= since]


class SqliteStore:
    """WAL SQLite store for events, learned counts and labels. All public methods are locked."""

    def __init__(self, path: str | Path = ":memory:", retention_days: int = 14) -> None:
        self.path = str(path)
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._conn = self._open()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    def _open(self) -> sqlite3.Connection:
        """Connect; a database from another schema version is deleted and started fresh."""
        conn = sqlite3.connect(self.path, check_same_thread=False)
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, SCHEMA_VERSION):
            conn.close()
            if self.path != ":memory:":
                for suffix in ("", "-wal", "-shm"):
                    Path(self.path + suffix).unlink(missing_ok=True)
            print(
                f"jevmon: store {self.path} has schema {version}, expected {SCHEMA_VERSION}: "
                "discarded and recreated empty",
                file=sys.stderr,
            )
            conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def schema_version(self) -> int:
        with self._lock:
            return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> None:
        """Create the schema. A store written by another schema version is not upgraded: it is
        discarded and recreated empty (see `__init__`), so this only ever creates."""
        with self._lock:
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if version == SCHEMA_VERSION:
                return
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time REAL NOT NULL,
                    source TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    fields TEXT NOT NULL,
                    command TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS events_time ON events(time);
                CREATE INDEX IF NOT EXISTS events_severity ON events(severity);
                CREATE TABLE IF NOT EXISTS counts (
                    kind TEXT NOT NULL,
                    key TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    first_time REAL NOT NULL,
                    last_time REAL NOT NULL,
                    PRIMARY KEY (kind, key)
                );
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    value REAL NOT NULL,
                    time REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS samples_kind_time ON samples(kind, time);
                CREATE TABLE IF NOT EXISTS labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    source TEXT NOT NULL,
                    time REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS labels_event ON labels(event_id);
                """
            )
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._conn.commit()

    # -- events ---------------------------------------------------------------------------

    def add_event(self, event: Event) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events "
                "(time, source, subject, kind, severity, message, fields, command) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.time,
                    event.source,
                    event.subject,
                    event.kind,
                    int(event.severity),
                    event.message,
                    json.dumps(dict(event.fields), ensure_ascii=True, separators=(",", ":")),
                    event.command,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_events(self, events: Iterable[Event]) -> int:
        """Insert several events in one transaction; returns how many."""
        rows = [
            (
                e.time,
                e.source,
                e.subject,
                e.kind,
                int(e.severity),
                e.message,
                json.dumps(dict(e.fields), ensure_ascii=True, separators=(",", ":")),
                e.command,
            )
            for e in events
        ]
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT INTO events "
                "(time, source, subject, kind, severity, message, fields, command) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def set_counts(self, rows: Iterable[tuple[str, str, int, float, float]]) -> None:
        """Write absolute counts `(kind, key, count, first_time, last_time)` in one transaction."""
        with self._lock:
            self._conn.executemany(
                "INSERT INTO counts (kind, key, count, first_time, last_time) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(kind, key) DO UPDATE SET count = excluded.count, "
                "last_time = excluded.last_time",
                list(rows),
            )
            self._conn.commit()

    def prune(self, now: float | None = None) -> int:
        """Delete events and samples older than the retention window; returns events removed."""
        cutoff = (now if now is not None else time.time()) - self.retention_days * DAY
        with self._lock:
            cur = self._conn.execute("DELETE FROM events WHERE time < ?", (cutoff,))
            self._conn.execute("DELETE FROM samples WHERE time < ?", (cutoff,))
            self._conn.commit()
            return int(cur.rowcount)

    def count_events(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def recent_events(self, limit: int = 100) -> list[Event]:
        return [event for _, event in self.recent_with_ids(limit)]

    def recent_with_ids(self, limit: int = 100) -> list[tuple[int, Event]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [(int(r["id"]), _event_from_row(r)) for r in rows]

    def max_event_id(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0])

    def query_events(
        self,
        *,
        since_id: int = 0,
        min_severity: int = 0,
        limit: int = 100,
        unlabeled: bool = False,
        text: str | None = None,
    ) -> list[tuple[int, Event, list[str]]]:
        """Ascending by id. `unlabeled` drops events with any human label; `text` filters message,
        subject and the stored fields (which include the model context)."""
        where = ["e.id > ?", "e.severity >= ?"]
        params: list[object] = [int(since_id), int(min_severity)]
        if unlabeled:
            where.append("NOT EXISTS (SELECT 1 FROM labels l WHERE l.event_id = e.id)")
        if text:
            like = f"%{text}%"
            where.append("(e.message LIKE ? OR e.subject LIKE ? OR e.fields LIKE ?)")
            params += [like, like, like]
        params.append(int(limit))
        sql = (
            "SELECT e.*, "
            "(SELECT GROUP_CONCAT(l.label, char(31)) FROM labels l WHERE l.event_id = e.id) "
            "AS labels FROM events e WHERE " + " AND ".join(where) + " ORDER BY e.id ASC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_with_labels(r) for r in rows]

    def event_with_labels(self, event_id: int) -> tuple[int, Event, list[str]] | None:
        sql = (
            "SELECT e.*, "
            "(SELECT GROUP_CONCAT(l.label, char(31)) FROM labels l WHERE l.event_id = e.id) "
            "AS labels FROM events e WHERE e.id = ?"
        )
        with self._lock:
            row = self._conn.execute(sql, (int(event_id),)).fetchone()
        return self._row_with_labels(row) if row is not None else None

    def _row_with_labels(self, row: sqlite3.Row) -> tuple[int, Event, list[str]]:
        raw = row["labels"]
        labels = [part for part in str(raw).split(LABEL_SEP) if part] if raw else []
        return int(row["id"]), _event_from_row(row), labels

    # -- learned counts -------------------------------------------------------------------

    def bump(self, kind: str, key: str, now: float | None = None) -> int:
        stamp = now if now is not None else time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT count FROM counts WHERE kind = ? AND key = ?", (kind, key)
            ).fetchone()
            prior = int(row["count"]) if row is not None else 0
            self._conn.execute(
                "INSERT INTO counts (kind, key, count, first_time, last_time) "
                "VALUES (?, ?, 1, ?, ?) "
                "ON CONFLICT(kind, key) DO UPDATE SET count = counts.count + 1, "
                "last_time = excluded.last_time",
                (kind, key, stamp, stamp),
            )
            self._conn.commit()
            return prior

    def count(self, kind: str, key: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT count FROM counts WHERE kind = ? AND key = ?", (kind, key)
            ).fetchone()
            return int(row["count"]) if row is not None else 0

    def first_time(self, kind: str, key: str) -> float | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT first_time FROM counts WHERE kind = ? AND key = ?", (kind, key)
            ).fetchone()
            return float(row["first_time"]) if row is not None else None

    def total(self, kind: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(count), 0) AS n FROM counts WHERE kind = ?", (kind,)
            ).fetchone()
            return int(row["n"])

    def distinct(self, kind: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM counts WHERE kind = ?", (kind,)
            ).fetchone()
            return int(row["n"])

    def distinct_prefix(self, kind: str, prefix: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM counts WHERE kind = ? AND substr(key, 1, ?) = ?",
                (kind, len(prefix), prefix),
            ).fetchone()
            return int(row["n"])

    def add_sample(self, kind: str, value: float, now: float | None = None) -> None:
        stamp = now if now is not None else time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO samples (kind, value, time) VALUES (?, ?, ?)", (kind, value, stamp)
            )
            self._conn.commit()

    def samples(self, kind: str, since: float, limit: int = 5000) -> list[float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT value FROM samples WHERE kind = ? AND time >= ? ORDER BY time LIMIT ?",
                (kind, since, int(limit)),
            ).fetchall()
        return [float(r["value"]) for r in rows]

    # -- labels ---------------------------------------------------------------------------

    def add_label(self, event_id: int, label: str, source: str = "human") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO labels (event_id, label, source, time) VALUES (?, ?, ?, ?)",
                (int(event_id), label, source, time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def labels(self, event_id: int) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT label, source FROM labels WHERE event_id = ? ORDER BY id", (int(event_id),)
            ).fetchall()
        return [(str(r["label"]), str(r["source"])) for r in rows]

    def label_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0])

    def labelled_events(self) -> Iterable[tuple[int, str, Event]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.*, l.label AS label FROM events e JOIN labels l ON l.event_id = e.id "
                "ORDER BY e.id"
            ).fetchall()
        for r in rows:
            yield (int(r["id"]), str(r["label"]), _event_from_row(r))

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class BufferedCounts:
    """Write-behind `Counts` over a `SqliteStore`, for a source that execs hundreds of times a second.

    Reads and bumps are served from memory (a key is loaded from the store the first time it is
    touched); changed counts are written in one transaction every `flush_seconds` and on `close`.
    The rarity samples behind the percentile live in memory only: they are a one-hour window that
    refills after a restart (one SQL row per exec was the cost, not the number). The distinct
    count under an edge prefix is re-read from the store at most every `prefix_ttl` seconds.
    """

    SAMPLE_CAP = 5000

    def __init__(
        self, store: SqliteStore, flush_seconds: float = 1.0, prefix_ttl: float = 2.0
    ) -> None:
        self.store = store
        self.flush_seconds = flush_seconds
        self.prefix_ttl = prefix_ttl
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str], list] = {}  # [count, first, last]
        self._dirty: set[tuple[str, str]] = set()
        self._totals: dict[str, int] = {}
        self._distinct: dict[str, int] = {}
        self._prefix: dict[tuple[str, str], tuple[int, float]] = {}
        self._samples: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.SAMPLE_CAP)
        )
        self._sorted: dict[str, tuple[float, float, list[float]]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="jevmon-counts", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.flush_seconds):
            self.flush()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None
        self.flush()

    def flush(self) -> None:
        with self._lock:
            self.flush_locked()

    def _load(self, kind: str, key: str) -> list:
        entry = self._counts.get((kind, key))
        if entry is None:
            first = self.store.first_time(kind, key)
            entry = [self.store.count(kind, key), first if first is not None else 0.0, 0.0]
            self._counts[(kind, key)] = entry
        return entry

    def bump(self, kind: str, key: str, now: float | None = None) -> int:
        stamp = now if now is not None else time.time()
        with self._lock:
            entry = self._load(kind, key)
            prior = entry[0]
            if prior == 0:
                entry[1] = stamp
                if kind in self._distinct:
                    self._distinct[kind] += 1
            entry[0] = prior + 1
            entry[2] = stamp
            if kind in self._totals:
                self._totals[kind] += 1
            self._dirty.add((kind, key))
            return prior

    def count(self, kind: str, key: str) -> int:
        with self._lock:
            return self._load(kind, key)[0]

    def first_time(self, kind: str, key: str) -> float | None:
        with self._lock:
            entry = self._load(kind, key)
            return entry[1] if entry[0] > 0 else None

    def total(self, kind: str) -> int:
        with self._lock:
            if kind not in self._totals:
                self.flush_locked()
                self._totals[kind] = self.store.total(kind)
            return self._totals[kind]

    def distinct(self, kind: str) -> int:
        with self._lock:
            if kind not in self._distinct:
                self.flush_locked()
                self._distinct[kind] = self.store.distinct(kind)
            return self._distinct[kind]

    def flush_locked(self) -> None:
        rows = [(k, key, *self._counts[(k, key)]) for k, key in self._dirty]
        self._dirty.clear()
        if rows:
            self.store.set_counts(rows)

    def distinct_prefix(self, kind: str, prefix: str) -> int:
        now = time.monotonic()
        with self._lock:
            cached = self._prefix.get((kind, prefix))
            if cached is not None and now - cached[1] < self.prefix_ttl:
                return cached[0]
            self.flush_locked()
            value = self.store.distinct_prefix(kind, prefix)
            if len(self._prefix) > 4096:
                self._prefix.clear()
            self._prefix[(kind, prefix)] = (value, now)
            return value

    def add_sample(self, kind: str, value: float, now: float | None = None) -> None:
        with self._lock:
            self._samples[kind].append((now if now is not None else time.time(), value))

    def samples(self, kind: str, since: float) -> list[float]:
        with self._lock:
            return [v for t, v in self._samples[kind] if t >= since]

    def rank(self, kind: str, since: float, value: float) -> float | None:
        """Percentage of recent samples at or below `value` (None when there are none).

        Uses a sorted copy of the window rebuilt at most once a second."""
        now = time.monotonic()
        with self._lock:
            cached = self._sorted.get(kind)
            if cached is None or now - cached[0] >= 1.0:
                values = sorted(v for t, v in self._samples[kind] if t >= since)
                cached = (now, 0.0, values)
                self._sorted[kind] = cached
            values = cached[2]
        if not values:
            return None
        return 100.0 * bisect.bisect_right(values, value) / len(values)


class Pipeline:
    """Drain a `BoundedQueue` into a list of sinks on one worker thread."""

    def __init__(self, sinks: list[Sink], capacity: int = 10_000) -> None:
        self.queue = BoundedQueue(capacity)
        self.sinks = list(sinks)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def submit(self, event: Event) -> bool:
        return self.queue.put(event)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="jevmon-pipeline", daemon=True)
        self._thread.start()

    def _flush(self) -> None:
        for sink in self.sinks:
            flush = getattr(sink, "flush", None)
            if callable(flush):
                try:
                    flush()
                except Exception:  # noqa: BLE001, S112 - one broken sink must not stop the pipeline
                    continue

    def _run(self) -> None:
        last_flush = time.monotonic()
        while True:
            event = self.queue.get(timeout=0.25)
            if event is None:
                self._flush()
                last_flush = time.monotonic()
                if self._stop.is_set() and len(self.queue) == 0:
                    break
                continue
            for sink in self.sinks:
                try:
                    sink.emit(event)
                except Exception:  # noqa: BLE001, S112 - one broken sink must not stop the pipeline
                    continue
            if time.monotonic() - last_flush >= 0.25:  # a steady stream still reaches the store
                self._flush()
                last_flush = time.monotonic()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.queue.close()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        self._flush()
        for sink in self.sinks:
            sink.close()

    def stats(self) -> QueueStats:
        return self.queue.stats()


class StoreSink:
    """Persist verdict events always, and exec events when configured.

    Writes are batched (one transaction per flush). A verdict reused from the memo (`cached`)
    for a situation already stored within `repeat_seconds` is not stored again: it is counted in
    `coalesced` and the next stored row of that situation carries the count as `repeats`. At a few
    hundred execs a second the same situation repeats thousands of times; the judged verdict, the
    first sighting and one row per window are kept, and nothing is dropped uncounted.
    """

    def __init__(
        self,
        store: SqliteStore,
        store_exec: bool,
        repeat_seconds: float = 60.0,
        max_batch: int = 500,
    ) -> None:
        self.store = store
        self.store_exec = store_exec
        self.repeat_seconds = repeat_seconds
        self.max_batch = max_batch
        self.coalesced = 0
        self._batch: list[Event] = []
        self._seen: dict[str, tuple[float, int]] = {}  # signature -> (last stored, skipped since)

    def emit(self, event: Event) -> None:
        if event.kind == EVENT_KIND_VERDICT:
            event = self._coalesce(event)
            if event is None:
                return
        elif not (self.store_exec and event.kind == EVENT_KIND_EXEC):
            return
        self._batch.append(event)
        if len(self._batch) >= self.max_batch:
            self.flush()

    def _coalesce(self, event: Event) -> Event | None:
        if self.repeat_seconds <= 0 or not event.fields.get("cached"):
            return event
        signature = str(event.fields.get("signature", ""))
        last, skipped = self._seen.get(signature, (None, 0))
        if last is not None and event.time - last < self.repeat_seconds:
            self._seen[signature] = (last, skipped + 1)
            self.coalesced += 1
            return None
        if len(self._seen) > 20_000:
            self._seen.clear()
        self._seen[signature] = (event.time, 0)
        if skipped:
            return replace(event, fields={**event.fields, "repeats": skipped})
        return event

    def flush(self) -> None:
        if self._batch:
            batch, self._batch = self._batch, []
            self.store.add_events(batch)

    def close(self) -> None:
        self.flush()

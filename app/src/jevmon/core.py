"""Storage and the runner: a minimal SQLite store and a thread that drains a queue into sinks.

`SqliteStore` holds events, the learned counts used to build model context (exe/edge/template
frequencies, recently observed surprisal values) and human labels for later fine-tuning. WAL mode,
a schema version in `PRAGMA user_version`, day-based retention, safe from several threads.

`MemoryCounts` is the in-process counterpart of the count tables, for runs with no database.

`Pipeline` is the neutral runner: it owns a `BoundedQueue`, an ordered list of sinks and one worker
thread. Producers call `submit`; the worker moves each event to every sink.
"""

import json
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
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

    def _run(self) -> None:
        while True:
            event = self.queue.get(timeout=0.5)
            if event is None:
                if self._stop.is_set() and len(self.queue) == 0:
                    break
                continue
            for sink in self.sinks:
                try:
                    sink.emit(event)
                except Exception:  # noqa: BLE001, S112 - one broken sink must not stop the pipeline
                    continue

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.queue.close()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        for sink in self.sinks:
            sink.close()

    def stats(self) -> QueueStats:
        return self.queue.stats()


class StoreSink:
    """Persist verdict events always, and exec events when configured."""

    def __init__(self, store: SqliteStore, store_exec: bool) -> None:
        self.store = store
        self.store_exec = store_exec

    def emit(self, event: Event) -> None:
        if event.kind == EVENT_KIND_VERDICT or (self.store_exec and event.kind == EVENT_KIND_EXEC):
            self.store.add_event(event)

    def close(self) -> None:
        pass

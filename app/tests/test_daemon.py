"""Daemon server and client: requests, filters, labels, pause, subscription, sockets, shutdown."""

import asyncio
import contextlib
import json
import socket
import time
from pathlib import Path

import pytest

from jevmon.core import SqliteStore
from jevmon.daemon.client import DaemonClient, DaemonError
from jevmon.daemon.protocol import MAX_LINE
from jevmon.daemon.server import DaemonError as ServerError
from jevmon.daemon.server import DaemonServer
from jevmon.events import Event

pytestmark = pytest.mark.anyio


class FakeMonitor:
    """A stand-in for `Monitor`: only the surface `DaemonServer` touches."""

    def __init__(self) -> None:
        self.paused = False
        self.stopped = False
        self.reloads = 0

    def status(self) -> dict:
        return {
            "uptime": 1.0,
            "model_status": "fake-ready",
            "effective_interval": 0.05,
            "miss_ratio": 0.1,
            "observed": 3,
            "estimated_missed": 1,
            "judged": 2,
            "cached": 1,
            "dropped": 0,
            "queue_depth": 0,
            "paused": self.paused,
            "store": "mem",
            "running": True,
        }

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def reload(self) -> dict:
        self.reloads += 1
        return {"reloaded": True, "config": ""}

    def stop(self) -> None:
        self.stopped = True


def add_verdict(
    store: SqliteStore,
    *,
    severity: int = 2,
    context: str = "ctx",
    subject: str = "7",
    human: str | None = None,
    probs: dict | None = None,
) -> int:
    event_id = store.add_event(
        Event(
            time=time.time(),
            source="exec",
            subject=subject,
            kind="verdict",
            severity=severity,
            message=f"judge: crit {context}",
            fields={
                "context": context,
                "label": "crit",
                "confidence": 0.9,
                "judged": 1,
                "verdict": "model: crit 0.90",
                "probabilities": json.dumps(probs or {"crit": 0.9, "safe": 0.1}),
            },
        )
    )
    if human:
        store.add_label(event_id, human)
    return event_id


async def start(socket_path: Path, store: SqliteStore, monitor: FakeMonitor):
    server = DaemonServer(monitor, store, socket_path, poll_interval=0.05, stats_interval=100.0)
    task = asyncio.create_task(server.serve())
    for _ in range(300):
        if server.bound and socket_path.exists():
            return server, task
        await asyncio.sleep(0.01)
    raise AssertionError("server did not bind")


async def stop(server: DaemonServer, task: asyncio.Task) -> None:
    server.request_shutdown()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(task, timeout=3.0)


@pytest.fixture
def store(tmp_path):
    opened = SqliteStore(tmp_path / "e.db")
    try:
        yield opened
    finally:
        with contextlib.suppress(Exception):
            opened.close()


async def test_status_events_filters_labels_ack(tmp_path, store):
    socket_path = tmp_path / "s.sock"
    monitor = FakeMonitor()
    server, task = await start(socket_path, store, monitor)
    client = DaemonClient(socket_path)
    try:
        status = await client.status()
        assert status["model_status"] == "fake-ready" and status["paused"] is False

        first = add_verdict(store, severity=0, context="quiet", subject="1")
        second = add_verdict(store, severity=2, context="loud", subject="2")
        result = await client.events(limit=10)
        assert [e["id"] for e in result["events"]] == [first, second]
        assert result["events"][0]["probabilities"]["crit"] == 0.9

        alerts = await client.events(min_severity=2)
        assert [e["id"] for e in alerts["events"]] == [second]

        filtered = await client.events(text="loud")
        assert [e["id"] for e in filtered["events"]] == [second]

        labeled = await client.label(second, "true-positive")
        assert labeled["event"]["label"] == "true-positive"

        unlabeled = await client.events(unlabeled=True)
        assert [e["id"] for e in unlabeled["events"]] == [first]

        acked = await client.ack(second)
        assert acked["event"]["acked"] is True and acked["event"]["label"] == "true-positive"

        one = await client.event(first)
        assert one["event"]["subject"] == "1"
        with pytest.raises(DaemonError):
            await client.event(999_999)
        with pytest.raises(DaemonError):
            await client.request("status", bogus=1)
    finally:
        await client.close()
        await stop(server, task)


async def test_pause_resume_and_reload(tmp_path, store):
    socket_path = tmp_path / "s.sock"
    monitor = FakeMonitor()
    server, task = await start(socket_path, store, monitor)
    client = DaemonClient(socket_path)
    try:
        paused = await client.pause()
        assert paused["paused"] is True and monitor.paused is True
        resumed = await client.resume()
        assert resumed["paused"] is False and monitor.paused is False
        reloaded = await client.reload()
        assert reloaded["reloaded"] is True and monitor.reloads == 1
    finally:
        await client.close()
        await stop(server, task)


async def test_subscription_delivers_new_events_and_stats(tmp_path, store):
    socket_path = tmp_path / "s.sock"
    monitor = FakeMonitor()
    server, task = await start(socket_path, store, monitor)
    sub = DaemonClient(socket_path)
    try:
        stream = sub.subscribe()
        ready = await asyncio.wait_for(anext(stream), timeout=2.0)
        assert ready["stream"] == "ready"
        event_id = add_verdict(store, context="arrived")
        seen_event = None
        for _ in range(50):
            message = await asyncio.wait_for(anext(stream), timeout=2.0)
            if message["stream"] == "event":
                seen_event = message["event"]
                break
        assert seen_event is not None and seen_event["id"] == event_id
        assert seen_event["context"] == "arrived"
    finally:
        await sub.close()
        await stop(server, task)


async def test_oversized_request_is_rejected(tmp_path, store):
    socket_path = tmp_path / "s.sock"
    monitor = FakeMonitor()
    server, task = await start(socket_path, store, monitor)
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        writer.write(b'{"type":"status","x":"' + b"a" * MAX_LINE + b'"}\n')
        with contextlib.suppress(ConnectionError):
            await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        assert b"too large" in line
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        await stop(server, task)


async def test_stale_socket_is_replaced_and_live_socket_refused(tmp_path, store):
    socket_path = tmp_path / "s.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(socket_path))
    stale.close()  # leaves the file behind with nothing listening
    assert socket_path.exists()

    monitor = FakeMonitor()
    first, task = await start(socket_path, store, monitor)
    try:
        second = DaemonServer(monitor, store, socket_path)
        with pytest.raises(ServerError):
            second._prepare_socket()
        # A refused second daemon must not remove the live daemon's socket or take the lock.
        assert socket_path.exists() and second._owns_socket is False
        assert (await DaemonClient(socket_path).status())["model_status"] == "fake-ready"
    finally:
        await stop(first, task)


async def test_graceful_shutdown_removes_socket_and_stops_monitor(tmp_path, store):
    socket_path = tmp_path / "s.sock"
    monitor = FakeMonitor()
    _server, task = await start(socket_path, store, monitor)
    client = DaemonClient(socket_path)
    result = await client.shutdown()
    assert result == {"shutdown": True}
    await asyncio.wait_for(task, timeout=3.0)
    assert monitor.stopped is True
    assert not socket_path.exists()


async def test_client_reconnects_once_the_daemon_is_up(tmp_path, store):
    socket_path = tmp_path / "s.sock"
    client = DaemonClient(socket_path)
    with pytest.raises(DaemonError):
        await client.status()
    monitor = FakeMonitor()
    server, task = await start(socket_path, store, monitor)
    try:
        assert (await client.status())["model_status"] == "fake-ready"
    finally:
        await client.close()
        await stop(server, task)


def test_daemon_says_what_it_is_doing(tmp_path, capsys):
    async def run():
        server = DaemonServer(FakeMonitor(), SqliteStore(":memory:"), tmp_path / "s.sock")
        task = asyncio.create_task(server._report(heartbeat=0.05, tick=0.02))
        await asyncio.sleep(0.2)
        task.cancel()

    asyncio.run(run())
    err = capsys.readouterr().err
    assert "jevmond: model: fake-ready" in err
    assert "judged 2" in err and "cached 1" in err

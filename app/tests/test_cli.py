"""CLI: `jevmon exec --once` over a fake /proc tree, config errors and the label/ack commands."""

import json
import os
from pathlib import Path

from jevmon.cli import main
from jevmon.core import SqliteStore
from jevmon.events import Event


def stat_line(pid: int, comm: str, state: str, ppid: int, starttime: int) -> str:
    rest = [state, str(ppid), *(["0"] * 17), str(starttime)]
    return f"{pid} ({comm}) " + " ".join(rest) + "\n"


def fake_proc(root: Path, pids: list[tuple[int, list[str]]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "stat").write_text("cpu 1 2 3\nprocesses 1000\n")
    for pid, argv in pids:
        d = root / str(pid)
        d.mkdir(exist_ok=True)
        (d / "stat").write_text(stat_line(pid, "proc", "S", 1, 100))
        (d / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))
        (d / "status").write_text("Name:\tproc\nUid:\t1000\t1000\t1000\t1000\n")
    return root


def last_json(text: str) -> dict:
    return json.loads(text.strip().splitlines()[-1])


def test_once_prints_summary(tmp_path, capsys):
    root = fake_proc(tmp_path / "proc", [(100, ["/bin/a"]), (101, ["/bin/b"])])
    assert main(["exec", "--once", "--proc-root", str(root)]) == 0
    doc = last_json(capsys.readouterr().out)
    assert doc["kind"] == "summary" and doc["fields"]["tracked"] == 2


def test_once_persists_exec_events_when_configured(tmp_path):
    root = fake_proc(tmp_path / "proc", [(100, ["/bin/a"]), (101, ["/bin/b"])])
    cfg = tmp_path / "config.toml"
    cfg.write_text("[store]\nstore_exec_events = true\n")
    db = tmp_path / "e.db"
    assert (
        main(["exec", "--once", "--proc-root", str(root), "--config", str(cfg), "--db", str(db)])
        == 0
    )
    store = SqliteStore(db)
    try:
        assert store.count_events() == 2
    finally:
        store.close()


def test_once_does_not_create_state_without_db(tmp_path, capsys):
    root = fake_proc(tmp_path / "proc", [])
    assert main(["exec", "--once", "--proc-root", str(root)]) == 0
    assert last_json(capsys.readouterr().out)["fields"]["tracked"] == 0
    assert not (Path(os.environ["HOME"]) / ".local" / "state" / "jevmon").exists()


def test_missing_config_is_a_clean_error(tmp_path, capsys):
    root = fake_proc(tmp_path / "proc", [])
    code = main(["exec", "--once", "--proc-root", str(root), "--config", str(tmp_path / "no.toml")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_label_and_ack_write_labels(tmp_path):
    db = tmp_path / "e.db"
    store = SqliteStore(db)
    eid = store.add_event(
        Event(time=1.0, source="exec", subject="1", kind="verdict", severity=2, message="x")
    )
    store.close()
    assert main(["label", str(eid), "true-positive", "--db", str(db)]) == 0
    assert main(["ack", str(eid), "--db", str(db)]) == 0
    reopened = SqliteStore(db)
    try:
        assert reopened.labels(eid) == [("true-positive", "human"), ("ack", "human")]
    finally:
        reopened.close()


def test_ctl_status_without_a_daemon_is_a_clean_error(capsys):
    assert main(["ctl", "status"]) == 1
    assert "cannot connect" in capsys.readouterr().err

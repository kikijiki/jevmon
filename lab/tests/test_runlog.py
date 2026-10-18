import builtins
import json
import os
from pathlib import Path

import pytest

from jevmon_lab import metrics, runlog
from jevmon_lab.runlog import RunLog, class_table, score_scalars


class FakeBackend:
    def __init__(self) -> None:
        self.init_args: tuple | None = None
        self.logged: list[tuple[int, dict]] = []
        self.finished = 0

    def init(self, project, name, config) -> None:
        self.init_args = (project, name, config)

    def log(self, metrics, step) -> None:
        self.logged.append((step, metrics))

    def table(self, rows):
        return ("table", rows)

    def finish(self) -> None:
        self.finished += 1


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    return os.environ


def test_records_scalars_tables_and_finish():
    backend = FakeBackend()
    run = RunLog.start("p", "n", {"lr": 0.1, "out": Path("a/b"), "ids": (1, 2)}, backend=backend)
    assert run.enabled
    assert backend.init_args == ("p", "n", {"lr": 0.1, "out": "a/b", "ids": [1, 2]})
    run.log(3, loss=0.5, acc=0.25)
    run.table("per_class", [{"label": "safe", "recall": 1.0}], step=4)
    run.finish()
    assert backend.logged[0] == (3, {"loss": 0.5, "acc": 0.25})
    step, payload = backend.logged[1]
    assert step == 4
    assert payload["per_class"] == ("table", [{"label": "safe", "recall": 1.0}])
    assert backend.finished == 1
    run.log(5, loss=1.0)  # after finish: ignored
    run.finish()
    assert backend.finished == 1 and len(backend.logged) == 2


def test_disabled_is_a_no_op():
    backend = FakeBackend()
    run = RunLog.start("p", "n", {}, backend=backend, enabled=False)
    run.log(0, x=1.0)
    run.table("t", [{"a": 1}])
    run.finish()
    assert not run.enabled and backend.init_args is None and backend.logged == []


def test_no_trackio_is_a_no_op(monkeypatch, tmp_path, env, capsys):
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "trackio" or name.startswith("trackio."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.setattr(runlog.importlib, "import_module", lambda name: refuse(name))
    run = RunLog.start("p", "n", {"a": 1}, directory=tmp_path / "runs")
    run.log(0, x=1.0)
    run.table("t", [{"a": 1}])
    run.finish()
    assert not run.enabled
    assert "trackio not installed" in capsys.readouterr().err


def test_backend_failures_never_raise(capsys):
    class Broken(FakeBackend):
        def log(self, metrics, step) -> None:
            raise RuntimeError("disk full")

    run = RunLog.start("p", "n", {}, backend=Broken())
    run.log(0, x=1.0)
    assert not run.enabled
    run.log(1, x=2.0)
    run.finish()
    assert "disk full" in capsys.readouterr().err


def test_local_env_clears_sync_and_sets_dir(tmp_path, env):
    env["TRACKIO_SPACE_ID"] = "someone/space"
    env["TRACKIO_SERVER_URL"] = "https://example.invalid"
    env["SPACE_ID"] = "x"
    runlog.local_env(tmp_path)
    for key in ("TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "SPACE_ID", "TRACKIO_WEBHOOK_URL"):
        assert key not in env
    assert env["TRACKIO_DIR"] == str(tmp_path.resolve())
    assert env["GRADIO_SHARE"] == "false"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_jsonl_mirror_records_config_scalars_tables(monkeypatch, tmp_path, env):
    monkeypatch.setattr(runlog, "_load_trackio", lambda directory: None)
    path = tmp_path / "metrics.jsonl"
    run = RunLog.start("p", "n", {"lr": 0.2}, directory=tmp_path, jsonl=path)
    assert not run.enabled and run.recording
    run.log(3, **{"loss/train": 0.5, "macro_f1/dev": 0.7})
    run.table("per_class", [{"label": "safe", "recall": 1.0}], step=4)
    run.finish()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0] == {"project": "p", "run": "n", "kind": "config", "config": {"lr": 0.2}}
    assert records[1]["kind"] == "scalar" and records[1]["step"] == 3
    assert records[1]["metrics"]["loss/train"] == 0.5
    assert records[2]["kind"] == "table" and records[2]["name"] == "per_class"
    assert records[2]["rows"] == [{"label": "safe", "recall": 1.0}]
    assert records[3]["kind"] == "finish"


def test_score_scalars_carry_baselines_and_per_class():
    rows = [{"label": "safe", "verdict": "safe"}] * 6 + [
        {"label": "crit", "verdict": "safe"},
        {"label": "crit", "verdict": "crit"},
    ]
    score = metrics.score(rows)
    scalars = score_scalars("heldout", score)
    assert scalars["heldout/macro_f1"] == score["macro_f1"]
    assert scalars["heldout/always_safe_macro_f1"] == score["always_safe_macro_f1"]
    assert scalars["heldout/majority_rate"] == pytest.approx(0.75)
    assert scalars["heldout/recall/crit"] == pytest.approx(0.5)
    assert scalars["heldout/precision/crit"] == pytest.approx(1.0)
    table = class_table(score)
    assert {row["label"] for row in table} == {"safe", "crit"}

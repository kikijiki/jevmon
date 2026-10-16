"""Config loading: shipped defaults, user overrides, unknown keys and bad types."""

from pathlib import Path

import pytest

from jevmon.config import ConfigError, load_config, user_config_path
from jevmon.detect.context import ContextSettings
from jevmon.detect.judge import JudgeSettings
from jevmon.sources.exec import ExecSettings


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_defaults_come_from_shipped_toml():
    cfg = load_config()
    assert cfg.exec.interval == ExecSettings().interval
    assert cfg.exec.cpu_budget == 0.03 and cfg.exec.summary_seconds == 30.0
    assert cfg.context.model_max == 1000
    assert tuple(cfg.judge.categories) == ("safe", "warn", "crit")
    assert "best describes" in cfg.judge.question
    assert cfg.store.retention_days == 14 and cfg.store.store_exec_events is False


def test_exec_and_context_overrides_keep_other_defaults(tmp_path):
    cfg = load_config(write(tmp_path, "[exec]\ninterval = 0.2\n[context]\nmodel_max = 500\n"))
    assert cfg.exec.interval == 0.2 and cfg.context.model_max == 500
    assert cfg.exec.cpu_budget == 0.03 and cfg.context.sibling_seconds == 10.0


def test_judge_section_overrides_question_and_class_text(tmp_path):
    text = """
[judge]
question = "Is this expected?"
[judge.categories]
safe = "fine"
crit = "not fine"
"""
    cfg = load_config(write(tmp_path, text))
    assert cfg.judge.question == "Is this expected?"
    assert cfg.judge.categories == {"safe": "fine", "crit": "not fine"}


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "[exec]\nintervall = 0.2\n"))
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "[judge]\nquestionn = 'x'\n"))
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "[context]\nwindow_seconds = 30\n"))
    with pytest.raises(ConfigError, match="unknown section"):
        load_config(write(tmp_path, "[nope]\nx = 1\n"))


def test_wrong_type_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="interval"):
        load_config(write(tmp_path, '[exec]\ninterval = "fast"\n'))
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "[judge]\nalert_band = 0.9\n"))
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "[judge.severity]\nsafe = 0\n"))
    with pytest.raises(ConfigError, match="unknown class"):
        load_config(write(tmp_path, '[judge.categories]\nsafe = "a"\nodd = "b"\n'))
    with pytest.raises(ConfigError, match="categories"):
        load_config(write(tmp_path, "[judge]\ncategories = 3\n"))


def test_missing_explicit_path_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")


def test_user_path_uses_xdg_config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    target = tmp_path / "jevmon" / "config.toml"
    target.parent.mkdir()
    target.write_text("[exec]\ninterval = 0.5\n")
    assert user_config_path() == target
    assert load_config().exec.interval == 0.5


def test_dataclass_defaults_match_shipped_toml():
    cfg = load_config()
    assert ContextSettings().model_max == 1000
    assert JudgeSettings().memo_ttl_seconds == 300.0
    assert cfg.judge.question == JudgeSettings().question
    assert cfg.judge.checkpoint == JudgeSettings().checkpoint
    assert cfg.judge.checkpoint_dir == JudgeSettings().checkpoint_dir
    assert cfg.judge.categories == JudgeSettings().categories
    assert tuple(cfg.context.order) == ContextSettings().order
    assert cfg.context.sep == ContextSettings().sep


def test_judge_device_and_dtype_defaults_and_validation(tmp_path):
    cfg = load_config()
    assert cfg.judge.device == "auto" and cfg.judge.dtype == "auto"
    assert cfg.judge.batch_size == 0 and cfg.judge.threads_gpu == 2
    with pytest.raises(ConfigError, match="device"):
        load_config(write(tmp_path, "[judge]\ndevice = 'gup'\n"))
    with pytest.raises(ConfigError, match="dtype"):
        load_config(write(tmp_path, "[judge]\ndtype = 'int8'\n"))


def test_judge_device_dtype_threads_gpu_overrides(tmp_path):
    cfg = load_config(write(tmp_path, "[judge]\ndevice = 'gpu'\ndtype = 'fp16'\nthreads_gpu = 1\n"))
    assert (cfg.judge.device, cfg.judge.dtype, cfg.judge.threads_gpu) == ("gpu", "fp16", 1)

"""READMEs: every embedded TOML example must be a real, loadable jevmon config."""

import re
import tomllib
from pathlib import Path

import pytest

from jevmon.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
APP_README = ROOT / "app" / "README.md"
BEGIN = "<!-- screenshots:begin -->"
END = "<!-- screenshots:end -->"


def toml_blocks() -> list[str]:
    """TOML examples of the app README (the configuration reference) and the root README."""
    blocks: list[str] = []
    for path in (README, APP_README):
        text = path.read_text(encoding="utf-8")
        blocks += re.findall(r"```toml\n(.*?)```", text, flags=re.DOTALL)
    return blocks


def test_the_readmes_show_config_examples_and_keep_the_screenshot_block():
    text = README.read_text(encoding="utf-8")
    assert len(toml_blocks()) >= 4, "the configuration section should show several examples"
    assert BEGIN in text and END in text, "the screenshots block markers are required"
    assert "## Design principle" in APP_README.read_text(encoding="utf-8"), (
        "the design-principle section was dropped from the app README"
    )


def test_every_toml_block_in_the_readme_parses_and_loads(tmp_path):
    for index, block in enumerate(toml_blocks()):
        tomllib.loads(block)  # valid TOML at all
        path = tmp_path / f"example-{index}.toml"
        path.write_text(block, encoding="utf-8")
        config = load_config(path)  # ... and a config the loader accepts
        assert config.judge.mode in ("laya", "fake"), f"block {index}"
        assert config.exec.source in ("ebpf", "proc", "replay"), f"block {index}"


def test_a_typo_in_an_example_would_be_caught(tmp_path):
    path = tmp_path / "typo.toml"
    path.write_text("[judge]\nalert_ban = 0.95\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)

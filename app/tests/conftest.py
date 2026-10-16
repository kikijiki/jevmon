"""Test isolation: nothing a test does may touch the user's real jevmon state or config."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Default paths (`~/.local/state/jevmon/*`, `~/.config/jevmon`, runtime socket) use a temp HOME."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(home / "run"))


@pytest.fixture
def anyio_backend():
    """Run async tests on asyncio only (trio is not a dependency)."""
    return "asyncio"

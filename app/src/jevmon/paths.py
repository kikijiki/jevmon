"""Path resolution shared by the CLI, the daemon and the client (isolated by HOME in tests)."""

import os
from pathlib import Path


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "jevmon"


def default_socket_path() -> Path:
    """`$XDG_RUNTIME_DIR/jevmon/jevmond.sock`, else a socket under the state dir."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "jevmon" / "jevmond.sock"
    return state_dir() / "jevmond.sock"


__all__ = ["default_socket_path", "state_dir"]

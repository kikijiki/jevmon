//! Default control-socket path, mirroring `src/jevmon/paths.py`: `$XDG_RUNTIME_DIR/jevmon/
//! jevmond.sock`, else a socket under the state dir (`$XDG_STATE_HOME` or `~/.local/state`).

use std::path::PathBuf;

pub fn default_socket_path() -> PathBuf {
    if let Some(runtime) = std::env::var_os("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime).join("jevmon").join("jevmond.sock");
    }
    state_dir().join("jevmond.sock")
}

fn state_dir() -> PathBuf {
    let base = match std::env::var_os("XDG_STATE_HOME") {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/state"),
    };
    base.join("jevmon")
}

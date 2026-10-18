//! `jevmon-tui`: the Ratatui client for a running `jevmond`.
//!
//! The library holds everything testable — the wire protocol, the sanitizer, the view state and
//! the renderer; [`main`] is only terminal setup, the key/event loop and the channel plumbing to
//! the daemon connection task.

pub mod allow;
pub mod client;
pub mod format;
pub mod list;
pub mod paths;
pub mod protocol;
pub mod sanitize;
pub mod state;
pub mod theme;
pub mod ui;

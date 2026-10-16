//! Terminal setup, the event loop and the plumbing between the UI thread and the daemon task.
//!
//! The UI thread never blocks: it drains daemon messages, polls keys with a short timeout and
//! redraws. The daemon connection runs on a background thread with its own tokio runtime and
//! keeps retrying forever, so a dead or restarting daemon is a badge and a countdown, never a
//! crash.

use std::io::{self, Stdout};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use anyhow::{Result, bail};
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use jevmon_tui::client::{self, DaemonCmd, DaemonMsg};
use jevmon_tui::state::App;
use jevmon_tui::theme::Theme;
use jevmon_tui::ui;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;

const TICK: Duration = Duration::from_millis(100);

struct Args {
    socket: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut socket = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--socket" | "-s" => match args.next() {
                Some(path) => socket = Some(PathBuf::from(path)),
                None => bail!("--socket needs a path"),
            },
            "--help" | "-h" => {
                println!("jevmon-tui — Ratatui client for the jevmon daemon");
                println!();
                println!(
                    "  --socket PATH   control socket (default: $XDG_RUNTIME_DIR/jevmon/jevmond.sock)"
                );
                println!("  --help          this text");
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?} (try --help)"),
        }
    }
    Ok(Args {
        socket: socket.unwrap_or_else(jevmon_tui::paths::default_socket_path),
    })
}

/// Restores the terminal on any exit path, including a panic mid-draw.
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            LeaveAlternateScreen,
            DisableMouseCapture,
            crossterm::cursor::Show
        );
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let theme = Theme::from_env();

    let (out_tx, mut out_rx) = mpsc::channel::<DaemonMsg>(512);
    let (cmd_tx, cmd_rx) = mpsc::channel::<DaemonCmd>(64);
    let socket = args.socket.clone();
    let daemon = thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        runtime.block_on(client::run(socket, cmd_rx, out_tx));
    });

    let result = run(
        App::new(args.socket.display().to_string()),
        theme,
        cmd_tx,
        &mut out_rx,
    );
    // Dropping the receiver makes the daemon task notice the quit even if it is mid-retry.
    drop(out_rx);
    let _ = daemon.join();
    result
}

fn run(
    mut app: App,
    theme: Theme,
    cmd_tx: mpsc::Sender<DaemonCmd>,
    out_rx: &mut mpsc::Receiver<DaemonMsg>,
) -> Result<()> {
    app.theme = theme;
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let guard = TerminalGuard;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let outcome = event_loop(&mut terminal, &mut app, &cmd_tx, out_rx);
    drop(guard);
    outcome
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
    cmd_tx: &mpsc::Sender<DaemonCmd>,
    out_rx: &mut mpsc::Receiver<DaemonMsg>,
) -> Result<()> {
    loop {
        while let Ok(msg) = out_rx.try_recv() {
            app.handle_msg(msg);
        }
        if event::poll(TICK)? {
            match event::read()? {
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    for cmd in app.handle_key(key) {
                        if cmd_tx.try_send(cmd).is_err() {
                            app.flash("daemon is busy — try again");
                        }
                    }
                }
                Event::Mouse(mouse) => match mouse.kind {
                    event::MouseEventKind::ScrollDown => app.scroll(3),
                    event::MouseEventKind::ScrollUp => app.scroll(-3),
                    event::MouseEventKind::Down(event::MouseButton::Left) => {
                        app.click(mouse.column, mouse.row);
                    }
                    _ => {}
                },
                _ => {} // resize and anything else: the next draw adapts
            }
        }
        if app.quit {
            break;
        }
        terminal.draw(|frame| ui::draw(frame, app))?;
    }
    Ok(())
}

//! The daemon connection task: a thin, patient client of the control socket.
//!
//! One task owns two connections — a request connection (`events`, `label`, `ack`, `pause`,
//! `resume`) and a `subscribe` stream — and talks to the UI through channels. It never decides
//! anything: it forwards events, stats and command results verbatim, and reports failures so the
//! UI can show them.
//!
//! It is built to outlive the daemon: any failure closes both connections, reports the reason and
//! the backoff delay (1s → 2s → 4s → 8s → 15s), then retries forever. Every fresh connection
//! reloads the latest batch of events and resubscribes from the last event id it has seen, so a
//! daemon restart leaves no gap.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tokio::time::timeout;

use crate::protocol::{
    self, EventPayload, MAX_LIMIT, MAX_REPLY, Message, Stats, encode_line, parse_line,
};

const BACKOFF: [u64; 5] = [1, 2, 4, 8, 15];

/// UI -> daemon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaemonCmd {
    Label { event_id: u64, label: String },
    Ack { event_id: u64 },
    Pause,
    Resume,
    Reconnect,
}

/// daemon -> UI.
#[derive(Debug, Clone)]
pub enum DaemonMsg {
    Connected,
    /// The latest batch of events, loaded on every (re)connect.
    Initial(Vec<EventPayload>),
    Event(EventPayload),
    Stats(Stats),
    /// The event payload returned by a `label`/`ack` request.
    Labeled(EventPayload),
    /// A request was answered with an error (or could not be sent).
    Failed(String),
    /// The connection is gone; the client will retry after `retry_in` seconds.
    Disconnected {
        reason: String,
        retry_in: u64,
    },
}

/// Run the connection loop until the UI goes away or the runtime is dropped.
///
/// `cmds` closing means the UI quit; `out` failing means the UI stopped draining.
pub async fn run(
    socket: PathBuf,
    mut cmds: mpsc::Receiver<DaemonCmd>,
    out: mpsc::Sender<DaemonMsg>,
) {
    let mut attempt = 0usize;
    loop {
        match session(&socket, &mut cmds, &out).await {
            Ok(()) => return, // the UI is gone
            Err(reason) => {
                let delay = BACKOFF[attempt.min(BACKOFF.len() - 1)];
                attempt = attempt.saturating_add(1);
                if out
                    .send(DaemonMsg::Disconnected {
                        reason,
                        retry_in: delay,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
                // Sleep the backoff, but wake early for an explicit `r` (Reconnect) or a quit.
                loop {
                    match timeout(Duration::from_secs(delay), cmds.recv()).await {
                        Err(_) => break, // slept it out: retry now
                        Ok(None) => return,
                        Ok(Some(DaemonCmd::Reconnect)) => {
                            attempt = 0;
                            break;
                        }
                        Ok(Some(_)) => {} // the daemon is down; drop the command and keep waiting
                    }
                }
            }
        }
    }
}

/// One connection's lifetime: connect, load, subscribe, stream. `Err` is a reason to retry;
/// `Ok(())` only when the UI has gone away.
async fn session(
    socket: &Path,
    cmds: &mut mpsc::Receiver<DaemonCmd>,
    out: &mpsc::Sender<DaemonMsg>,
) -> Result<(), String> {
    let mut cmd = BufReader::new(connect(socket).await?);
    let sub = connect(socket).await?;
    if out.send(DaemonMsg::Connected).await.is_err() {
        return Ok(());
    }

    // Latest batch first: it also tells us the id to resubscribe from, so nothing between the
    // reload and the subscription is lost.
    let mut next_id: u64 = 1;
    let request = protocol::events_request(next_id, 0, 0, MAX_LIMIT, false, None);
    let result = request_once(&mut cmd, next_id, request).await?;
    let result = result?;
    next_id += 1;
    let batch: EventsResult =
        serde_json::from_value(result).map_err(|e| format!("bad events reply: {e}"))?;
    let since_id = batch.events.iter().map(|e| e.id).max();
    if out.send(DaemonMsg::Initial(batch.events)).await.is_err() {
        return Ok(());
    }

    let mut sub = BufReader::new(sub);
    let subscribe = protocol::subscribe_request(since_id);
    sub.get_mut()
        .write_all(&encode_line(subscribe))
        .await
        .map_err(|e| format!("subscribe: {e}"))?;
    match read_message(&mut sub).await? {
        Message::Ready => {}
        other => return Err(format!("expected ready, got {other:?}")),
    }

    loop {
        tokio::select! {
            incoming = read_message(&mut sub) => match incoming? {
                Message::Event(event) => {
                    if out.send(DaemonMsg::Event(*event)).await.is_err() {
                        return Ok(());
                    }
                }
                Message::Stats(stats) => {
                    if out.send(DaemonMsg::Stats(*stats)).await.is_err() {
                        return Ok(());
                    }
                }
                Message::Ready | Message::UnknownStream | Message::Response { .. } => {}
            },
            command = cmds.recv() => match command {
                None => return Ok(()), // the UI quit
                Some(DaemonCmd::Reconnect) => return Err("reconnecting".to_string()),
                Some(command) => {
                    let message = dispatch(&mut cmd, &mut next_id, command).await?;
                    if out.send(message).await.is_err() {
                        return Ok(());
                    }
                }
            }
        }
    }
}

async fn dispatch(
    cmd: &mut BufReader<UnixStream>,
    next_id: &mut u64,
    command: DaemonCmd,
) -> Result<DaemonMsg, String> {
    let (request, is_status) = match command {
        DaemonCmd::Label { event_id, label } => {
            (protocol::label_request(*next_id, event_id, &label), false)
        }
        DaemonCmd::Ack { event_id } => (protocol::ack_request(*next_id, event_id), false),
        DaemonCmd::Pause => (protocol::pause_request(*next_id), true),
        DaemonCmd::Resume => (protocol::resume_request(*next_id), true),
        DaemonCmd::Reconnect => unreachable!("handled by the caller"),
    };
    let id = *next_id;
    *next_id += 1;
    let result = match request_once(cmd, id, request).await {
        Err(fatal) => return Err(fatal), // I/O or protocol failure: reconnect
        Ok(Err(daemon_error)) => return Ok(DaemonMsg::Failed(daemon_error)), // the daemon said no
        Ok(Ok(result)) => result,
    };
    if is_status {
        let stats: Stats =
            serde_json::from_value(result).map_err(|e| format!("bad status reply: {e}"))?;
        return Ok(DaemonMsg::Stats(stats));
    }
    #[derive(Deserialize)]
    struct EventResult {
        event: EventPayload,
    }
    match serde_json::from_value::<EventResult>(result) {
        Ok(EventResult { event }) => Ok(DaemonMsg::Labeled(event)),
        Err(e) => Ok(DaemonMsg::Failed(format!("bad event reply: {e}"))),
    }
}

/// Write one request and read exactly one reply for it.
///
/// The outer `Result` is fatal (the connection is broken or the daemon spoke nonsense); the
/// inner one is the daemon's own `ok:false` error for this request.
async fn request_once(
    stream: &mut BufReader<UnixStream>,
    id: u64,
    request: serde_json::Value,
) -> Result<Result<serde_json::Value, String>, String> {
    stream
        .get_mut()
        .write_all(&encode_line(request))
        .await
        .map_err(|e| format!("send: {e}"))?;
    match read_message(stream).await? {
        Message::Response {
            id: reply_id,
            ok,
            result,
            error,
        } => {
            if let Some(reply) = reply_id
                && reply != id
            {
                return Err(format!("reply id {reply} does not match request {id}"));
            }
            if ok {
                Ok(Ok(result))
            } else {
                Ok(Err(error.unwrap_or_else(|| "request failed".to_string())))
            }
        }
        other => Err(format!("unexpected reply: {other:?}")),
    }
}

async fn connect(path: &Path) -> Result<UnixStream, String> {
    UnixStream::connect(path)
        .await
        .map_err(|e| format!("cannot connect to {}: {e}", path.display()))
}

/// Read one newline-terminated JSON message, bounded so a hostile or broken daemon cannot
/// balloon memory (the server's own reply cap is `MAX_REPLY`).
async fn read_message(reader: &mut BufReader<UnixStream>) -> Result<Message, String> {
    let mut line: Vec<u8> = Vec::new();
    loop {
        let available = reader.fill_buf().await.map_err(|e| format!("read: {e}"))?;
        if available.is_empty() {
            if line.is_empty() {
                return Err("daemon closed the connection".to_string());
            }
            break; // EOF mid-line: parse what we have
        }
        if let Some(pos) = available.iter().position(|&b| b == b'\n') {
            line.extend_from_slice(&available[..=pos]);
            reader.consume(pos + 1);
            break;
        }
        if line.len() + available.len() > MAX_REPLY {
            return Err("reply too large".to_string());
        }
        let len = available.len();
        line.extend_from_slice(available);
        reader.consume(len);
    }
    while line.last().is_some_and(|&b| b == b'\n' || b == b'\r') {
        line.pop();
    }
    parse_line(&line).map_err(|e| e.to_string())
}

#[derive(Debug, Default, Deserialize)]
struct EventsResult {
    #[serde(default)]
    events: Vec<EventPayload>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn backoff_is_bounded() {
        assert_eq!(BACKOFF, [1, 2, 4, 8, 15]);
        let _ = crate::paths::default_socket_path();
    }

    #[tokio::test]
    async fn run_against_a_missing_socket_reports_and_retries() {
        let (cmds_tx, cmds_rx) = mpsc::channel::<DaemonCmd>(4);
        let (out_tx, mut out_rx) = mpsc::channel::<DaemonMsg>(16);
        let socket =
            std::env::temp_dir().join(format!("jevmon-missing-{}.sock", std::process::id()));
        let task = tokio::spawn(run(socket, cmds_rx, out_tx));
        match out_rx.recv().await {
            Some(DaemonMsg::Disconnected { reason, retry_in }) => {
                assert!(reason.contains("cannot connect to"), "reason: {reason}");
                assert_eq!(retry_in, 1);
            }
            other => panic!("expected a disconnect, got {other:?}"),
        }
        // `r` (Reconnect) wakes the backoff sleep immediately and restarts the backoff.
        cmds_tx.send(DaemonCmd::Reconnect).await.unwrap();
        match out_rx.recv().await {
            Some(DaemonMsg::Disconnected { retry_in, .. }) => assert_eq!(retry_in, 1),
            other => panic!("expected the next attempt, got {other:?}"),
        }
        drop(cmds_tx);
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("client stops when the UI goes away")
            .unwrap();
    }
}

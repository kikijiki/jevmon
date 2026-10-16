//! Integration test: the real client against a fake daemon — a `tokio` `UnixListener` speaking
//! the newline-JSON protocol. Covers connect, the initial batch, subscribe (from the right id),
//! the event/stats stream, label and ack round trips, and a clean shutdown.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::time::timeout;

use jevmon_tui::client::{self, DaemonCmd, DaemonMsg};

fn event_payload(id: u64) -> Value {
    json!({
        "id": id,
        "time": 1_728_000_000.0 + id as f64,
        "source": "exec",
        "subject": id.to_string(),
        "kind": "verdict",
        "severity": 2,
        "message": format!("judge verdict for event {id}"),
        "context": "exe: /usr/bin/curl  basename: curl",
        "verdict": "model: crit 0.90",
        "class": "crit",
        "confidence": 0.9,
        "probabilities": {"crit": 0.9, "safe": 0.1},
        "judged": true,
        "cached": false,
        "available": true,
        "label": null,
        "labels": [],
        "acked": false,
    })
}

fn stats(paused: bool) -> Value {
    json!({
        "uptime": 5.0,
        "model_status": "ready",
        "effective_interval": 0.05,
        "miss_ratio": 0.1,
        "judged": 3,
        "cached": 1,
        "dropped": 0,
        "queue_depth": 0,
        "paused": paused,
    })
}

/// Accept connections forever; answer requests and stream one event after `subscribe`.
async fn fake_daemon(path: PathBuf, seen: Arc<Mutex<Vec<Value>>>) {
    let listener = UnixListener::bind(&path).expect("bind fake socket");
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let seen = Arc::clone(&seen);
        tokio::spawn(async move {
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {}
                }
                let Ok(request) = serde_json::from_str::<Value>(line.trim()) else {
                    return;
                };
                seen.lock().unwrap().push(request.clone());
                let id = request["id"].clone();
                let kind = request["type"].as_str().unwrap_or_default().to_string();
                let reply = match kind.as_str() {
                    "events" => json!({"v":4,"id":id,"ok":true,"result":{
                        "events":[event_payload(10), event_payload(11)],
                        "count":2,"more":false}}),
                    "label" => {
                        let event_id = request["event_id"].as_u64().unwrap_or(0);
                        if event_id == 999 {
                            json!({"v":4,"id":id,"ok":false,"error":"no event 999"})
                        } else {
                            let mut event = event_payload(event_id);
                            event["label"] = request["label"].clone();
                            event["labels"] = json!([request["label"]]);
                            json!({"v":4,"id":id,"ok":true,"result":{"event":event}})
                        }
                    }
                    "ack" => {
                        let mut event = event_payload(request["event_id"].as_u64().unwrap_or(0));
                        event["label"] = json!("true-positive");
                        event["labels"] = json!(["true-positive", "ack"]);
                        event["acked"] = json!(true);
                        json!({"v":4,"id":id,"ok":true,"result":{"event":event}})
                    }
                    "status" | "pause" | "resume" => json!({
                        "v":4,"id":id,"ok":true,"result":stats(kind == "pause")
                    }),
                    "subscribe" => {
                        write(
                            &mut reader,
                            &json!({"v":4,"stream":"ready","result":{"subscribed":true}}),
                        )
                        .await;
                        write(
                            &mut reader,
                            &json!({"v":4,"stream":"event","event":event_payload(12)}),
                        )
                        .await;
                        write(
                            &mut reader,
                            &json!({"v":4,"stream":"stats","stats":stats(false)}),
                        )
                        .await;
                        // Keep the stream open until the client goes away.
                        let mut sink = Vec::new();
                        let _ = reader.read_to_end(&mut sink).await;
                        return;
                    }
                    other => json!({"v":4,"id":id,"ok":false,"error":format!("unknown {other}")}),
                };
                write(&mut reader, &reply).await;
            }
        });
    }
}

async fn write(reader: &mut BufReader<UnixStream>, message: &Value) {
    let mut line = serde_json::to_vec(message).expect("encode");
    line.push(b'\n');
    let _ = reader.get_mut().write_all(&line).await;
}

/// The `since_id` the client subscribed with, read outside any async context.
fn subscription_since(seen: &Arc<Mutex<Vec<Value>>>) -> u64 {
    let recorded = seen.lock().unwrap();
    let subscribe = recorded
        .iter()
        .find(|r| r["type"] == "subscribe")
        .expect("the client subscribed");
    subscribe["since_id"].as_u64().expect("since_id")
}

async fn next(out: &mut mpsc::Receiver<DaemonMsg>) -> DaemonMsg {
    timeout(std::time::Duration::from_secs(5), out.recv())
        .await
        .expect("client message within 5s")
        .expect("client still running")
}

#[tokio::test]
async fn connect_subscribe_load_label_and_ack() {
    let socket = std::env::temp_dir().join(format!("jevmon-fake-{}-a.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);
    let seen = Arc::new(Mutex::new(Vec::new()));
    tokio::spawn(fake_daemon(socket.clone(), Arc::clone(&seen)));

    let (cmd_tx, cmd_rx) = mpsc::channel::<DaemonCmd>(8);
    let (out_tx, mut out_rx) = mpsc::channel::<DaemonMsg>(64);
    let client = tokio::spawn(client::run(socket.clone(), cmd_rx, out_tx));

    // connect -> initial batch -> subscribed -> stream
    assert!(matches!(next(&mut out_rx).await, DaemonMsg::Connected));
    match next(&mut out_rx).await {
        DaemonMsg::Initial(events) => {
            let ids: Vec<u64> = events.iter().map(|e| e.id).collect();
            assert_eq!(ids, [10, 11]);
        }
        other => panic!("expected the initial batch, got {other:?}"),
    }
    match next(&mut out_rx).await {
        DaemonMsg::Event(event) => assert_eq!(event.id, 12),
        other => panic!("expected a streamed event, got {other:?}"),
    }
    match next(&mut out_rx).await {
        DaemonMsg::Stats(stats) => assert_eq!(stats.model_status, "ready"),
        other => panic!("expected stats, got {other:?}"),
    }

    // the subscription resumed from the last id of the initial batch (exclusive, as the store does)
    assert_eq!(subscription_since(&seen), 11);

    // label round trip
    cmd_tx
        .send(DaemonCmd::Label {
            event_id: 11,
            label: "true-positive".to_string(),
        })
        .await
        .unwrap();
    match next(&mut out_rx).await {
        DaemonMsg::Labeled(event) => {
            assert_eq!(event.id, 11);
            assert_eq!(event.label.as_deref(), Some("true-positive"));
        }
        other => panic!("expected the labelled event, got {other:?}"),
    }

    // ack round trip
    cmd_tx.send(DaemonCmd::Ack { event_id: 11 }).await.unwrap();
    match next(&mut out_rx).await {
        DaemonMsg::Labeled(event) => {
            assert!(event.acked);
            assert_eq!(event.labels, ["true-positive", "ack"]);
        }
        other => panic!("expected the acked event, got {other:?}"),
    }

    // pause returns the daemon's status, not a verdict of ours
    cmd_tx.send(DaemonCmd::Pause).await.unwrap();
    match next(&mut out_rx).await {
        DaemonMsg::Stats(stats) => assert!(stats.paused),
        other => panic!("expected status after pause, got {other:?}"),
    }

    // dropping the command channel (the UI quit) ends the client cleanly
    drop(cmd_tx);
    timeout(std::time::Duration::from_secs(5), client)
        .await
        .expect("client exits promptly")
        .expect("client did not panic");
    let _ = std::fs::remove_file(&socket);
}

#[tokio::test]
async fn a_daemon_error_reply_is_reported_without_dropping_the_stream() {
    let socket = std::env::temp_dir().join(format!("jevmon-fake-{}-b.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);
    let seen = Arc::new(Mutex::new(Vec::new()));
    tokio::spawn(fake_daemon(socket.clone(), Arc::clone(&seen)));

    let (cmd_tx, cmd_rx) = mpsc::channel::<DaemonCmd>(8);
    let (out_tx, mut out_rx) = mpsc::channel::<DaemonMsg>(64);
    tokio::spawn(client::run(socket.clone(), cmd_rx, out_tx));

    // drain the handshake: connected -> initial batch -> streamed event -> stats
    while !matches!(next(&mut out_rx).await, DaemonMsg::Stats(_)) {}

    // the fake daemon answers ok:false for this id: reported, stream untouched
    cmd_tx
        .send(DaemonCmd::Label {
            event_id: 999,
            label: "x".to_string(),
        })
        .await
        .unwrap();
    match next(&mut out_rx).await {
        DaemonMsg::Failed(error) => assert!(error.contains("no event 999"), "error: {error}"),
        other => panic!("expected a failure report, got {other:?}"),
    }

    // the connection is still healthy: the next request is answered normally
    cmd_tx.send(DaemonCmd::Ack { event_id: 11 }).await.unwrap();
    match next(&mut out_rx).await {
        DaemonMsg::Labeled(event) => assert_eq!(event.id, 11),
        other => panic!("expected the acked event, got {other:?}"),
    }
    drop(cmd_tx);
    let _ = std::fs::remove_file(&socket);
}

/// The client must never write anything but well-formed protocol lines.
#[tokio::test]
async fn requests_reach_the_wire_as_one_line_each() {
    let socket = std::env::temp_dir().join(format!("jevmon-fake-{}-c.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);
    let seen = Arc::new(Mutex::new(Vec::new()));
    tokio::spawn(fake_daemon(socket.clone(), Arc::clone(&seen)));

    let (cmd_tx, cmd_rx) = mpsc::channel::<DaemonCmd>(8);
    let (out_tx, mut out_rx) = mpsc::channel::<DaemonMsg>(64);
    tokio::spawn(client::run(socket.clone(), cmd_rx, out_tx));

    while !matches!(next(&mut out_rx).await, DaemonMsg::Initial(_)) {}
    cmd_tx
        .send(DaemonCmd::Label {
            event_id: 10,
            label: "a label with spaces".to_string(),
        })
        .await
        .unwrap();
    while !matches!(next(&mut out_rx).await, DaemonMsg::Labeled(_)) {}
    drop(cmd_tx);

    let recorded = seen.lock().unwrap();
    let label = recorded
        .iter()
        .find(|r| r["type"] == "label")
        .expect("label request");
    assert_eq!(label["event_id"], 10);
    assert_eq!(label["label"], "a label with spaces");
    assert_eq!(label["v"], 4);
    assert_eq!(label["id"], 2, "request ids are ordered per connection");
    drop(recorded);
    let _ = std::fs::remove_file(&socket);
}

//! The jevmond wire protocol, version 5: newline-delimited JSON over a Unix domain socket.
//!
//! This module speaks exactly what `src/jevmon/daemon/protocol.py` defines. Outgoing requests
//! carry only the fields the server accepts (it rejects unknown ones); incoming messages are
//! parsed into strict typed structs — required fields must be present and correctly typed,
//! unknown fields are tolerated, anything else is a protocol error.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Deserializer};
use serde_json::{Value, json};

pub const PROTOCOL_VERSION: u32 = 5;
/// Bytes of one request line (mirrors the server).
pub const MAX_LINE: usize = 131_072;
/// Bytes of one reply line the client will read.
pub const MAX_REPLY: usize = 8_000_000;
pub const MAX_LIMIT: u32 = 100;
pub const DEFAULT_LIMIT: u32 = 50;
pub const MAX_TEXT: usize = 512;
pub const MAX_LABEL: usize = 128;

/// A reply or stream line we could not (or will not) understand.
#[derive(Debug, Clone, PartialEq)]
pub struct ProtocolError(pub String);

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProtocolError {}

/// One event as the daemon presents it (see `protocol.event_payload`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EventPayload {
    pub id: u64,
    pub time: f64,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub subject: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub severity: i64,
    #[serde(default)]
    pub message: String,
    /// The short redacted command line (`basename args...`); empty from an older daemon.
    #[serde(default)]
    pub command: String,
    /// The exe path the daemon matched on, and the redacted space-joined argv (argv[0] first).
    #[serde(default)]
    pub exe: String,
    #[serde(default)]
    pub argv: String,
    #[serde(default)]
    pub context: String,
    #[serde(default)]
    pub verdict: String,
    #[serde(default)]
    pub class: String,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default, deserialize_with = "de_probabilities")]
    pub probabilities: BTreeMap<String, f64>,
    #[serde(default)]
    pub judged: bool,
    #[serde(default)]
    pub cached: bool,
    #[serde(default)]
    pub available: bool,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub acked: bool,
}

/// `probabilities` is a JSON object of class -> number on the wire; looser producers
/// may send it as a JSON string containing the object. Anything else yields an empty map.
fn de_probabilities<'de, D: Deserializer<'de>>(d: D) -> Result<BTreeMap<String, f64>, D::Error> {
    let raw = Option::<Value>::deserialize(d)?;
    Ok(match raw {
        None | Some(Value::Null) => BTreeMap::new(),
        Some(Value::Object(map)) => number_map(map),
        Some(Value::String(text)) => match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => number_map(map),
            _ => BTreeMap::new(),
        },
        Some(_) => BTreeMap::new(),
    })
}

fn number_map(map: serde_json::Map<String, Value>) -> BTreeMap<String, f64> {
    map.into_iter()
        .filter_map(|(k, v)| v.as_f64().map(|n| (k, n)))
        .collect()
}

/// The monitor's status object (`Monitor.status`), as carried by the `stats` stream and
/// pause/resume replies. Every field is optional on the wire: an older daemon simply lacks some.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct Stats {
    #[serde(default)]
    pub uptime: f64,
    #[serde(default)]
    pub model_status: String,
    #[serde(default)]
    pub effective_interval: f64,
    #[serde(default)]
    pub miss_ratio: f64,
    #[serde(default)]
    pub observed: u64,
    #[serde(default)]
    pub estimated_missed: u64,
    #[serde(default)]
    pub tracked: u64,
    #[serde(default)]
    pub judged: u64,
    #[serde(default)]
    pub cached: u64,
    #[serde(default)]
    pub dropped: u64,
    #[serde(default)]
    pub unavailable: u64,
    #[serde(default)]
    pub queue_depth: u64,
    #[serde(default)]
    pub queue_age: f64,
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub store: String,
    #[serde(default)]
    pub config: String,
    #[serde(default)]
    pub running: bool,
    #[serde(default)]
    pub allow_rules: u64,
    #[serde(default)]
    pub allow_skipped: u64,
    /// "ebpf", "proc" or "replay" (empty from an older daemon).
    #[serde(default)]
    pub exec_source: String,
    /// Non-empty while the ebpf source has no helper: shown in the header.
    #[serde(default)]
    pub exec_error: String,
    #[serde(default)]
    pub exec_notes: String,
    #[serde(default)]
    pub exec_per_s: f64,
    #[serde(default)]
    pub exec_lost: u64,
}

impl Stats {
    pub fn is_ebpf(&self) -> bool {
        self.exec_source == "ebpf"
    }
}

/// One decoded line from the daemon.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// `{"ok":true,"result":{...}}` or `{"ok":false,"error":"..."}`.
    Response {
        id: Option<u64>,
        ok: bool,
        result: Value,
        error: Option<String>,
    },
    /// `{"stream":"ready"}`: the subscription is live.
    Ready,
    /// `{"stream":"event","event":{...}}`.
    Event(Box<EventPayload>),
    /// `{"stream":"stats","stats":{...}}`.
    Stats(Box<Stats>),
    /// A stream kind we do not know; ignored so a newer daemon cannot break this client.
    UnknownStream,
}

/// Parse one reply/stream line (without the trailing newline).
pub fn parse_line(line: &[u8]) -> Result<Message, ProtocolError> {
    let value: Value =
        serde_json::from_slice(line).map_err(|e| ProtocolError(format!("invalid reply: {e}")))?;
    let obj = value
        .as_object()
        .ok_or_else(|| ProtocolError("invalid reply: not an object".to_string()))?;
    if let Some(version) = obj.get("v") {
        let version = version
            .as_u64()
            .ok_or_else(|| ProtocolError("invalid reply: bad version".to_string()))?;
        if version != u64::from(PROTOCOL_VERSION) {
            return Err(ProtocolError(format!(
                "unsupported protocol version {version}"
            )));
        }
    }
    if let Some(kind) = obj.get("stream").and_then(Value::as_str) {
        return match kind {
            "ready" => Ok(Message::Ready),
            "event" => {
                let event = obj
                    .get("event")
                    .cloned()
                    .ok_or_else(|| ProtocolError("stream event without payload".to_string()))?;
                let event: EventPayload = serde_json::from_value(event)
                    .map_err(|e| ProtocolError(format!("bad event payload: {e}")))?;
                Ok(Message::Event(Box::new(event)))
            }
            "stats" => {
                let stats = obj
                    .get("stats")
                    .cloned()
                    .ok_or_else(|| ProtocolError("stream stats without payload".to_string()))?;
                let stats: Stats = serde_json::from_value(stats)
                    .map_err(|e| ProtocolError(format!("bad stats payload: {e}")))?;
                Ok(Message::Stats(Box::new(stats)))
            }
            _ => Ok(Message::UnknownStream),
        };
    }
    match obj.get("ok") {
        Some(ok) => {
            let ok = ok
                .as_bool()
                .ok_or_else(|| ProtocolError("invalid reply: bad ok flag".to_string()))?;
            Ok(Message::Response {
                id: obj.get("id").and_then(Value::as_u64),
                ok,
                result: obj.get("result").cloned().unwrap_or(Value::Null),
                error: obj.get("error").and_then(Value::as_str).map(str::to_string),
            })
        }
        None => Err(ProtocolError(
            "invalid reply: neither ok nor stream".to_string(),
        )),
    }
}

/// Outgoing requests. The server rejects unknown fields, so these are exact.
pub fn status_request(id: u64) -> Value {
    request(id, "status", json!({}))
}

pub fn events_request(
    id: u64,
    since_id: u64,
    min_severity: i64,
    limit: u32,
    unlabeled: bool,
    text: Option<&str>,
) -> Value {
    let mut params = json!({
        "since_id": since_id,
        "min_severity": min_severity.clamp(0, 2),
        "limit": limit.clamp(1, MAX_LIMIT),
        "unlabeled": unlabeled,
    });
    let text = text.filter(|t| !t.is_empty());
    if let Some(text) = text {
        params["text"] = Value::String(crate::sanitize::clean(text, MAX_TEXT));
    }
    request(id, "events", params)
}

pub fn label_request(id: u64, event_id: u64, label: &str) -> Value {
    request(
        id,
        "label",
        json!({"event_id": event_id, "label": crate::sanitize::clean(label, MAX_LABEL)}),
    )
}

pub fn ack_request(id: u64, event_id: u64) -> Value {
    request(id, "ack", json!({"event_id": event_id}))
}

pub fn allow_add_request(id: u64, rule: &str) -> Value {
    request(id, "allow_add", json!({"rule": rule}))
}

pub fn pause_request(id: u64) -> Value {
    request(id, "pause", json!({}))
}

pub fn resume_request(id: u64) -> Value {
    request(id, "resume", json!({}))
}

pub fn subscribe_request(since_id: Option<u64>) -> Value {
    let mut obj = json!({"v": PROTOCOL_VERSION, "type": "subscribe"});
    if let Some(since_id) = since_id.filter(|&id| id > 0) {
        obj["since_id"] = json!(since_id);
    }
    obj
}

fn request(id: u64, kind: &str, params: Value) -> Value {
    let mut obj = params;
    obj["v"] = json!(PROTOCOL_VERSION);
    obj["id"] = json!(id);
    obj["type"] = json!(kind);
    obj
}

/// One request as a newline-terminated JSON line (ASCII-safe escaping is serde's default).
pub fn encode_line(mut request: Value) -> Vec<u8> {
    if !request.is_object() {
        request = json!({});
    }
    let mut line = serde_json::to_vec(&request).unwrap_or_else(|_| b"{}".to_vec());
    line.push(b'\n');
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_request_has_only_server_accepted_fields() {
        let req = label_request(7, 42, "true-positive");
        let obj = req.as_object().unwrap();
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["event_id", "id", "label", "type", "v"]);
        assert_eq!(obj["type"], "label");
        assert_eq!(obj["id"], 7);
        assert_eq!(obj["event_id"], 42);
        assert_eq!(obj["v"], 5);
    }

    #[test]
    fn allow_add_request_has_only_server_accepted_fields() {
        let req = allow_add_request(3, "/usr/bin/git status *");
        let obj = req.as_object().unwrap();
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["id", "rule", "type", "v"]);
        assert_eq!(obj["type"], "allow_add");
    }

    #[test]
    fn events_request_omits_empty_text_and_clamps() {
        let no_text = events_request(1, 0, 0, 50, false, None)
            .as_object()
            .unwrap()
            .clone();
        assert!(!no_text.contains_key("text"));
        let with_text = events_request(1, 3, 5, 500, true, Some("curl"));
        assert_eq!(with_text["text"], "curl");
        assert_eq!(with_text["min_severity"], 2);
        assert_eq!(with_text["limit"], 100);
        assert_eq!(with_text["unlabeled"], true);
    }

    #[test]
    fn subscribe_omits_since_id_when_absent_or_zero() {
        assert!(
            !subscribe_request(None)
                .as_object()
                .unwrap()
                .contains_key("since_id")
        );
        assert!(
            !subscribe_request(Some(0))
                .as_object()
                .unwrap()
                .contains_key("since_id")
        );
        assert_eq!(subscribe_request(Some(9))["since_id"], 9);
    }

    #[test]
    fn parses_ok_and_error_responses() {
        let ok = parse_line(br#"{"v":5,"id":7,"ok":true,"result":{"x":1}}"#).unwrap();
        match ok {
            Message::Response { id, ok, result, .. } => {
                assert_eq!(id, Some(7));
                assert!(ok);
                assert_eq!(result["x"], 1);
            }
            other => panic!("wrong message: {other:?}"),
        }
        let err = parse_line(br#"{"v":5,"id":7,"ok":false,"error":"nope"}"#).unwrap();
        match err {
            Message::Response { ok, error, .. } => {
                assert!(!ok);
                assert_eq!(error.as_deref(), Some("nope"));
            }
            other => panic!("wrong message: {other:?}"),
        }
    }

    const EVENT_LINE: &[u8] = br#"{"v":5,"stream":"event","event":{"id":3,"time":1.5,"source":"exec","subject":"7","kind":"verdict","severity":2,"message":"m","context":"c","verdict":"model: crit 0.90","class":"crit","confidence":0.9,"probabilities":{"crit":0.9,"safe":0.1},"judged":true,"cached":false,"available":true,"label":null,"labels":[],"acked":false,"future_field":"ignored"}}"#;

    #[test]
    fn parses_stream_event_tolerating_unknown_fields() {
        let msg = parse_line(EVENT_LINE).unwrap();
        let event = match msg {
            Message::Event(e) => e,
            other => panic!("wrong message: {other:?}"),
        };
        assert_eq!(event.id, 3);
        assert_eq!(event.severity, 2);
        assert_eq!(event.probabilities["crit"], 0.9);
        assert!(event.label.is_none());
    }

    #[test]
    fn rejects_wrongly_typed_or_missing_required_fields() {
        assert!(parse_line(br#"{"stream":"event","event":{"time":1.0}}"#).is_err());
        assert!(parse_line(br#"{"stream":"event","event":{"id":"3","time":1.0}}"#).is_err());
        assert!(parse_line(br#"{"stream":"event","event":{"id":3,"time":"now"}}"#).is_err());
        assert!(parse_line(br#"{"stream":"event"}"#).is_err());
        assert!(parse_line(br#"{"stream":"stats","stats":{"paused":"yes"}}"#).is_err());
        assert!(parse_line(b"not json").is_err());
        assert!(parse_line(b"[1,2]").is_err());
        assert!(parse_line(br#"{"ok":"maybe"}"#).is_err());
    }

    #[test]
    fn tolerates_missing_optional_fields_with_defaults() {
        let msg = parse_line(br#"{"v":5,"stream":"event","event":{"id":9,"time":2.0}}"#).unwrap();
        match msg {
            Message::Event(e) => {
                assert_eq!(e.id, 9);
                assert_eq!(e.message, "");
                assert!(!e.acked);
                assert!(e.probabilities.is_empty());
            }
            other => panic!("wrong message: {other:?}"),
        }
    }

    #[test]
    fn parses_stats_and_ready_and_unknown_streams() {
        match parse_line(br#"{"v":5,"stream":"stats","stats":{"uptime":5.0,"paused":true}}"#)
            .unwrap()
        {
            Message::Stats(s) => {
                assert_eq!(s.uptime, 5.0);
                assert!(s.paused);
                assert_eq!(s.queue_depth, 0);
            }
            other => panic!("wrong message: {other:?}"),
        }
        assert_eq!(
            parse_line(br#"{"v":5,"stream":"ready"}"#).unwrap(),
            Message::Ready
        );
        assert_eq!(
            parse_line(br#"{"v":5,"stream":"telemetry"}"#).unwrap(),
            Message::UnknownStream
        );
    }

    #[test]
    fn parses_command_and_class_and_tolerates_their_absence() {
        let line = br#"{"v":5,"stream":"event","event":{"id":1,"time":1.0,"command":"curl -s x","class":"warn"}}"#;
        match parse_line(line).unwrap() {
            Message::Event(e) => {
                assert_eq!(e.command, "curl -s x");
                assert_eq!(e.class, "warn");
            }
            other => panic!("wrong message: {other:?}"),
        }
        match parse_line(br#"{"v":5,"stream":"event","event":{"id":1,"time":1.0}}"#).unwrap() {
            Message::Event(e) => assert_eq!(e.class, ""),
            other => panic!("wrong message: {other:?}"),
        }
    }

    #[test]
    fn rejects_version_mismatch() {
        assert!(parse_line(br#"{"v":1,"ok":true}"#).is_err());
    }

    #[test]
    fn probabilities_may_come_as_a_json_string() {
        let line = br#"{"v":5,"stream":"event","event":{"id":1,"time":1.0,"probabilities":"{\"a\":0.5}"}}"#;
        match parse_line(line).unwrap() {
            Message::Event(e) => assert_eq!(e.probabilities["a"], 0.5),
            other => panic!("wrong message: {other:?}"),
        }
        let junk =
            br#"{"v":5,"stream":"event","event":{"id":1,"time":1.0,"probabilities":"nope"}}"#;
        match parse_line(junk).unwrap() {
            Message::Event(e) => assert!(e.probabilities.is_empty()),
            other => panic!("wrong message: {other:?}"),
        }
    }

    #[test]
    fn encode_line_is_newline_terminated() {
        let line = encode_line(label_request(1, 2, "x"));
        assert_eq!(line.last(), Some(&b'\n'));
        assert!(serde_json::from_slice::<Value>(&line[..line.len() - 1]).is_ok());
    }
}

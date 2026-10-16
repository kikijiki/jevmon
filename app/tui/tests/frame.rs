//! Renders one populated frame into a `TestBackend` and checks its content. The fixture is
//! deterministic (fixed epochs and stats).

use std::time::{Duration, Instant};

use jevmon_tui::protocol::{EventPayload, Stats};
use jevmon_tui::state::App;
use jevmon_tui::theme::Theme;
use jevmon_tui::ui;

fn event(id: u64, severity: i64, class: &str, subject: &str, message: &str) -> EventPayload {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "time": 1_728_000_000.0 + (id as f64) * 37.0,
        "source": "exec",
        "subject": subject,
        "kind": "verdict",
        "severity": severity,
        "message": message,
        "context": "process launch (tier 0 /proc poller)\n\
                    exe: /usr/bin/curl  basename: curl\n\
                    parents: bash -> systemd\n\
                    user(uid): 1000  cgroup: user.slice  cwd: /home/dev\n\
                    argv(template): curl http://10.0.0.5:8080/diag --max-time 3\n\
                    struct: writable_dir=True nonroot_owned=1 recent_write=1 mount=tmpfs\n\
                    env: hour=13 fork_rate=2.10 miss_ratio=0.13",
        "verdict": "model: crit 0.90",
        "class": class,
        "confidence": 0.9,
        "probabilities": {"crit": 0.90, "safe": 0.08, "warn": 0.02},
        "judged": true,
        "cached": false,
        "available": true,
        "label": (id == 46).then_some("true-positive"),
        "labels": if id == 46 { vec!["true-positive", "ack"] } else { Vec::<&str>::new() },
        "acked": id == 46,
    }))
    .expect("fixture payload")
}

#[test]
fn writes_a_populated_frame_to_docs() {
    let mut app = App::new("/run/user/1000/jevmon/jevmond.sock".to_string());
    app.theme = Theme::color();
    app.ingest_all(vec![
        event(41, 0, "safe", "1821", "judge: safe shell activity in bash"),
        event(
            42,
            1,
            "warn",
            "2933",
            "judge: unusual parent for sshd-session",
        ),
        event(
            43,
            2,
            "crit",
            "3041",
            "judge: curl to an internal address with a fresh binary",
        ),
        event(44, 0, "safe", "1177", "judge: safe package manager run"),
        event(
            45,
            2,
            "crit",
            "3099",
            "argv [bold red]injected[/] \u{1b}[31mansi\u{1b}[0m shown literally",
        ),
        event(
            46,
            1,
            "warn",
            "3120",
            "judge: unseen template, scoring rarity",
        ),
        event(47, 0, "safe", "1402", "judge: periodic systemd timer"),
        event(
            48,
            2,
            "crit",
            "3155",
            "judge: exec in a tmpfs mount with recent writes",
        ),
    ]);
    app.handle_msg(jevmon_tui::client::DaemonMsg::Connected);
    // A little sparkline history: a steady 1.4 judged/s (7 per 5s sample).
    let start = Instant::now();
    let mut judged = 400;
    for step in 0..45 {
        judged += 7;
        app.metrics.on_stats_at(
            Stats {
                uptime: 3_725.0 + step as f64 * 5.0,
                model_status: "ready".to_string(),
                effective_interval: 0.05 + if step % 7 == 0 { 0.02 } else { 0.0 },
                miss_ratio: 0.13,
                judged,
                queue_depth: (step % 5),
                queue_age: 200.0 + (step % 10) as f64,
                ..Stats::default()
            },
            start + Duration::from_secs(step * 5),
        );
    }
    // The status the header shows (set after the history so the rate stays the last delta).
    app.metrics.stats = Some(Stats {
        uptime: 3_725.0,
        model_status: "ready".to_string(),
        effective_interval: 0.05,
        miss_ratio: 0.13,
        judged: 400,
        cached: 120,
        dropped: 2,
        queue_depth: 4,
        queue_age: 200.0,
        paused: false,
        ..Stats::default()
    });
    // Look at event 46 (a labelled/interesting row is not required, but the detail must be full).
    app.list.select_id(46);

    let frame = ui::render_to_string(&mut app, 120, 44);

    // The frame must actually be the interface we claim to ship.
    assert!(frame.contains("jevmon"));
    assert!(frame.contains("● connected"));
    assert!(frame.contains("model ready"));
    assert!(frame.contains("judged/s"));
    assert!(frame.contains("1.4/s"), "judged/s rate: {frame}");
    assert!(frame.contains("3m 2"), "queue age shown");
    assert!(frame.contains("probabilities"));
    assert!(frame.contains("detail #46"));
    assert!(frame.contains("TP ✓"));
    assert!(
        !frame.contains('\u{1b}'),
        "no escapes in a TestBackend frame"
    );
    assert_eq!(frame.lines().count(), 44);
}

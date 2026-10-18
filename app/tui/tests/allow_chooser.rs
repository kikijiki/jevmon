//! The allow-rule chooser driven through the real input path (`App::handle_key`) and rendered into
//! a `TestBackend`: opening, the three scopes, adjusting the prefix, the live preview and the
//! command that Enter produces.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use jevmon_tui::client::DaemonMsg;
use jevmon_tui::protocol::EventPayload;
use jevmon_tui::state::{App, Cmd, LayoutMode};
use jevmon_tui::theme::Theme;
use jevmon_tui::ui;

const GIT: &str = "/nix/store/abc-git-2.55.0/bin/git";
const CLANG: &str = "/nix/store/abc-clang-21/bin/clang";

fn payload(id: u64, exe: &str, argv: &str) -> EventPayload {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "time": 1_728_000_000.0 + f64::from(id as u32),
        "source": "exec",
        "subject": id.to_string(),
        "command": argv,
        "exe": exe,
        "argv": argv,
        "kind": "verdict",
        "severity": 0,
        "message": "judge: safe",
        "context": "ctx",
        "verdict": "model: safe 0.99",
        "class": "safe",
        "confidence": 0.99,
        "probabilities": {"safe": 0.99},
        "judged": true,
        "label": null,
        "labels": [],
        "acked": false,
    }))
    .expect("fixture payload")
}

fn app_with(exe: &str, argv: &str) -> App {
    let mut app = App::new("test.sock".to_string());
    app.theme = Theme::monochrome();
    app.ingest(payload(1, exe, argv));
    app
}

fn press(app: &mut App, code: KeyCode) -> Vec<Cmd> {
    app.handle_key(KeyEvent::new(code, KeyModifiers::NONE))
}

fn screen(app: &mut App) -> String {
    ui::render_to_string(app, 130, 40)
}

#[test]
fn a_opens_a_chooser_with_the_subcommand_prefix_selected_and_previews_it() {
    let mut app = app_with(GIT, &format!("{GIT} -C /w/repo diff --name-only HEAD"));
    assert!(press(&mut app, KeyCode::Char('a')).is_empty());
    assert!(app.chooser.is_some());
    let text = screen(&mut app);
    assert!(text.contains("allow launches like #1"), "{text}");
    assert!(text.contains("> prefix k=1"), "{text}");
    assert!(text.contains(&format!("{GIT} -C *")), "{text}");
    assert!(text.contains(&format!("{GIT} -C /w/repo diff --name-only HEAD")));
    assert!(text.contains(&format!("{GIT} *")));
    assert!(text.contains("rule to add:"));
}

#[test]
fn enter_sends_the_previewed_rule_and_esc_cancels() {
    let mut app = app_with(GIT, &format!("{GIT} -C /w/repo diff --name-only HEAD"));
    press(&mut app, KeyCode::Char('a'));
    assert!(press(&mut app, KeyCode::Esc).is_empty());
    assert!(app.chooser.is_none());

    press(&mut app, KeyCode::Char('a'));
    press(&mut app, KeyCode::Right);
    press(&mut app, KeyCode::Char('+'));
    assert!(screen(&mut app).contains(&format!("{GIT} -C /w/repo diff *")));
    let cmds = press(&mut app, KeyCode::Enter);
    assert_eq!(
        cmds,
        vec![Cmd::AllowAdd {
            rule: format!("{GIT} -C /w/repo diff *")
        }]
    );
    assert!(app.chooser.is_none());
}

#[test]
fn scopes_cycle_and_k_stays_within_bounds() {
    let mut app = app_with(CLANG, &format!("{CLANG} -c src/a.c -o build/a.o"));
    press(&mut app, KeyCode::Char('a'));
    for _ in 0..10 {
        press(&mut app, KeyCode::Left);
    }
    assert_eq!(
        app.chooser.as_ref().unwrap().rule(),
        format!("{CLANG} -c *")
    );
    for _ in 0..10 {
        press(&mut app, KeyCode::Char('+'));
    }
    assert_eq!(
        app.chooser.as_ref().unwrap().rule(),
        format!("{CLANG} -c src/a.c -o build/a.o *")
    );
    press(&mut app, KeyCode::Down);
    assert_eq!(app.chooser.as_ref().unwrap().rule(), format!("{CLANG} *"));
    press(&mut app, KeyCode::Down);
    assert_eq!(
        app.chooser.as_ref().unwrap().rule(),
        format!("{CLANG} -c src/a.c -o build/a.o")
    );
    let cmds = press(&mut app, KeyCode::Enter);
    assert_eq!(cmds.len(), 1);
}

#[test]
fn a_launch_without_arguments_offers_exact_and_exe_only() {
    let mut app = app_with("/usr/bin/true", "/usr/bin/true");
    press(&mut app, KeyCode::Char('a'));
    let text = screen(&mut app);
    assert!(!text.contains("prefix"), "{text}");
    assert!(text.contains("> exe only"), "{text}");
    assert_eq!(
        press(&mut app, KeyCode::Enter),
        vec![Cmd::AllowAdd {
            rule: "/usr/bin/true *".to_string()
        }]
    );
}

#[test]
fn the_chooser_is_modal_and_needs_the_detail_pane_and_an_event() {
    let mut app = app_with(GIT, &format!("{GIT} status"));
    press(&mut app, KeyCode::Char('a'));
    // other keys do nothing while it is open
    assert!(press(&mut app, KeyCode::Char('t')).is_empty());
    assert!(press(&mut app, KeyCode::Char('q')).is_empty());
    assert!(!app.quit && app.chooser.is_none(), "q cancels the chooser");

    app.layout = LayoutMode::Hidden;
    press(&mut app, KeyCode::Char('a'));
    assert!(app.chooser.is_none());
    assert!(app.flash_text().is_some_and(|t| t.contains("detail pane")));

    let mut empty = App::new("test.sock".to_string());
    press(&mut empty, KeyCode::Char('a'));
    assert!(empty.chooser.is_none());
    assert!(empty.flash_text().is_some_and(|t| t.contains("no event")));

    let mut bare = app_with("", "");
    press(&mut bare, KeyCode::Char('a'));
    assert!(bare.chooser.is_none() && bare.flash_text().is_some());
}

#[test]
fn replies_show_in_the_status_line() {
    let mut app = app_with(GIT, &format!("{GIT} status"));
    app.handle_msg(DaemonMsg::AllowAdded {
        rule: format!("{GIT} status *"),
        count: 3,
        changed: true,
    });
    assert!(app.flash_text().is_some_and(|t| t.contains("(3 rules)")));
    app.handle_msg(DaemonMsg::AllowAdded {
        rule: "x".to_string(),
        count: 3,
        changed: false,
    });
    assert!(
        app.flash_text()
            .is_some_and(|t| t.contains("already allowed"))
    );
    app.handle_msg(DaemonMsg::Failed("RuleError: bad".to_string()));
    assert!(
        app.flash_text()
            .is_some_and(|t| t.contains("RuleError: bad"))
    );
    let text = screen(&mut app);
    assert!(text.contains("RuleError: bad"), "{text}");
}

#[test]
fn the_footer_and_help_list_the_key() {
    let mut app = app_with(GIT, &format!("{GIT} status"));
    assert!(screen(&mut app).contains("a allow"));
    app.help = true;
    assert!(screen(&mut app).contains("skip future launches"));
}

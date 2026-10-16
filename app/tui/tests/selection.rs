//! Integration tests for the implicit-tail / pinned-selection model, driven through the REAL
//! input path — `App::handle_key` for keys and `App::ingest`/`handle_msg` for the daemon stream
//! — then rendered into a ratatui `TestBackend` so the highlighted row and the offset are the
//! ones a terminal would actually show.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::TestBackend;

use jevmon_tui::protocol::EventPayload;
use jevmon_tui::state::{App, Focus};
use jevmon_tui::theme::Theme;
use jevmon_tui::ui;

fn payload(id: u64, message: &str) -> EventPayload {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "time": 1_728_000_000.0 + f64::from(id as u32),
        "source": "exec",
        "subject": id.to_string(),
        "command": format!("cmd-{id}"),
        "kind": "verdict",
        "severity": 2,
        "message": message,
        "context": "ctx",
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
    }))
    .expect("fixture payload")
}

fn app() -> App {
    let mut app = App::new("test.sock".to_string());
    app.theme = Theme::color();
    app
}

fn press(app: &mut App, code: KeyCode) {
    app.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
}

/// Render once so the list learns its viewport height from the frame (as `ui::draw` does).
fn render(app: &mut App, width: u16, height: u16) -> String {
    ui::render_to_string(app, width, height)
}

/// The rows of the frame that carry the selection background, as plain text.
fn highlighted_rows(app: &mut App, width: u16, height: u16) -> Vec<String> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal.draw(|frame| ui::draw(frame, app)).expect("draw");
    let buffer = terminal.backend().buffer();
    let mut rows = Vec::new();
    for y in 0..buffer.area.height {
        let marked = (0..buffer.area.width).any(|x| {
            buffer
                .cell((x, y))
                .is_some_and(|cell| cell.bg == jevmon_tui::theme::SELECTED_BG)
        });
        if marked {
            let mut text = String::new();
            for x in 0..buffer.area.width {
                text.push_str(buffer.cell((x, y)).map_or(" ", |c| c.symbol()));
            }
            rows.push(text);
        }
    }
    rows
}

/// The terminal row the event with this subject id is rendered on — read from the drawn frame,
/// so clicks are driven by where a terminal would really put the row, not by app internals.
/// The severity word keeps the list title (" events N ") out of the match.
fn row_y(app: &mut App, width: u16, height: u16, subject: u64) -> u16 {
    let text = render(app, width, height);
    text.lines()
        .position(|line| line.contains("crit") && line.contains(&format!(" cmd-{subject} ")))
        .expect("row is on screen") as u16
}

fn tail_in_window(app: &App) -> bool {
    let Some(tail) = app.list.rows().last().copied() else {
        return false;
    };
    app.list
        .rows()
        .get(app.list.offset()..app.list.offset() + app.list.viewport())
        .is_some_and(|window| window.contains(&tail))
}

fn selected_in_window(app: &App) -> bool {
    let Some(sel) = app.list.selected_id() else {
        return false;
    };
    app.list
        .rows()
        .get(app.list.offset()..app.list.offset() + app.list.viewport())
        .is_some_and(|window| window.contains(&sel))
}

/// (a) No key presses at all: the view must autoscroll — the window always shows the newest
/// row and the offset keeps advancing, no matter how long the stream runs.
#[test]
fn autoscroll_follows_the_tail_without_any_keys() {
    let mut app = app();
    app.ingest_all((5_000..5_400u64).map(|id| payload(id, "initial")).collect());
    render(&mut app, 120, 40);
    assert!(app.list.following());
    assert_eq!(app.list.pinned_id(), None, "no pin before any navigation");
    let mut last_offset = app.list.offset();

    for i in 0..1_000u64 {
        app.ingest(payload(1_400_000 + i, "streamed"));
        if i % 100 == 0 {
            render(&mut app, 120, 40);
            assert!(
                tail_in_window(&app),
                "the newest row must stay in the window (i={i}, offset {})",
                app.list.offset()
            );
            if app.list.len() < jevmon_tui::state::MAX_CACHE {
                assert!(
                    app.list.offset() > last_offset,
                    "the offset must keep advancing (i={i}: {} -> {})",
                    last_offset,
                    app.list.offset()
                );
                last_offset = app.list.offset();
            }
        }
    }
    assert!(app.list.following());
    assert_eq!(app.list.pinned_id(), None);
    let text = render(&mut app, 120, 40);
    assert!(
        text.contains("sel tail"),
        "implicit tail in the title: {text:?}"
    );
    assert_eq!(
        app.selected_event().map(|e| e.id),
        Some(1_400_999),
        "the detail pane shows the newest event"
    );
}

/// (b) Down pins the newest row; a long stream then moves neither the view nor the selection,
/// and the pinned row stays rendered.
#[test]
fn down_pins_and_the_stream_never_moves_it() {
    let mut app = app();
    app.ingest_all((1..=200u64).map(|id| payload(id, "initial")).collect());
    render(&mut app, 120, 40);

    press(&mut app, KeyCode::Down); // pins the implicit tail
    assert_eq!(app.list.pinned_id(), Some(200));
    assert_eq!(app.list.selected_id(), Some(200));
    assert!(!app.list.following());
    let offset = app.list.offset();

    for i in 0..200u64 {
        app.ingest(payload(10_000 + i, "streamed"));
    }
    assert_eq!(
        app.list.selected_id(),
        Some(200),
        "the pin survives the stream"
    );
    assert_eq!(app.list.offset(), offset, "the view is parked");
    render(&mut app, 120, 40);
    assert!(selected_in_window(&app));
    let highlighted = highlighted_rows(&mut app, 120, 40);
    assert_eq!(
        highlighted.len(),
        1,
        "exactly one marked row: {highlighted:?}"
    );
    assert!(
        highlighted[0].contains(" cmd-200 "),
        "the pinned row is rendered with the selection background: {highlighted:?}"
    );
    let text = render(&mut app, 120, 40);
    assert!(
        text.contains("sel #200"),
        "pinned id in the title: {text:?}"
    );
}

/// (c) G / End / wheel-to-the-bottom clear the pin, resume following, and the tail stays in
/// view as the stream continues.
#[test]
fn end_and_wheel_resume_following_and_clear_the_pin() {
    let mut app = app();
    app.ingest_all((1..=100u64).map(|id| payload(id, "initial")).collect());
    render(&mut app, 120, 40);

    press(&mut app, KeyCode::Down); // pin the tail
    assert_eq!(app.list.pinned_id(), Some(100));
    press(&mut app, KeyCode::Char('G')); // G -> back to the implicit tail
    assert!(app.list.following());
    assert_eq!(app.list.pinned_id(), None);
    assert_eq!(app.list.selected_id(), Some(100));
    let mut offset = app.list.offset();
    for i in 0..50u64 {
        app.ingest(payload(20_000 + i, "streamed"));
    }
    assert!(tail_in_window(&app), "following keeps the tail in view");
    assert!(app.list.offset() > offset, "the offset advanced again");
    assert_eq!(app.list.selected_id(), Some(20_049));

    // Park with the wheel, then wheel back to the bottom: pin cleared, following resumes.
    app.scroll(-30);
    assert!(!app.list.following());
    assert!(app.list.pinned_id().is_some(), "wheel-up pins");
    app.scroll(10_000);
    assert!(app.list.following());
    assert_eq!(app.list.pinned_id(), None);
    offset = app.list.offset();
    for i in 0..50u64 {
        app.ingest(payload(30_000 + i, "streamed"));
    }
    assert!(tail_in_window(&app));
    assert!(app.list.offset() > offset);

    // End is the same resume.
    press(&mut app, KeyCode::Up);
    assert!(!app.list.following());
    press(&mut app, KeyCode::End);
    assert!(app.list.following());
    assert_eq!(app.list.pinned_id(), None);
}

/// (d) The screenshot's scenario — hundreds of preloaded events, a long stream, NO keys — ends
/// with the tail in view and the newest event in the detail pane.
#[test]
fn screenshot_scenario_keeps_the_tail_in_view_and_newest_in_detail() {
    let mut app = app();
    app.ingest_all((5_000..5_400u64).map(|id| payload(id, "initial")).collect());
    render(&mut app, 120, 40);
    for i in 0..300u64 {
        app.ingest(payload(1_400_000 + i, "streamed"));
    }
    render(&mut app, 120, 40);

    assert!(app.list.following());
    assert_eq!(app.list.pinned_id(), None);
    assert!(tail_in_window(&app), "the newest row is on screen");
    assert_eq!(app.list.selected_id(), Some(1_400_299));
    assert_eq!(app.selected_event().map(|e| e.id), Some(1_400_299));
    let highlighted = highlighted_rows(&mut app, 120, 40);
    assert_eq!(
        highlighted.len(),
        1,
        "the newest row is marked: {highlighted:?}"
    );
    assert!(highlighted[0].contains("1400299"), "{highlighted:?}");
}

/// (e) Down/Up move by exactly one event across interleaved stream batches, and the pin never
/// drifts.
#[test]
fn down_and_up_step_one_event_and_survive_the_stream() {
    let mut app = app();
    app.ingest_all((1..=200u64).map(|id| payload(id, "initial")).collect());
    render(&mut app, 120, 40);

    // Step off the tail, then let a long stream arrive between the key presses.
    press(&mut app, KeyCode::Up);
    for i in 0..100u64 {
        app.ingest(payload(10_000 + i, "streamed"));
    }
    render(&mut app, 120, 40);
    assert!(
        selected_in_window(&app),
        "the pinned selection must stay inside the window (offset {}, len {})",
        app.list.offset(),
        app.list.len()
    );

    let mut index = app.list.selected_index().expect("a row is selected");
    for step in 1..=20u64 {
        press(&mut app, KeyCode::Down);
        let now = app.list.selected_index().expect("still selected");
        assert_eq!(
            now,
            index + 1,
            "Down must advance by exactly one row (step {step})"
        );
        index = now;

        let selected = app.list.selected_id();
        app.ingest(payload(20_000 + step, "streamed"));
        assert_eq!(
            app.list.selected_id(),
            selected,
            "the stream must not move the selection (step {step})"
        );
        assert!(
            selected_in_window(&app),
            "selection must stay in the window after Down + stream (step {step}, offset {}, len {})",
            app.list.offset(),
            app.list.len()
        );
        render(&mut app, 120, 40);

        press(&mut app, KeyCode::Up);
        let now = app.list.selected_index().expect("still selected");
        assert_eq!(
            now,
            index - 1,
            "Up must go back by exactly one row (step {step})"
        );
        index = now;
        app.ingest(payload(30_000 + step, "streamed"));
        assert!(selected_in_window(&app), "after Up + stream (step {step})");
        render(&mut app, 120, 40);
    }
    assert!(
        !highlighted_rows(&mut app, 120, 40).is_empty(),
        "the selected row is drawn with the selection background after every step"
    );
}

/// (f) A filter change keeps the pinned id (at its new index), through the real filter input.
#[test]
fn filter_change_keeps_the_pinned_id() {
    let mut app = app();
    for id in 1..=100u64 {
        let message = if id.is_multiple_of(2) {
            "alpha run"
        } else {
            "beta run"
        };
        app.ingest(payload(id, message));
    }
    render(&mut app, 120, 40);
    press(&mut app, KeyCode::Home);
    press(&mut app, KeyCode::Down); // pin id 2 ("alpha run")
    assert_eq!(app.list.pinned_id(), Some(2));

    press(&mut app, KeyCode::Char('/'));
    for c in "alpha".chars() {
        press(&mut app, KeyCode::Char(c));
    }
    press(&mut app, KeyCode::Enter);
    assert_eq!(app.list.pinned_id(), Some(2), "the pin survives the filter");
    assert_eq!(app.list.selected_id(), Some(2));
    assert_eq!(app.list.selected_index(), Some(0), "at its new index");
    assert_eq!(app.list.len(), 50);
}

/// The pinned id shows in the list title; the implicit tail shows "sel tail"; the marked row
/// carries the selection background and the row marker.
#[test]
fn frame_shows_the_selection_title_and_a_marked_row() {
    let mut app = app();
    app.ingest_all((1..=100u64).map(|id| payload(id, "m")).collect());
    press(&mut app, KeyCode::Home);
    render(&mut app, 120, 40);
    let text = render(&mut app, 120, 40);
    assert!(
        text.contains("sel #1"),
        "the list title must show the pinned id: {text:?}"
    );
    let highlighted = highlighted_rows(&mut app, 120, 40);
    assert_eq!(
        highlighted.len(),
        1,
        "exactly one marked row: {highlighted:?}"
    );
    assert!(
        highlighted[0].contains(" cmd-1 "),
        "the marked row is the selected event: {highlighted:?}"
    );
}

/// (g) A left click on a row selects it through the real input path (`App::click`, the same
/// method main.rs calls for `MouseDown(Left)`): it pins the event, focuses the list, moves the
/// detail pane to it — and neither the pin nor the view moves when the stream continues.
#[test]
fn click_pins_the_row_under_the_pointer_and_the_stream_never_moves_it() {
    let mut app = app();
    app.ingest_all((1..=200u64).map(|id| payload(id, "initial")).collect());
    render(&mut app, 120, 40);
    assert!(app.list.following());

    press(&mut app, KeyCode::Tab); // detail focused: the click must still work the list
    assert_eq!(app.focus, Focus::Detail);

    let target = app.list.rows()[app.list.offset() + 2]; // third visible row, not the tail
    assert_ne!(Some(target), app.list.rows().last().copied());
    let y = row_y(&mut app, 120, 40, target);
    assert!(app.click(2, y), "the row under the pointer is hit");
    assert_eq!(app.list.pinned_id(), Some(target));
    assert_eq!(app.list.selected_id(), Some(target));
    assert!(
        !app.list.following(),
        "clicking pins like keyboard navigation"
    );
    assert_eq!(app.focus, Focus::List, "the click focuses the list");

    render(&mut app, 120, 40);
    let highlighted = highlighted_rows(&mut app, 120, 40);
    assert_eq!(highlighted.len(), 1, "one marked row: {highlighted:?}");
    assert!(
        highlighted[0].contains(&format!(" cmd-{target} ")),
        "the clicked row carries the selection: {highlighted:?}"
    );
    let text = render(&mut app, 120, 40);
    assert!(
        text.contains(&format!("detail #{target}")),
        "the detail pane follows the click: {text:?}"
    );

    let offset = app.list.offset();
    for i in 0..200u64 {
        app.ingest(payload(10_000 + i, "streamed"));
    }
    assert_eq!(
        app.list.selected_id(),
        Some(target),
        "the pin survives the stream"
    );
    assert_eq!(app.list.offset(), offset, "the view is parked");
    assert!(selected_in_window(&app));
}

/// (h) Clicking the newest row has End/G's semantics: it clears the pin and resumes following,
/// whether the list was parked or already riding the tail.
#[test]
fn click_on_the_newest_row_resumes_following() {
    let mut app = app();
    app.ingest_all((1..=100u64).map(|id| payload(id, "initial")).collect());
    render(&mut app, 120, 40);

    press(&mut app, KeyCode::Up); // park on id 99, tail still on screen
    assert_eq!(app.list.pinned_id(), Some(99));
    assert!(!app.list.following());
    let tail = *app.list.rows().last().expect("rows");
    let y = row_y(&mut app, 120, 40, tail);
    assert!(app.click(2, y));
    assert!(
        app.list.following(),
        "clicking the newest row resumes follow"
    );
    assert_eq!(app.list.pinned_id(), None);
    assert_eq!(app.list.selected_id(), Some(tail));

    // Already following: clicking the tail keeps the implicit tail.
    let y = row_y(&mut app, 120, 40, tail);
    assert!(app.click(2, y));
    assert!(app.list.following());
    assert_eq!(app.list.pinned_id(), None);

    // And the stream rides the tail again.
    for i in 0..50u64 {
        app.ingest(payload(10_000 + i, "streamed"));
    }
    assert!(tail_in_window(&app));
    assert_eq!(app.list.selected_id(), Some(10_049));
    assert_eq!(app.list.pinned_id(), None);
}

/// (i) Clicks that are not on a rendered row change nothing: the border column, below the list,
/// and any click while the filter input owns the keyboard.
#[test]
fn clicks_off_a_row_do_nothing() {
    let mut app = app();
    app.ingest_all((1..=100u64).map(|id| payload(id, "initial")).collect());
    render(&mut app, 120, 40);
    let tail = *app.list.rows().last().expect("rows");
    let y = row_y(&mut app, 120, 40, tail);

    assert!(!app.click(0, y), "x=0 is the border column");
    assert!(!app.click(2, y + 50), "far below the list");
    assert!(
        !app.click(2, 39),
        "the footer row (the list ends well above)"
    );
    assert!(app.list.following());
    assert_eq!(app.list.pinned_id(), None);
    assert_eq!(app.list.selected_id(), Some(tail));

    // The filter input takes the keyboard - and the pointer.
    press(&mut app, KeyCode::Char('/'));
    let y = row_y(&mut app, 120, 40, tail); // the bar pushed the list down one row
    assert!(!app.click(2, y), "no clicking while the filter is open");
    press(&mut app, KeyCode::Esc);
    let y = row_y(&mut app, 120, 40, tail);
    assert!(app.click(2, y), "back to normal after closing the filter");
    assert!(app.list.following());
    assert_eq!(app.list.pinned_id(), None);
}

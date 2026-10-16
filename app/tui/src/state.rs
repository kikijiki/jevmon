//! Application state: the event cache, filters, view, connection status and the pure key
//! handling. Nothing here touches a terminal; `ui.rs` renders a `&App` and `main.rs` feeds keys
//! and daemon messages in.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::Rect;

use crate::client::DaemonMsg;
use crate::list::ListState;
use crate::protocol::{EventPayload, MAX_LABEL, MAX_TEXT, Stats};
use crate::sanitize;
use crate::theme::Theme;

/// Events kept in memory (oldest evicted).
pub const MAX_CACHE: usize = 1_000;
pub const SEVERITY_NAMES: [&str; 3] = ["safe", "warn", "crit"];
/// Samples kept per sparkline.
pub const HISTORY: usize = 90;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    List,
    Detail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    Below,
    Beside,
    Hidden,
}

impl LayoutMode {
    pub fn next(self) -> Self {
        match self {
            LayoutMode::Below => LayoutMode::Beside,
            LayoutMode::Beside => LayoutMode::Hidden,
            LayoutMode::Hidden => LayoutMode::Below,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            LayoutMode::Below => "below",
            LayoutMode::Beside => "beside",
            LayoutMode::Hidden => "hidden",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Filters {
    pub min_severity: i64,
    pub unlabeled: bool,
    pub text: String,
}

impl Filters {
    pub fn any_active(&self) -> bool {
        self.min_severity > 0 || self.unlabeled || !self.text.is_empty()
    }
}

/// In-progress editing of the `/` filter line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterInput {
    pub buffer: String,
    /// Cursor position, in characters.
    pub cursor: usize,
    /// The committed filter, restored on Esc.
    pub restore: String,
}

#[derive(Debug, Clone)]
pub struct Connection {
    pub connected: bool,
    pub ever_connected: bool,
    /// Last failure, for the header/detail when down (daemon-supplied, sanitized at render).
    pub reason: String,
    /// When the client will retry (its backoff countdown).
    pub retry_deadline: Option<Instant>,
}

impl Default for Connection {
    fn default() -> Self {
        Self {
            connected: false,
            ever_connected: false,
            reason: "not connected".to_string(),
            retry_deadline: None,
        }
    }
}

impl Connection {
    pub fn retry_remaining(&self) -> Option<Duration> {
        self.retry_deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }
}

/// Sparkline histories fed from the `stats` stream.
#[derive(Debug, Default, Clone)]
pub struct Metrics {
    pub stats: Option<Stats>,
    /// Judged-per-second, computed from the counter deltas between stats messages.
    pub rate: f64,
    pub hist_rate: VecDeque<u64>,
    pub hist_depth: VecDeque<u64>,
    pub hist_age: VecDeque<u64>,
    pub hist_interval: VecDeque<u64>,
    pub hist_miss: VecDeque<u64>,
    last_judged: Option<(u64, Instant)>,
}

impl Metrics {
    pub fn on_stats(&mut self, stats: Stats) {
        self.on_stats_at(stats, Instant::now());
    }

    pub fn on_stats_at(&mut self, stats: Stats, now: Instant) {
        if let Some((previous, then)) = self.last_judged
            && stats.judged >= previous
        {
            let elapsed = now.duration_since(then).as_secs_f64();
            if elapsed > 0.0 {
                self.rate = (stats.judged - previous) as f64 / elapsed;
            }
        }
        self.last_judged = Some((stats.judged, now));
        push_sample(&mut self.hist_rate, (self.rate * 100.0).round() as u64);
        push_sample(&mut self.hist_depth, stats.queue_depth);
        push_sample(&mut self.hist_age, stats.queue_age.max(0.0) as u64);
        push_sample(
            &mut self.hist_interval,
            (stats.effective_interval.max(0.0) * 1_000.0) as u64,
        );
        push_sample(
            &mut self.hist_miss,
            (stats.miss_ratio.clamp(0.0, 1.0) * 1_000.0) as u64,
        );
        self.stats = Some(stats);
    }

    pub fn paused(&self) -> bool {
        self.stats.as_ref().is_some_and(|s| s.paused)
    }
}

fn push_sample(history: &mut VecDeque<u64>, sample: u64) {
    if history.len() >= HISTORY {
        history.pop_front();
    }
    history.push_back(sample);
}

/// A command for the daemon connection task (the client's own type, re-exported so the state
/// module and its tests speak one language).
pub use crate::client::DaemonCmd as Cmd;

/// One navigation step, resolved against the focused pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nav {
    Prev,
    Next,
    PageUp,
    PageDown,
    Home,
    End,
}

pub struct App {
    pub socket: String,
    pub theme: Theme,
    pub events: HashMap<u64, EventPayload>,
    pub order: VecDeque<u64>,
    pub list: ListState,
    pub filters: Filters,
    pub input: Option<FilterInput>,
    pub focus: Focus,
    pub layout: LayoutMode,
    pub help: bool,
    pub detail_offset: usize,
    /// Rows visible in the detail pane (set by the renderer; used for page scrolls).
    pub detail_viewport: usize,
    /// Where the renderer last drew the list's rows (set by `ui::draw`; click hit-testing).
    pub(crate) list_rows: Option<Rect>,
    pub conn: Connection,
    pub metrics: Metrics,
    /// A transient status message, shown at the right of the footer.
    pub flash: Option<(String, Instant)>,
    pub quit: bool,
}

impl App {
    pub fn new(socket: String) -> Self {
        Self {
            socket,
            theme: Theme::from_env(),
            events: HashMap::new(),
            order: VecDeque::new(),
            list: ListState::new(),
            filters: Filters::default(),
            input: None,
            focus: Focus::List,
            layout: LayoutMode::Below,
            help: false,
            detail_offset: 0,
            detail_viewport: 10,
            list_rows: None,
            conn: Connection::default(),
            metrics: Metrics::default(),
            flash: None,
            quit: false,
        }
    }

    pub fn flash(&mut self, message: impl Into<String>) {
        self.flash = Some((message.into(), Instant::now()));
    }

    /// The flash text if it is younger than four seconds.
    pub fn flash_text(&self) -> Option<&str> {
        let (text, at) = self.flash.as_ref()?;
        (at.elapsed() < Duration::from_secs(4)).then_some(text.as_str())
    }

    pub fn total_events(&self) -> usize {
        self.order.len()
    }

    pub fn selected_event(&self) -> Option<&EventPayload> {
        self.list.selected_id().and_then(|id| self.events.get(&id))
    }

    // -- daemon messages ------------------------------------------------------------------

    pub fn handle_msg(&mut self, msg: DaemonMsg) {
        match msg {
            DaemonMsg::Connected => {
                self.conn.connected = true;
                self.conn.ever_connected = true;
                self.conn.reason.clear();
                self.conn.retry_deadline = None;
            }
            DaemonMsg::Initial(events) => self.ingest_all(events),
            DaemonMsg::Event(event) => self.ingest(event),
            DaemonMsg::Stats(stats) => self.metrics.on_stats(stats),
            DaemonMsg::Labeled(event) => self.ingest(event),
            DaemonMsg::Failed(reason) => {
                self.flash(format!("daemon: {}", sanitize::clean(&reason, 200)))
            }
            DaemonMsg::Disconnected { reason, retry_in } => {
                self.conn.connected = false;
                self.conn.reason = sanitize::clean(&reason, 200);
                self.conn.retry_deadline = Some(Instant::now() + Duration::from_secs(retry_in));
            }
        }
    }

    // -- event cache ----------------------------------------------------------------------

    /// Ingest one payload from the daemon: sanitize every string it carries before anything else
    /// (the terminal only ever sees cleaned text), then refresh the view.
    pub fn ingest(&mut self, mut payload: EventPayload) {
        sanitize_event(&mut payload);
        let id = payload.id;
        let is_new = !self.events.contains_key(&id);
        self.events.insert(id, payload);
        if is_new {
            self.order.push_back(id);
            self.prune();
        }
        self.rebuild();
        if self.list.selected_id().is_none() && !self.list.is_empty() {
            self.list.select_end();
            self.detail_offset = 0;
        }
    }

    pub fn ingest_all(&mut self, events: Vec<EventPayload>) {
        for mut payload in events {
            sanitize_event(&mut payload);
            let id = payload.id;
            if !self.events.contains_key(&id) {
                self.order.push_back(id);
            }
            self.events.insert(id, payload);
        }
        self.prune();
        self.rebuild();
        if self.list.selected_id().is_none() && !self.list.is_empty() {
            self.list.select_end();
            self.detail_offset = 0;
        }
    }

    fn prune(&mut self) {
        while self.order.len() > MAX_CACHE {
            let old = self.order.pop_front();
            if let Some(old) = old {
                self.events.remove(&old);
            }
        }
    }

    pub fn passes(&self, event: &EventPayload) -> bool {
        if event.severity < self.filters.min_severity {
            return false;
        }
        if self.filters.unlabeled && event.label.is_some() {
            return false;
        }
        if !self.filters.text.is_empty() {
            let needle = self.filters.text.to_lowercase();
            let haystack = format!(
                "{} {} {} {} {}",
                event.message, event.command, event.subject, event.context, event.class
            )
            .to_lowercase();
            if !haystack.contains(&needle) {
                return false;
            }
        }
        true
    }

    pub fn rebuild(&mut self) {
        let rows: Vec<u64> = self
            .order
            .iter()
            .copied()
            .filter(|id| self.events.get(id).is_some_and(|e| self.passes(e)))
            .collect();
        self.list.set_rows(rows);
    }

    // -- filters --------------------------------------------------------------------------

    pub fn cycle_severity(&mut self) {
        self.filters.min_severity = (self.filters.min_severity + 1) % 3;
        self.rebuild();
    }

    pub fn toggle_unlabeled(&mut self) {
        self.filters.unlabeled = !self.filters.unlabeled;
        self.rebuild();
    }

    pub fn open_filter(&mut self) {
        let restore = self.filters.text.clone();
        let buffer = restore.clone();
        let cursor = buffer.chars().count();
        self.input = Some(FilterInput {
            buffer,
            cursor,
            restore,
        });
    }

    pub fn close_filter(&mut self, cancel: bool) {
        if let Some(input) = self.input.take()
            && cancel
        {
            self.filters.text = input.restore;
            self.rebuild();
        }
    }

    // -- keys -----------------------------------------------------------------------------

    /// Handle one key press; returns the commands to send to the daemon.
    pub fn handle_key(&mut self, key: KeyEvent) -> Vec<Cmd> {
        if matches!(key.kind, KeyEventKind::Release) {
            return Vec::new();
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.quit = true;
            return Vec::new();
        }
        if self.input.is_some() {
            self.handle_filter_key(key);
            return Vec::new();
        }
        if self.help {
            match key.code {
                KeyCode::Char('?') | KeyCode::Esc => self.help = false,
                KeyCode::Char('q') => self.quit = true,
                _ => {}
            }
            return Vec::new();
        }
        match key.code {
            KeyCode::Char('q') => {
                self.quit = true;
                Vec::new()
            }
            KeyCode::Esc => {
                if self.help {
                    self.help = false;
                }
                Vec::new()
            }
            KeyCode::Char('?') => {
                self.help = true;
                Vec::new()
            }
            KeyCode::Char('t') => self.label_selected("true-positive"),
            KeyCode::Char('f') => self.label_selected("false-positive"),
            KeyCode::Char('y') => self.ack_selected(),
            KeyCode::Char('p') => self.toggle_pause(),
            KeyCode::Char('/') => {
                self.open_filter();
                Vec::new()
            }
            KeyCode::Char('u') => {
                self.toggle_unlabeled();
                Vec::new()
            }
            KeyCode::Char('s') => {
                self.cycle_severity();
                Vec::new()
            }
            KeyCode::Char('r') => {
                self.conn.connected = false;
                self.conn.reason = "reconnecting".to_string();
                self.conn.retry_deadline = None;
                vec![Cmd::Reconnect]
            }
            KeyCode::Char('d') => {
                self.layout = self.layout.next();
                self.detail_offset = 0;
                Vec::new()
            }
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::List => Focus::Detail,
                    Focus::Detail => Focus::List,
                };
                Vec::new()
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.navigate(Nav::Prev);
                Vec::new()
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.navigate(Nav::Next);
                Vec::new()
            }
            KeyCode::PageUp => {
                self.navigate(Nav::PageUp);
                Vec::new()
            }
            KeyCode::PageDown => {
                self.navigate(Nav::PageDown);
                Vec::new()
            }
            KeyCode::Home | KeyCode::Char('g') => {
                self.navigate(Nav::Home);
                Vec::new()
            }
            KeyCode::End | KeyCode::Char('G') => {
                self.navigate(Nav::End);
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// List keys move the selection (and scroll just enough to keep it visible); with the detail
    /// focused they scroll the detail pane instead.
    fn navigate(&mut self, nav: Nav) {
        if self.focus == Focus::Detail {
            let page = self.detail_viewport.saturating_sub(1).max(1) as isize;
            match nav {
                Nav::Prev => self.scroll_detail(-1),
                Nav::Next => self.scroll_detail(1),
                Nav::PageUp => self.scroll_detail(-page),
                Nav::PageDown => self.scroll_detail(page),
                Nav::Home => self.detail_offset = 0,
                Nav::End => {} // clamped at render; the pane stays where the user left it
            }
            return;
        }
        let before = self.list.selected_id();
        match nav {
            Nav::Prev => self.list.select_prev(),
            Nav::Next => self.list.select_next(),
            Nav::PageUp => self.list.select_page(false),
            Nav::PageDown => self.list.select_page(true),
            Nav::Home => self.list.select_home(),
            Nav::End => self.list.select_end(),
        }
        if self.list.selected_id() != before {
            self.detail_offset = 0;
        }
    }

    /// Mouse wheel over the focused pane.
    pub fn scroll(&mut self, lines: i32) {
        let delta = lines as isize;
        match self.focus {
            Focus::List => self.list.scroll_by(delta),
            Focus::Detail => self.scroll_detail(-delta),
        }
    }

    /// A left click at terminal cell (x, y): select the list row under the pointer with the
    /// same semantics as keyboard navigation — any row pins the selection and focuses the list,
    /// the newest row resumes following — and reset the detail scroll like any other selection
    /// change. Returns false when the press hit no rendered row (outside the list, an empty
    /// area below the last row, or while the filter input / help overlay takes the keyboard).
    pub fn click(&mut self, x: u16, y: u16) -> bool {
        if self.help || self.input.is_some() {
            return false;
        }
        let Some(rows) = self.list_rows else {
            return false;
        };
        let inside = x >= rows.x
            && x < rows.x.saturating_add(rows.width)
            && y >= rows.y
            && y < rows.y.saturating_add(rows.height);
        if !inside {
            return false;
        }
        let visible = usize::from(y - rows.y);
        let Some(&id) = self.list.rows().get(self.list.offset() + visible) else {
            return false;
        };
        let before = self.list.selected_id();
        self.focus = Focus::List;
        self.list.select_id(id);
        if self.list.selected_id() != before {
            self.detail_offset = 0;
        }
        true
    }

    fn scroll_detail(&mut self, delta: isize) {
        let next = self.detail_offset as isize + delta;
        self.detail_offset = next.max(0) as usize;
    }

    fn handle_filter_key(&mut self, key: KeyEvent) {
        let Some(input) = self.input.as_mut() else {
            return;
        };
        match key.code {
            KeyCode::Esc => {
                self.close_filter(true);
            }
            KeyCode::Enter => self.close_filter(false),
            KeyCode::Backspace if input.cursor > 0 => {
                let at = char_index(&input.buffer, input.cursor - 1);
                input.buffer.remove(at);
                input.cursor -= 1;
                self.sync_filter();
            }
            KeyCode::Delete if input.cursor < input.buffer.chars().count() => {
                let at = char_index(&input.buffer, input.cursor);
                input.buffer.remove(at);
                self.sync_filter();
            }
            KeyCode::Left => input.cursor = input.cursor.saturating_sub(1),
            KeyCode::Right => input.cursor = (input.cursor + 1).min(input.buffer.chars().count()),
            KeyCode::Home => input.cursor = 0,
            KeyCode::End => input.cursor = input.buffer.chars().count(),
            KeyCode::Char(c)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && input.buffer.chars().count() < MAX_TEXT =>
            {
                let at = char_index(&input.buffer, input.cursor);
                input.buffer.insert(at, c);
                input.cursor += 1;
                self.sync_filter();
            }
            _ => {}
        }
    }

    fn sync_filter(&mut self) {
        if let Some(input) = &self.input {
            self.filters.text = input.buffer.clone();
        }
        self.rebuild();
    }

    // -- daemon commands ------------------------------------------------------------------

    fn label_selected(&mut self, label: &str) -> Vec<Cmd> {
        match self.selected_id() {
            Some(event_id) => vec![Cmd::Label {
                event_id,
                label: crate::sanitize::clean(label, MAX_LABEL),
            }],
            None => {
                self.flash("no event selected");
                Vec::new()
            }
        }
    }

    fn ack_selected(&mut self) -> Vec<Cmd> {
        match self.selected_id() {
            Some(event_id) => vec![Cmd::Ack { event_id }],
            None => {
                self.flash("no event selected");
                Vec::new()
            }
        }
    }

    fn toggle_pause(&mut self) -> Vec<Cmd> {
        if self.metrics.paused() {
            vec![Cmd::Resume]
        } else {
            vec![Cmd::Pause]
        }
    }

    fn selected_id(&self) -> Option<u64> {
        self.list.selected_id()
    }
}

fn char_index(text: &str, chars: usize) -> usize {
    text.char_indices()
        .nth(chars)
        .map_or(text.len(), |(i, _)| i)
}

/// Strip escapes/control/format characters from every daemon-supplied string in an event.
/// Length bounds mirror what the daemon promises; the view truncates further for the space it has.
pub fn sanitize_event(event: &mut EventPayload) {
    event.source = sanitize::clean(&event.source, 64);
    event.subject = sanitize::clean(&event.subject, 256);
    event.kind = sanitize::clean(&event.kind, 64);
    event.message = sanitize::clean(&event.message, sanitize::DEFAULT_LIMIT);
    event.command = sanitize::clean(&event.command, sanitize::DEFAULT_LIMIT);
    // The context keeps its line structure: it is a document, not a caption.
    event.context = sanitize::clean_lines(&event.context, sanitize::DEFAULT_LIMIT);
    event.verdict = sanitize::clean(&event.verdict, 256);
    event.class = sanitize::clean(&event.class, MAX_LABEL);
    if let Some(label) = &event.label {
        event.label = Some(sanitize::clean(label, MAX_LABEL));
    }
    for label in &mut event.labels {
        *label = sanitize::clean(label, MAX_LABEL);
    }
    let probs = std::mem::take(&mut event.probabilities);
    event.probabilities = probs
        .into_iter()
        .map(|(k, v)| (sanitize::clean(&k, 64), v))
        .collect();
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    pub fn payload(id: u64, message: &str) -> EventPayload {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "time": 1_728_000_000.0 + f64::from(id as u32),
            "source": "exec",
            "subject": "42",
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
        app.theme = Theme::monochrome();
        app
    }

    fn press(app: &mut App, code: KeyCode) -> Vec<Cmd> {
        app.handle_key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn ingest_sanitizes_every_daemon_string() {
        let mut app = app();
        let mut event = payload(1, "msg \x1b[31mred\x1b[0m \u{202e}evil");
        event.subject = "argv [bold red]x[/]\x1b]0;pwn\x07".to_string();
        event.context = "ctx\u{200b}hidden".to_string();
        event.command = "curl \x1b[2J\u{202e}-s \x1b]0;pwn\x07host".to_string();
        app.ingest(event);
        let event = app.events.get(&1).unwrap();
        assert!(!event.message.contains('\u{1b}'));
        assert!(event.message.contains("msg red"));
        assert!(!event.subject.contains('\u{1b}'));
        assert!(event.subject.contains("[bold red]x[/]"));
        assert!(!event.context.contains('\u{200b}'));
        assert!(!event.command.contains('\u{1b}') && !event.command.contains('\u{202e}'));
        assert!(event.command.starts_with("curl "));
    }

    #[test]
    fn filters_apply_to_new_and_rebuilt_rows() {
        let mut app = app();
        app.ingest(payload(1, "curl suspicious"));
        app.ingest(payload(2, "systemd routine"));
        assert_eq!(app.list.len(), 2);

        app.filters.text = "curl".to_string();
        app.rebuild();
        assert_eq!(app.list.rows(), &[1]);

        app.filters.text.clear();
        app.filters.min_severity = 1;
        app.rebuild();
        assert_eq!(app.list.len(), 2); // both fixtures are severity 2

        app.filters.unlabeled = true;
        let mut labelled = payload(1, "curl suspicious");
        labelled.label = Some("true-positive".to_string());
        app.ingest(labelled);
        app.rebuild();
        assert_eq!(app.list.rows(), &[2]);
    }

    #[test]
    fn selection_survives_filter_changes_and_follows_the_id() {
        let mut app = app();
        for id in 1..=5 {
            app.ingest(payload(id, &format!("event {id}")));
        }
        press(&mut app, KeyCode::Home);
        assert_eq!(app.list.selected_id(), Some(1));
        assert_eq!(app.list.pinned_id(), Some(1)); // explicit navigation pins
        app.filters.text = "event 3".to_string();
        app.rebuild();
        // 1 is filtered out: the selection falls to whatever took its place, not to nowhere.
        assert_eq!(app.list.selected_id(), Some(3));
        app.filters.text.clear();
        app.rebuild();
        assert_eq!(app.list.selected_id(), Some(3)); // still pinned by id
        press(&mut app, KeyCode::End);
        assert_eq!(app.list.selected_id(), Some(5)); // back to the implicit tail
        assert_eq!(app.list.pinned_id(), None);
        assert!(app.list.following());
        app.ingest(payload(6, "event 6"));
        assert_eq!(app.list.selected_id(), Some(6)); // implicit tail tracks the stream
        assert!(app.list.at_bottom());
        press(&mut app, KeyCode::Up);
        assert_eq!(app.list.selected_id(), Some(5)); // pinned
        app.ingest(payload(7, "event 7"));
        assert_eq!(
            app.list.selected_id(),
            Some(5) // a pinned selection never moves with the stream
        );
    }

    #[test]
    fn keys_produce_daemon_commands() {
        let mut app = app();
        app.ingest(payload(7, "hello"));
        app.conn.connected = true;
        let cmds = press(&mut app, KeyCode::Char('t'));
        assert_eq!(
            cmds,
            vec![Cmd::Label {
                event_id: 7,
                label: "true-positive".to_string()
            }]
        );
        let cmds = press(&mut app, KeyCode::Char('f'));
        assert_eq!(
            cmds,
            vec![Cmd::Label {
                event_id: 7,
                label: "false-positive".to_string()
            }]
        );
        let cmds = press(&mut app, KeyCode::Char('y'));
        assert_eq!(cmds, vec![Cmd::Ack { event_id: 7 }]);
        let cmds = press(&mut app, KeyCode::Char('p'));
        assert_eq!(cmds, vec![Cmd::Pause]);
        app.handle_msg(DaemonMsg::Stats(Stats {
            paused: true,
            ..Stats::default()
        }));
        let cmds = press(&mut app, KeyCode::Char('p'));
        assert_eq!(cmds, vec![Cmd::Resume]);
        let cmds = press(&mut app, KeyCode::Char('r'));
        assert_eq!(cmds, vec![Cmd::Reconnect]);
    }

    #[test]
    fn label_keys_without_a_selection_flash_instead_of_commanding() {
        let mut app = app();
        app.conn.connected = true;
        assert!(press(&mut app, KeyCode::Char('t')).is_empty());
        assert!(
            app.flash_text()
                .is_some_and(|f| f.contains("no event selected"))
        );
    }

    #[test]
    fn filter_input_edits_live_and_esc_restores() {
        let mut app = app();
        app.ingest(payload(1, "curl suspicious"));
        app.ingest(payload(2, "systemd routine"));
        press(&mut app, KeyCode::Char('/'));
        assert!(app.input.is_some());
        for c in "curl".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        assert_eq!(app.filters.text, "curl");
        assert_eq!(app.list.rows(), &[1]);
        press(&mut app, KeyCode::Esc);
        assert!(app.input.is_none());
        assert_eq!(app.filters.text, "");
        assert_eq!(app.list.len(), 2);
        // committed with Enter keeps the text
        press(&mut app, KeyCode::Char('/'));
        for c in "systemd".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.filters.text, "systemd");
        assert_eq!(app.list.rows(), &[2]);
        // typing '/' inside the filter is a literal slash, not "open filter"
        press(&mut app, KeyCode::Char('/'));
        press(&mut app, KeyCode::Char('/'));
        assert_eq!(app.input.as_ref().unwrap().buffer, "systemd/");
    }

    #[test]
    fn g_and_g_jump_to_the_first_and_last_row() {
        let mut app = app();
        for id in 1..=5 {
            app.ingest(payload(id, "m"));
        }
        press(&mut app, KeyCode::Char('g'));
        assert_eq!(app.list.selected_id(), Some(1));
        press(&mut app, KeyCode::Char('G'));
        assert_eq!(app.list.selected_id(), Some(5));
    }

    #[test]
    fn reconnect_and_quit_keys() {
        let mut app = app();
        press(&mut app, KeyCode::Char('?'));
        assert!(app.help);
        press(&mut app, KeyCode::Esc);
        assert!(!app.help);
        press(&mut app, KeyCode::Char('q'));
        assert!(app.quit);
    }

    #[test]
    fn metrics_compute_judged_rate_from_counter_deltas() {
        let mut metrics = Metrics::default();
        let start = Instant::now();
        metrics.on_stats_at(
            Stats {
                judged: 10,
                ..Stats::default()
            },
            start,
        );
        assert_eq!(metrics.rate, 0.0);
        metrics.on_stats_at(
            Stats {
                judged: 20,
                queue_depth: 3,
                queue_age: 200.0,
                effective_interval: 0.05,
                miss_ratio: 0.098,
                ..Stats::default()
            },
            start + Duration::from_secs(2),
        );
        assert_eq!(metrics.rate, 5.0);
        assert_eq!(metrics.hist_depth.back(), Some(&3));
        assert_eq!(metrics.hist_age.back(), Some(&200));
        assert_eq!(metrics.hist_interval.back(), Some(&50));
        assert_eq!(metrics.hist_miss.back(), Some(&98));
    }

    #[test]
    fn cache_is_pruned_but_the_selection_id_survives() {
        let mut app = app();
        for id in 1..=u64::try_from(MAX_CACHE + 50).unwrap() {
            app.ingest(payload(id, "m"));
        }
        assert_eq!(app.order.len(), MAX_CACHE);
        assert!(!app.events.contains_key(&1));
        press(&mut app, KeyCode::End);
        assert_eq!(
            app.list.selected_id(),
            Some(u64::try_from(MAX_CACHE + 50).unwrap())
        );
    }

    #[test]
    fn connection_states_carry_a_countdown() {
        let mut app = app();
        app.handle_msg(DaemonMsg::Disconnected {
            reason: "cannot connect to /tmp/sock: \u{1b}[31mNo such file\u{1b}[0m".to_string(),
            retry_in: 2,
        });
        assert!(!app.conn.connected);
        assert!(!app.conn.reason.contains('\u{1b}'));
        assert!(app.conn.retry_remaining().is_some());
    }
}

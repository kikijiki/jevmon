//! Rendering: a dark, cohesive frame built from the pure `App` state. Every string drawn here has
//! already been through `sanitize` (event fields at ingest, the rest at the call site), so
//! attacker text arrives as plain text and can never restyle or forge the UI. With `NO_COLOR`
//! the palette collapses to the terminal default and only emphasis survives.

use std::collections::VecDeque;

use ratatui::backend::TestBackend;
use ratatui::layout::{Alignment, Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Clear, Paragraph, Sparkline};
use ratatui::{Frame, Terminal};
use unicode_width::UnicodeWidthStr;

use crate::allow::Scope;
use crate::format::{
    self, format_count, format_duration, format_interval, format_percent, format_rate,
};
use crate::list::visible_rows;
use crate::protocol::EventPayload;
use crate::sanitize;
use crate::state::{App, Focus, LayoutMode, SEVERITY_NAMES};
use crate::theme::{self, Theme};

/// Footer key hints: key, label.
const KEYS: [(&str, &str); 12] = [
    ("t", "TP"),
    ("f", "FP"),
    ("y", "ack"),
    ("a", "allow"),
    ("p", "pause"),
    ("/", "filter"),
    ("u", "unlab"),
    ("s", "class"),
    ("r", "reconnect"),
    ("d", "layout"),
    ("?", "help"),
    ("q", "quit"),
];

// -- entry point --------------------------------------------------------------------------------

/// Draw one frame.
pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let mut base = Style::default();
    if let Some(bg) = app.theme.bg(theme::BG) {
        base = base.bg(bg);
    }
    frame.render_widget(Block::default().style(base), area);

    if area.width < 30 || area.height < 8 {
        centered_text(
            frame,
            area,
            "terminal too small — resize to at least 30x8",
            app.theme.text(theme::DIM, Modifier::empty()),
        );
        return;
    }

    let filter_bar = usize::from(app.input.is_some() || app.filters.any_active()) as u16;
    let show_metrics = area.height >= 14;
    let mut constraints = vec![Constraint::Length(1)];
    if show_metrics {
        constraints.push(Constraint::Length(3));
    }
    if filter_bar > 0 {
        constraints.push(Constraint::Length(1));
    }
    constraints.push(Constraint::Min(4));
    constraints.push(Constraint::Length(1));
    let chunks = Layout::vertical(constraints).split(area);

    let mut index = 0;
    render_header(frame, chunks[index], app);
    index += 1;
    if show_metrics {
        render_metrics(frame, chunks[index], app);
        index += 1;
    }
    if filter_bar > 0 {
        render_filter_bar(frame, chunks[index], app);
        index += 1;
    }
    render_main(frame, chunks[index], app);
    index += 1;
    render_footer(frame, chunks[index], app);

    if app.help {
        render_help(frame, area, app);
    }
    if app.chooser.is_some() {
        render_chooser(frame, area, app);
    }
}

// -- header -------------------------------------------------------------------------------------

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let mut left: Vec<Span<'static>> = vec![
        Span::styled(
            " jevmon ",
            theme.paint_mod(theme::ACCENT, theme::CHIP_BG, Modifier::BOLD),
        ),
        Span::styled(" ", Style::default()),
    ];

    if app.conn.connected {
        left.push(Span::styled(
            "● connected",
            theme.text(theme::GREEN, Modifier::BOLD),
        ));
    } else {
        let countdown = app
            .conn
            .retry_remaining()
            .map(|d| d.as_secs_f64().ceil() as u64);
        let suffix = match countdown {
            Some(secs) => format!(" · retry in {secs}s"),
            None => String::new(),
        };
        if app.conn.ever_connected {
            left.push(Span::styled(
                format!("↻ reconnecting{suffix}"),
                theme.text(theme::YELLOW, Modifier::BOLD),
            ));
        } else {
            left.push(Span::styled(
                format!("✖ daemon not running{suffix}"),
                theme.text(theme::RED, Modifier::BOLD),
            ));
        }
    }

    if let Some(stats) = &app.metrics.stats {
        let status = sanitize::clean(&stats.model_status, 60);
        let (color, label) = if status.starts_with("ready") {
            (theme::GREEN, "model ready".to_string())
        } else if status.starts_with("loading") {
            (theme::YELLOW, "model loading".to_string())
        } else if status.starts_with("fake") {
            // The daemon's `[judge] mode = "fake"` stand-in: honest about what is (not) running.
            (theme::CYAN, "mock judge".to_string())
        } else {
            (theme::RED, format!("model {}", clip(&status, 34)))
        };
        left.push(Span::styled(
            "  │  ",
            theme.text(theme::DIM, Modifier::empty()),
        ));
        left.push(Span::styled(label, theme.text(color, Modifier::BOLD)));
        if !stats.exec_error.is_empty() {
            left.push(Span::styled(
                "  │  ",
                theme.text(theme::DIM, Modifier::empty()),
            ));
            left.push(Span::styled(
                clip(&sanitize::clean(&stats.exec_error, 160), 90),
                theme.paint_mod(theme::RED, theme::CRIT_BG, Modifier::BOLD),
            ));
        }
        left.push(Span::styled(
            format!("  up {}", format_duration(stats.uptime)),
            theme.text(theme::DIM, Modifier::empty()),
        ));
        if stats.allow_rules > 0 {
            left.push(Span::styled(
                format!(
                    "  allow {} rules · {} skipped",
                    format_count(stats.allow_rules),
                    format_count(stats.allow_skipped)
                ),
                theme.text(theme::DIM, Modifier::empty()),
            ));
        }
        if stats.paused {
            left.push(Span::styled(
                "  PAUSED",
                theme.paint_mod(theme::RED, theme::CRIT_BG, Modifier::BOLD),
            ));
        }
    }

    let right = Span::styled(
        format!(" {}", clip(&sanitize::clean(&app.socket, 200), 56)),
        theme.text(theme::DIM, Modifier::empty()),
    );
    render_left_right(frame, area, Line::from(left), right);
}

fn render_left_right(frame: &mut Frame, area: Rect, left: Line<'static>, right: Span<'static>) {
    let right_width = line_width(&Line::from(vec![right.clone()]));
    if line_width(&left) + right_width + 2 <= usize::from(area.width) {
        let [l, r] =
            Layout::horizontal([Constraint::Min(0), Constraint::Length(right_width as u16)])
                .areas(area);
        frame.render_widget(Paragraph::new(left), l);
        frame.render_widget(
            Paragraph::new(Line::from(right).alignment(Alignment::Right)),
            r,
        );
    } else {
        frame.render_widget(Paragraph::new(left), area);
    }
}

// -- metrics strip ------------------------------------------------------------------------------

const METRIC_LABELS: [&str; 5] = ["judged/s", "queue", "queue age", "interval", "miss"];
const METRIC_LABELS_EBPF: [&str; 5] = ["judged/s", "queue", "queue age", "exec/s", "lost"];
const METRIC_COLORS: [Color; 5] = [
    theme::CYAN,
    theme::MAGENTA,
    theme::YELLOW,
    theme::ACCENT,
    theme::RED,
];

fn render_metrics(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let ebpf = app.metrics.stats.as_ref().is_some_and(|s| s.is_ebpf());
    let labels = if ebpf {
        &METRIC_LABELS_EBPF
    } else {
        &METRIC_LABELS
    };
    let values: [String; 5] = match &app.metrics.stats {
        Some(stats) if ebpf => [
            format_rate(app.metrics.rate),
            format_count(stats.queue_depth),
            format_duration(stats.queue_age),
            format_rate(stats.exec_per_s),
            format_count(stats.exec_lost),
        ],
        Some(stats) => [
            format_rate(app.metrics.rate),
            format_count(stats.queue_depth),
            format_duration(stats.queue_age),
            format_interval(stats.effective_interval),
            format_percent(stats.miss_ratio),
        ],
        None => std::array::from_fn(|_| "–".to_string()),
    };
    let histories: [&VecDeque<u64>; 5] = [
        &app.metrics.hist_rate,
        &app.metrics.hist_depth,
        &app.metrics.hist_age,
        &app.metrics.hist_interval,
        &app.metrics.hist_miss,
    ];

    let cells = Layout::horizontal([Constraint::Ratio(1, 5); 5])
        .spacing(1)
        .split(area);
    for (index, cell) in cells.iter().enumerate() {
        if cell.width < 4 {
            continue;
        }
        let width = usize::from(cell.width);
        let label = labels[index];
        let label = clip(label, (width / 2).max(2));
        let value = clip(
            &values[index],
            width.saturating_sub(label_width(&label) + 1),
        );
        let header = Line::from(vec![
            Span::styled(label, theme.text(theme::DIM, Modifier::empty())),
            Span::styled(" ", Style::default()),
            Span::styled(value, theme.text(METRIC_COLORS[index], Modifier::BOLD)),
        ]);
        frame.render_widget(Paragraph::new(header), Rect { height: 1, ..*cell });
        if cell.height > 1 {
            let history = &histories[index];
            let samples = tail(history, usize::from(cell.width));
            let spark = Sparkline::default()
                .data(samples)
                .style(Style::default().fg(theme.fg(METRIC_COLORS[index])));
            frame.render_widget(
                spark,
                Rect {
                    y: cell.y + 1,
                    height: cell.height - 1,
                    ..*cell
                },
            );
        }
    }
}

/// The last `width` samples of a history (one bar per column, oldest first).
fn tail(history: &VecDeque<u64>, width: usize) -> Vec<u64> {
    let skip = history.len().saturating_sub(width);
    history.iter().skip(skip).copied().collect()
}

// -- filter bar ---------------------------------------------------------------------------------

fn render_filter_bar(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut cursor = None;
    if let Some(input) = &app.input {
        let prefix = " filter▸ ";
        spans.push(Span::styled(
            prefix.to_string(),
            theme.paint_mod(theme::ACCENT, theme::CHIP_BG, Modifier::BOLD),
        ));
        let buffer = sanitize::clean(&input.buffer, crate::protocol::MAX_TEXT);
        let before: String = buffer.chars().take(input.cursor).collect();
        cursor = Some(
            usize::from(area.x)
                + line_width(&Line::from(vec![Span::raw(format!("{prefix}{before}"))])),
        );
        spans.push(Span::styled(
            buffer,
            theme.text(theme::FG, Modifier::UNDERLINED),
        ));
        spans.push(Span::styled(" ", Style::default()));
    } else {
        spans.push(Span::styled(
            " filters ",
            theme.text(theme::DIM, Modifier::BOLD),
        ));
    }

    let mut chips: Vec<Span<'static>> = Vec::new();
    if app.input.is_none() && !app.filters.text.is_empty() {
        chips.push(chip(
            format!(
                " text: {} ",
                clip(&sanitize::clean(&app.filters.text, 40), 30)
            ),
            theme::ACCENT,
            theme,
        ));
    }
    if app.filters.unlabeled {
        chips.push(chip(" unlabeled ".to_string(), theme::MAGENTA, theme));
    }
    if app.filters.min_severity > 0 {
        let name = SEVERITY_NAMES
            .get(app.filters.min_severity as usize)
            .copied()
            .unwrap_or("?");
        let color = if app.filters.min_severity >= 2 {
            theme::RED
        } else {
            theme::YELLOW
        };
        chips.push(chip(format!(" ≥ {name} "), color, theme));
    }
    for (index, chip) in chips.into_iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw(" "));
        }
        spans.push(chip);
    }

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
    if let Some(x) = cursor
        && x < usize::from(area.x) + usize::from(area.width)
    {
        frame.set_cursor_position(Position::new(x as u16, area.y));
    }
}

fn chip(text: String, color: Color, theme: &Theme) -> Span<'static> {
    Span::styled(text, theme.paint_mod(color, theme::CHIP_BG, Modifier::BOLD))
}

// -- main area ----------------------------------------------------------------------------------

fn render_main(frame: &mut Frame, area: Rect, app: &mut App) {
    match app.layout {
        LayoutMode::Hidden => render_events(frame, area, app),
        LayoutMode::Below => {
            let [list, detail] =
                Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)])
                    .areas(area);
            render_events(frame, list, app);
            render_detail(frame, detail, app);
        }
        LayoutMode::Beside => {
            let [list, detail] =
                Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)])
                    .areas(area);
            render_events(frame, list, app);
            render_detail(frame, detail, app);
        }
    }
}

fn render_events(frame: &mut Frame, area: Rect, app: &mut App) {
    let theme = &app.theme;
    let rows = app.list.len();
    let total = app.total_events();
    let mut title: Vec<Span<'static>> = vec![Span::styled(
        if rows == total {
            format!(" events {total} ")
        } else {
            format!(" events {rows}/{total} ")
        },
        theme.text(theme::DIM, Modifier::BOLD),
    )];
    if let Some(id) = app.list.pinned_id() {
        title.push(Span::styled(
            format!(" sel #{id} "),
            theme.text(theme::ACCENT, Modifier::BOLD),
        ));
    } else if app.list.following() && !app.list.is_empty() {
        title.push(Span::styled(
            " sel tail ",
            theme.text(theme::ACCENT, Modifier::BOLD),
        ));
    }
    let border = if app.focus == Focus::List {
        theme::ACCENT
    } else {
        theme::DIM
    };
    let block = Block::bordered()
        .title(Line::from(title))
        .border_style(Style::default().fg(theme.fg(border)));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    app.list_rows = None; // no rows drawn below (empty/short frame): clicks hit nothing
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    if rows == 0 {
        render_empty_events(frame, inner, app);
        return;
    }

    app.list
        .set_viewport(usize::from(inner.height).saturating_sub(1));
    // Click hit-testing geometry: the header line takes the first inner row, events follow.
    app.list_rows = Some(Rect::new(
        inner.x,
        inner.y.saturating_add(1),
        inner.width,
        u16::try_from(app.list.viewport()).unwrap_or(u16::MAX),
    ));
    let columns = columns(inner.width);
    let mut lines = vec![header_line(&columns, theme)];
    for &id in visible_rows(&app.list) {
        let Some(event) = app.events.get(&id) else {
            continue;
        };
        let selected = app.list.selected_id() == Some(id);
        lines.push(event_line(event, &columns, theme, selected));
    }
    frame.render_widget(Paragraph::new(Text::from(lines)), inner);
}

fn render_empty_events(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let text = if !app.conn.connected && app.total_events() == 0 {
        let countdown = app
            .conn
            .retry_remaining()
            .map(|d| format!("retry in {}s", d.as_secs_f64().ceil() as u64))
            .unwrap_or_else(|| "retry pending".to_string());
        format!(
            "daemon not running\n{}\n\n{countdown} — press r to retry now",
            sanitize::clean(&app.conn.reason, 200)
        )
    } else if !app.conn.connected {
        format!(
            "daemon not running — showing the last {} events\n{}",
            app.total_events(),
            sanitize::clean(&app.conn.reason, 200)
        )
    } else if app.filters.any_active() {
        "no events match the current filter\n\n/ edit filter · s severity · u unlabeled".to_string()
    } else {
        "waiting for events…\n\nthe daemon streams new events as they are judged".to_string()
    };
    centered_text(
        frame,
        area,
        &text,
        theme.text(theme::DIM, Modifier::empty()),
    );
}

fn render_detail(frame: &mut Frame, area: Rect, app: &mut App) {
    let theme = &app.theme;
    let title = match app.list.selected_id() {
        Some(id) => format!(" detail #{id} "),
        None => " detail ".to_string(),
    };
    let border = if app.focus == Focus::Detail {
        theme::ACCENT
    } else {
        theme::DIM
    };
    let block = Block::bordered()
        .title(Span::styled(title, theme.text(theme::DIM, Modifier::BOLD)))
        .border_style(Style::default().fg(theme.fg(border)));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let Some(event) = app.selected_event().cloned() else {
        centered_text(
            frame,
            inner,
            "no event selected\n\npick a row in the list (↑ ↓ / j k)",
            theme.text(theme::DIM, Modifier::empty()),
        );
        return;
    };
    let lines = detail_lines(&event, usize::from(inner.width), theme);
    app.detail_viewport = usize::from(inner.height).max(1);
    let max_offset = lines.len().saturating_sub(app.detail_viewport);
    app.detail_offset = app.detail_offset.min(max_offset);
    frame.render_widget(
        Paragraph::new(Text::from(lines)).scroll((app.detail_offset as u16, 0)),
        inner,
    );
}

// -- footer -------------------------------------------------------------------------------------

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let mut spans: Vec<Span<'static>> = Vec::new();
    for (index, (key, label)) in KEYS.iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw("  "));
        }
        spans.push(Span::styled(
            format!("{key} "),
            theme.text(theme::ACCENT, Modifier::BOLD),
        ));
        spans.push(Span::styled(
            (*label).to_string(),
            theme.text(theme::DIM, Modifier::empty()),
        ));
    }
    let right = match app.flash_text() {
        Some(text) => Span::styled(
            format!(" {} ", clip(text, 44)),
            theme.paint_mod(theme::YELLOW, theme::CHIP_BG, Modifier::BOLD),
        ),
        None => Span::styled(
            format!(
                "{} · tab focus",
                match app.focus {
                    Focus::List => "list",
                    Focus::Detail => "detail",
                }
            ),
            theme.text(theme::DIM, Modifier::empty()),
        ),
    };
    render_left_right(frame, area, Line::from(spans), right);
}

// -- help overlay -------------------------------------------------------------------------------

fn render_help(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let width = area.width.saturating_sub(4).min(78);
    let height = area.height.saturating_sub(2).min(24);
    let rect = Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    };
    frame.render_widget(Clear, rect);
    let block = Block::bordered()
        .title(Span::styled(
            " help ",
            theme.text(theme::ACCENT, Modifier::BOLD),
        ))
        .border_style(Style::default().fg(theme.fg(theme::ACCENT)));
    let inner = block.inner(rect);
    let block = match theme.bg(theme::BG) {
        Some(bg) => block.style(Style::default().bg(bg)),
        None => block,
    };
    frame.render_widget(block, rect);

    let mut lines: Vec<Line<'static>> = Vec::new();
    help_section(&mut lines, "navigate", theme);
    help_row(
        &mut lines,
        "↑ ↓ / j k",
        "move selection (with detail focused: scroll it)",
        theme,
    );
    help_row(
        &mut lines,
        "pgup pgdn",
        "page up / down · home / g first · end / G last",
        theme,
    );
    help_row(
        &mut lines,
        "tab",
        "switch focus between the list and the detail pane",
        theme,
    );
    help_row(&mut lines, "wheel", "scroll the focused pane", theme);
    help_row(
        &mut lines,
        "click",
        "select the row under the pointer; the newest row resumes follow",
        theme,
    );
    lines.push(Line::default());
    help_section(
        &mut lines,
        "human input (sent to the daemon; the TUI decides nothing)",
        theme,
    );
    help_row(
        &mut lines,
        "t / f",
        "label true-positive / false-positive",
        theme,
    );
    help_row(&mut lines, "y", "ack the selected event", theme);
    help_row(
        &mut lines,
        "a",
        "allow: skip future launches like this one (detail pane open)",
        theme,
    );
    lines.push(Line::default());
    help_section(&mut lines, "view", theme);
    help_row(&mut lines, "p", "pause / resume the judge", theme);
    help_row(
        &mut lines,
        "/",
        "text filter — enter applies, esc cancels",
        theme,
    );
    help_row(
        &mut lines,
        "u",
        "unlabeled only · s cycle minimum class",
        theme,
    );
    help_row(
        &mut lines,
        "d",
        "detail panel: below / beside / hidden",
        theme,
    );
    help_row(&mut lines, "r", "reconnect now", theme);
    lines.push(Line::default());
    help_row(&mut lines, "?", "close this help · q quit", theme);

    frame.render_widget(Paragraph::new(Text::from(lines)), inner);
}

// -- allow chooser ------------------------------------------------------------------------------

fn render_chooser(frame: &mut Frame, area: Rect, app: &App) {
    let Some(chooser) = &app.chooser else {
        return;
    };
    let theme = &app.theme;
    let width = area.width.saturating_sub(4).min(100);
    let inner_width = usize::from(width.saturating_sub(2));
    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!("allow launches like #{}", chooser.event_id),
        theme.text(theme::DIM, Modifier::empty()),
    )));
    lines.push(Line::default());
    for scope in chooser.scopes() {
        let selected = scope == chooser.scope;
        let mut name = scope.name().to_string();
        if scope == Scope::Prefix {
            name = format!("prefix k={}", chooser.k);
        }
        let marker = if selected { "> " } else { "  " };
        let head = format!("{marker}{name:<12}");
        let rule = clip(
            &chooser.rule_for(scope),
            inner_width.saturating_sub(label_width(&head)),
        );
        let (head_style, rule_style) = if selected {
            (
                theme.text(theme::ACCENT, Modifier::BOLD),
                theme.text(theme::FG, Modifier::BOLD),
            )
        } else {
            (
                theme.text(theme::DIM, Modifier::empty()),
                theme.text(theme::DIM, Modifier::empty()),
            )
        };
        lines.push(Line::from(vec![
            Span::styled(head, head_style),
            Span::styled(rule, rule_style),
        ]));
    }
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        "rule to add:",
        theme.text(theme::DIM, Modifier::empty()),
    )));
    for row in wrap(&chooser.rule(), inner_width.saturating_sub(2)) {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(row, theme.text(theme::FG, Modifier::BOLD)),
        ]));
    }
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        "↑ ↓ scope · ← → / - + arguments kept · enter add · esc cancel",
        theme.text(theme::DIM, Modifier::empty()),
    )));

    let height = (lines.len() as u16 + 2).min(area.height);
    let rect = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };
    frame.render_widget(Clear, rect);
    let block = Block::bordered()
        .title(Span::styled(
            " allow ",
            theme.text(theme::ACCENT, Modifier::BOLD),
        ))
        .border_style(Style::default().fg(theme.fg(theme::ACCENT)));
    let inner = block.inner(rect);
    let block = match theme.bg(theme::BG) {
        Some(bg) => block.style(Style::default().bg(bg)),
        None => block,
    };
    frame.render_widget(block, rect);
    frame.render_widget(Paragraph::new(Text::from(lines)), inner);
}

fn help_section(lines: &mut Vec<Line<'static>>, title: &str, theme: &Theme) {
    lines.push(Line::from(Span::styled(
        format!("─ {title} "),
        theme.text(theme::DIM, Modifier::BOLD),
    )));
}

fn help_row(lines: &mut Vec<Line<'static>>, key: &str, description: &str, theme: &Theme) {
    lines.push(Line::from(vec![
        Span::styled(
            format!("  {key:<9} "),
            theme.text(theme::ACCENT, Modifier::BOLD),
        ),
        Span::styled(
            description.to_string(),
            theme.text(theme::FG, Modifier::empty()),
        ),
    ]));
}

// -- event table --------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Columns {
    /// The verdict: class name and the model's confidence beside it.
    verdict: u16,
    time: u16,
    command: u16,
    label: u16,
}

/// Width of the verdict column: marker, class name (4), space, confidence (4, e.g. " 80%"), separator.
const VERDICT_WIDTH: u16 = 11;

/// Column widths for a table of the given width; the widths always sum to `width`. The command
/// column takes whatever the fixed columns leave.
fn columns(width: u16) -> Columns {
    if width >= 100 {
        // verdict 11 + time 9 + label 14
        Columns {
            verdict: VERDICT_WIDTH,
            time: 9,
            label: 14,
            command: width - 34,
        }
    } else if width >= 44 {
        // verdict 11 + time 9
        Columns {
            verdict: VERDICT_WIDTH,
            time: 9,
            label: 0,
            command: width - 20,
        }
    } else {
        let verdict = width.min(VERDICT_WIDTH);
        let time = width.saturating_sub(verdict).min(9);
        let command = width.saturating_sub(verdict + time);
        Columns {
            verdict,
            time,
            label: 0,
            command,
        }
    }
}

fn header_line(columns: &Columns, theme: &Theme) -> Line<'static> {
    Line::from(vec![
        cell_span(
            "verdict",
            columns.verdict,
            theme,
            theme::DIM,
            Modifier::BOLD,
        ),
        cell_span("time", columns.time, theme, theme::DIM, Modifier::BOLD),
        cell_span(
            "command",
            columns.command,
            theme,
            theme::DIM,
            Modifier::BOLD,
        ),
        cell_span("label", columns.label, theme, theme::DIM, Modifier::BOLD),
    ])
}

fn cell_span(
    text: &str,
    width: u16,
    theme: &Theme,
    color: Color,
    modifiers: Modifier,
) -> Span<'static> {
    Span::styled(cell(text, width), theme.text(color, modifiers))
}

fn event_line(
    event: &EventPayload,
    columns: &Columns,
    theme: &Theme,
    selected: bool,
) -> Line<'static> {
    let name = severity_name(event.severity);
    let confidence = match event.confidence {
        Some(c) if event.judged => format!(" {:.0}%", c * 100.0),
        _ => String::new(),
    };
    let marker = if selected { '▸' } else { ' ' };
    let width = usize::from(columns.verdict.saturating_sub(1));
    let chip: String = format!("{marker}{name:<4}{confidence}")
        .chars()
        .take(width)
        .collect();
    let mut spans = vec![Span::styled(
        format!("{chip:<width$}"),
        severity_style(event.severity, theme),
    )];
    if columns.verdict > 0 {
        spans.push(Span::styled(" ", Style::default()));
    }
    spans.push(Span::styled(
        cell(&format::timestamp(event.time), columns.time),
        theme.text(theme::DIM, Modifier::empty()),
    ));
    // The command line; an older daemon sends none, so fall back to its one-line message.
    let command = if event.command.is_empty() {
        &event.message
    } else {
        &event.command
    };
    spans.extend(command_spans(&cell(command, columns.command), theme));
    if columns.label > 0 {
        let badge = label_badge(event);
        let (color, modifiers) = match event.label.as_deref() {
            Some("true-positive") => (theme::GREEN, Modifier::empty()),
            Some("false-positive") => (theme::MAGENTA, Modifier::empty()),
            Some(_) => (theme::ACCENT, Modifier::empty()),
            None => (theme::DIM, Modifier::empty()),
        };
        let (color, modifiers) = if event.acked {
            (theme::GREEN, Modifier::BOLD)
        } else {
            (color, modifiers)
        };
        spans.push(Span::styled(
            cell(&badge, columns.label),
            theme.paint_mod(color, theme::CHIP_BG, modifiers),
        ));
    }
    let line = Line::from(spans);
    if selected {
        // Unmistakable: a marker on the row, the selection background (or, without colour,
        // reverse video) and bold text.
        let mut style = Style::default()
            .bg(theme.bg(theme::SELECTED_BG).unwrap_or(Color::Reset))
            .add_modifier(Modifier::BOLD);
        if !theme.color {
            style = style.add_modifier(Modifier::REVERSED);
        }
        line.style(style)
    } else {
        line
    }
}

fn severity_style(severity: i64, theme: &Theme) -> Style {
    match severity {
        0 => theme.paint_mod(Color::Rgb(244, 255, 246), theme::SAFE_BG, Modifier::BOLD),
        1 => theme.paint_mod(Color::Rgb(30, 26, 16), theme::WARN_BG, Modifier::BOLD),
        2 => theme.paint_mod(Color::Rgb(255, 244, 244), theme::CRIT_BG, Modifier::BOLD),
        _ => theme.text(theme::FG, Modifier::BOLD),
    }
}

fn label_badge(event: &EventPayload) -> String {
    let base = match event.label.as_deref() {
        None => "-".to_string(),
        Some("true-positive") => "TP".to_string(),
        Some("false-positive") => "FP".to_string(),
        Some(other) => clip(other, 9),
    };
    if event.acked {
        format!("{base} ✓")
    } else {
        base
    }
}

// -- detail pane --------------------------------------------------------------------------------

/// The detail pane's content as lines, for any width: summary header, verdict probabilities as
/// bars, the model-facing context as aligned key/value facts, and the message.
pub fn detail_lines(event: &EventPayload, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let width = width.max(8);
    let mut lines: Vec<Line<'static>> = Vec::new();

    // Summary header.
    let mut summary = vec![
        Span::styled(
            format!("#{}", event.id),
            theme.text(theme::ACCENT, Modifier::BOLD),
        ),
        Span::styled(" ", Style::default()),
        Span::styled(
            format::timestamp(event.time),
            theme.text(theme::DIM, Modifier::empty()),
        ),
        Span::styled(" ", Style::default()),
        Span::styled(
            format!(" {:<4} ", severity_name(event.severity)),
            severity_style(event.severity, theme),
        ),
    ];
    if let Some(confidence) = event.confidence.filter(|_| event.judged) {
        summary.push(Span::styled(" ", Style::default()));
        summary.push(Span::styled(
            format!("conf {:.0}%", confidence * 100.0),
            theme.text(theme::FG, Modifier::BOLD),
        ));
    }
    lines.push(Line::from(summary));
    lines.push(Line::from(Span::styled(
        "─".repeat(width.min(200)),
        theme.text(theme::DIM, Modifier::empty()),
    )));

    // Key facts.
    let label = match event.label.as_deref() {
        Some(label) => {
            if event.acked {
                format!("{label} · acked")
            } else {
                label.to_string()
            }
        }
        None => {
            if event.acked {
                "ack".to_string()
            } else {
                "-".to_string()
            }
        }
    };
    let state = if event.judged || event.cached || event.available {
        [
            event.judged.then_some("judged"),
            event.cached.then_some("cached"),
            event.available.then_some("available"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" · ")
    } else {
        "not judged".to_string()
    };
    push_fact(&mut lines, "command", &event.command, width, theme);
    push_fact(&mut lines, "subject", &event.subject, width, theme);
    let context = structure_context(&event.context);
    if let Some(exe) = context_value(&context, "exe") {
        push_fact(&mut lines, "exe", &exe, width, theme);
    }
    push_fact(&mut lines, "verdict", &event.verdict, width, theme);
    push_fact(
        &mut lines,
        "state",
        &format!("{state} · label {label}"),
        width,
        theme,
    );
    push_fact(
        &mut lines,
        "kind",
        &format!("{} · {}", event.kind, event.source),
        width,
        theme,
    );

    // Probability bars.
    if !event.probabilities.is_empty() {
        lines.push(Line::default());
        lines.push(section("probabilities", theme));
        let mut probs: Vec<(&String, f64)> =
            event.probabilities.iter().map(|(k, v)| (k, *v)).collect();
        probs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let name_width = probs
            .iter()
            .map(|(name, _)| display_width(name))
            .max()
            .unwrap_or(0)
            .clamp(6, 24);
        let bar_width = width.saturating_sub(name_width + 14).clamp(6, 40);
        for (name, probability) in probs {
            let filled = (probability.clamp(0.0, 1.0) * bar_width as f64).round() as usize;
            let filled = filled.min(bar_width);
            let style = theme.tier_text(theme::class_tier(name));
            let color = style.fg.unwrap_or(Color::Reset);
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(format!("{:<name_width$}", clip(name, name_width)), style),
                Span::styled(" ", Style::default()),
                Span::styled("█".repeat(filled), Style::default().fg(color)),
                Span::styled(
                    "░".repeat(bar_width - filled),
                    theme.text(theme::DIM, Modifier::empty()),
                ),
                Span::styled(" ", Style::default()),
                Span::styled(
                    format!("{:>6}", format_percent(probability)),
                    theme.text(theme::FG, Modifier::empty()),
                ),
            ]));
        }
    }

    // Model-facing context as aligned facts.
    if !context.is_empty() {
        lines.push(Line::default());
        lines.push(section("model-facing context", theme));
        lines.extend(context_lines(&context, width, theme));
    }

    // Message.
    if !event.message.is_empty() {
        lines.push(Line::default());
        lines.push(section("message", theme));
        for chunk in wrap(&event.message, width.saturating_sub(2)) {
            lines.push(Line::from(Span::styled(
                format!("  {chunk}"),
                theme.text(theme::FG, Modifier::empty()),
            )));
        }
    }
    lines
}

fn severity_name(severity: i64) -> &'static str {
    usize::try_from(severity)
        .ok()
        .and_then(|s| SEVERITY_NAMES.get(s))
        .copied()
        .unwrap_or("sev?")
}

fn section(title: &str, theme: &Theme) -> Line<'static> {
    Line::from(Span::styled(
        format!("── {title} "),
        theme.text(theme::DIM, Modifier::BOLD),
    ))
}

/// Colour a command line zsh style: the executable (first word, and the word after `|`, `&&`,
/// `;`) in bold green, `-flags` in cyan, path- and URL-shaped words in the accent colour. Shape
/// only, no names; whitespace is kept so the cell width is unchanged.
fn command_spans(text: &str, theme: &Theme) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut expect_exe = true;
    let mut rest = text;
    while !rest.is_empty() {
        let blank = rest.len() - rest.trim_start().len();
        if blank > 0 {
            spans.push(Span::styled(
                rest[..blank].to_string(),
                theme.text(theme::FG, Modifier::empty()),
            ));
            rest = &rest[blank..];
            continue;
        }
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let word = &rest[..end];
        let (color, modifiers) = if matches!(word, "|" | "||" | "&&" | ";") {
            expect_exe = true;
            (theme::DIM, Modifier::empty())
        } else if expect_exe {
            expect_exe = false;
            (theme::GREEN, Modifier::BOLD)
        } else if word.starts_with('-') {
            (theme::CYAN, Modifier::empty())
        } else if word.contains('/') || word.starts_with('~') {
            (theme::ACCENT, Modifier::empty())
        } else {
            (theme::FG, Modifier::empty())
        };
        spans.push(Span::styled(word.to_string(), theme.text(color, modifiers)));
        rest = &rest[end..];
    }
    spans
}

fn push_fact(lines: &mut Vec<Line<'static>>, key: &str, value: &str, width: usize, theme: &Theme) {
    if value.is_empty() {
        return;
    }
    let key_width = 8usize;
    let indent = " ".repeat(key_width + 4);
    for (index, chunk) in wrap(value, width.saturating_sub(key_width + 4))
        .into_iter()
        .enumerate()
    {
        if index == 0 {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {key:<key_width$}"),
                    theme.text(theme::DIM, Modifier::BOLD),
                ),
                Span::styled(": ", theme.text(theme::DIM, Modifier::empty())),
                Span::styled(chunk, theme.text(theme::FG, Modifier::empty())),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                format!("{indent}{chunk}"),
                theme.text(theme::FG, Modifier::empty()),
            )));
        }
    }
}

/// The daemon flattens the model-facing context to one line (the model reads it that way).
/// Restore its facts for display: a whitespace-separated token shaped like a key (`exe:`,
/// `user(uid):`) starts a new line. Shape only, no names; text that already has lines passes
/// through unchanged.
fn structure_context(context: &str) -> String {
    if context.contains('\n') {
        return context.to_string();
    }
    fn is_key(token: &str) -> bool {
        let Some(name) = token.strip_suffix(':') else {
            return false;
        };
        let base = match name.split_once('(') {
            Some((base, rest)) => {
                if !rest.ends_with(')') {
                    return false;
                }
                base
            }
            None => name,
        };
        !base.is_empty()
            && base.len() <= 24
            && base
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    }
    let mut out = String::with_capacity(context.len() + 16);
    for token in context.split_whitespace() {
        if !out.is_empty() {
            out.push(if is_key(token) { '\n' } else { ' ' });
        }
        out.push_str(token);
    }
    out
}

/// The value of a `key: value` segment of the context (segments are split like `context_lines`).
fn context_value(context: &str, key: &str) -> Option<String> {
    context
        .lines()
        .flat_map(|line| line.split("  "))
        .filter_map(|part| part.trim().split_once(": "))
        .find(|(k, _)| *k == key)
        .map(|(_, value)| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Render the context string as aligned `key: value` facts: lines are split on the double-space
/// separators the daemon uses, pairs align on their keys, anything else passes through as text.
fn context_lines(context: &str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let mut segments: Vec<(&str, Option<(&str, &str)>)> = Vec::new();
    for line in context.lines() {
        for part in line.split("  ") {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            segments.push(match part.split_once(": ") {
                Some((key, value)) if key.len() <= 24 => (part, Some((key, value))),
                _ => (part, None),
            });
        }
    }
    let key_width = segments
        .iter()
        .filter_map(|(_, pair)| pair.as_ref())
        .map(|(key, _)| display_width(key))
        .max()
        .unwrap_or(0)
        .clamp(6, 24);

    let mut lines = Vec::new();
    for (part, pair) in segments {
        match pair {
            None => {
                for chunk in wrap(part, width.saturating_sub(4)) {
                    lines.push(Line::from(Span::styled(
                        format!("  {chunk}"),
                        theme.text(theme::FG, Modifier::empty()),
                    )));
                }
            }
            Some((key, value)) => {
                for (index, chunk) in wrap(value, width.saturating_sub(key_width + 6))
                    .into_iter()
                    .enumerate()
                {
                    if index == 0 {
                        lines.push(Line::from(vec![
                            Span::styled("  ", Style::default()),
                            Span::styled(
                                format!("{:<key_width$}", clip(key, key_width)),
                                theme.text(theme::ACCENT, Modifier::empty()),
                            ),
                            Span::styled(": ", theme.text(theme::DIM, Modifier::empty())),
                            Span::styled(chunk, theme.text(theme::FG, Modifier::empty())),
                        ]));
                    } else {
                        lines.push(Line::from(Span::styled(
                            format!("    {}", " ".repeat(key_width) + &chunk),
                            theme.text(theme::FG, Modifier::empty()),
                        )));
                    }
                }
            }
        }
    }
    lines
}

// -- text helpers --------------------------------------------------------------------------------

fn centered_text(frame: &mut Frame, area: Rect, text: &str, style: Style) {
    let count = text.lines().count().max(1) as u16;
    let height = count.min(area.height);
    let rect = Rect {
        y: area.y + (area.height.saturating_sub(height)) / 2,
        height,
        ..area
    };
    frame.render_widget(
        Paragraph::new(text.to_string())
            .style(style)
            .alignment(Alignment::Center),
        rect,
    );
}

fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn label_width(text: &str) -> usize {
    display_width(text)
}

fn line_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum()
}

/// Cut `text` to at most `max` display columns, marking the cut with `…`.
pub fn clip(text: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if display_width(text) <= max {
        return text.to_string();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in text.chars() {
        let width = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
        if used + width > max.saturating_sub(1) {
            break;
        }
        out.push(c);
        used += width;
    }
    if max == 1 {
        return "…".to_string();
    }
    out.push('…');
    out
}

/// A table cell: content clipped to `width - 1` and padded with a trailing space to `width`.
fn cell(text: &str, width: u16) -> String {
    let width = usize::from(width);
    if width == 0 {
        return String::new();
    }
    let content = clip(text, width.saturating_sub(1));
    let pad = width.saturating_sub(display_width(&content));
    format!("{content}{}", " ".repeat(pad))
}

/// Greedy display-width wrapping (no hyphenation: these are paths and machine strings).
fn wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut used = 0;
    for c in text.chars() {
        let char_width = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
        if used + char_width > width {
            lines.push(std::mem::take(&mut current));
            used = 0;
        }
        current.push(c);
        used += char_width;
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

// -- test/fixture rendering ---------------------------------------------------------------------

/// Render `app` into a `TestBackend` frame and return the buffer as plain text (rows joined by
/// newlines). Used by tests.
pub fn render_to_string(app: &mut App, width: u16, height: u16) -> String {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal.draw(|frame| draw(frame, app)).expect("draw");
    buffer_text(terminal.backend().buffer())
}

/// A frame's text content: every cell's symbol, row by row.
pub fn buffer_text(buffer: &ratatui::buffer::Buffer) -> String {
    let area = buffer.area;
    let mut out = String::new();
    for y in 0..area.height {
        for x in 0..area.width {
            out.push_str(buffer.cell((x, y)).map_or(" ", |cell| cell.symbol()));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::DaemonMsg;
    use crate::protocol::Stats;
    use crate::state::LayoutMode;

    fn payload(id: u64, message: &str, subject: &str) -> EventPayload {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "time": 1_728_000_000.0 + f64::from(id as u32),
            "source": "exec",
            "subject": subject,
            "kind": "verdict",
            "severity": (id % 3) as i64,
            "message": message,
            "context": "process launch (tier 0 /proc poller)\nexe: /usr/bin/curl  basename: curl\nenv: hour=13 fork_rate=2.10",
            "verdict": "model: crit 0.90",
            "class": match id % 3 { 0 => "safe", 1 => "warn", _ => "crit" },
            "confidence": 0.9,
            "probabilities": {"crit": 0.9, "safe": 0.1},
            "judged": true,
            "cached": false,
            "available": true,
            "label": (id == 1).then_some("true-positive"),
            "labels": [],
            "acked": id == 1,
        }))
        .expect("fixture payload")
    }

    fn app() -> App {
        let mut app = App::new("/run/user/1000/jevmon/jevmond.sock".to_string());
        app.theme = Theme::color();
        app
    }

    #[test]
    fn header_says_when_the_exec_helper_is_missing() {
        let mut app = app();
        app.handle_msg(DaemonMsg::Connected);
        app.handle_msg(DaemonMsg::Stats(Stats {
            model_status: "ready".to_string(),
            exec_source: "ebpf".to_string(),
            exec_error: "exec source unavailable (helper not running); run: jevmon start"
                .to_string(),
            exec_per_s: 12.0,
            ..Stats::default()
        }));
        let text = render_to_string(&mut app, 160, 30);
        assert!(text.contains("exec source unavailable (helper not running)"));
        assert!(text.contains("run: jevmon start"));
        assert!(text.contains("exec/s") && text.contains("lost"));
    }

    #[test]
    fn renders_header_metrics_table_detail_and_footer() {
        let mut app = app();
        app.ingest_all(vec![
            payload(1, "first message", "42"),
            payload(2, "second message", "7"),
        ]);
        app.handle_msg(DaemonMsg::Connected);
        app.handle_msg(DaemonMsg::Stats(Stats {
            uptime: 3_725.0,
            model_status: "ready".to_string(),
            effective_interval: 0.05,
            miss_ratio: 0.098,
            judged: 12,
            queue_depth: 3,
            queue_age: 200.0,
            ..Stats::default()
        }));
        let text = render_to_string(&mut app, 120, 40);

        assert!(text.contains("jevmon"), "header title");
        assert!(text.contains("● connected"), "connection badge");
        assert!(text.contains("model ready"), "model badge");
        assert!(text.contains("1h 2m"), "uptime");
        assert!(text.contains("judged/s"), "metric labels");
        assert!(text.contains("queue age"), "metric labels");
        assert!(text.contains("50ms"), "interval value");
        assert!(text.contains("9.8%"), "miss value");
        assert!(text.contains("time"), "table header");
        assert!(text.contains("verdict"), "table header");
        assert!(text.contains("second message"), "event row");
        assert!(
            text.contains("crit 0.90"),
            "class beside the model's confidence"
        );
        assert!(text.contains("TP ✓"), "label badge");
        assert!(text.contains("detail #"), "detail title");
        assert!(text.contains("probabilities"), "detail section");
        assert!(text.contains("model-facing context"), "detail section");
        assert!(text.contains("reconnect"), "footer keys");
        assert!(text.contains("q quit"), "footer keys");
        assert!(!text.contains('\u{1b}'), "no raw escapes");
    }

    #[test]
    fn fake_judge_status_shows_as_mock_judge_not_as_a_model_error() {
        let mut app = app();
        app.ingest_all(vec![payload(1, "first message", "42")]);
        app.handle_msg(DaemonMsg::Connected);
        app.handle_msg(DaemonMsg::Stats(Stats {
            model_status: "fake deterministic stand-in".to_string(),
            ..Stats::default()
        }));
        let text = render_to_string(&mut app, 100, 26);
        assert!(text.contains("mock judge"), "header badge: {text}");
        assert!(!text.contains("model fake"), "not an error badge: {text}");
    }

    #[test]
    fn attacker_text_renders_literally_without_escapes() {
        let mut app = app();
        let mut event = payload(1, "hello", "1");
        event.message =
            "argv [bold red]x[/] \u{1b}[31mred\u{1b}[0m \u{1b}]0;pwned\u{7} end".to_string();
        event.subject = "[b]s\u{1b}[2J".to_string();
        let mut with_command = payload(2, "m", "2");
        with_command.command = "run [inverse]cmd\u{1b}[2J".to_string();
        app.ingest_all(vec![event, with_command]);
        app.list.select_id(1);
        let text = render_to_string(&mut app, 120, 40);
        assert!(
            text.contains("[bold red]x[/]"),
            "markup shown literally: {text:?}"
        );
        assert!(text.contains("red"), "ansi payload shown as text");
        assert!(text.contains("[b]s"), "subject literal");
        assert!(text.contains("run [inverse]cmd"), "command literal");
        assert!(!text.contains('\u{1b}'), "no escape bytes reach the buffer");
        assert!(!text.contains('\u{7}'), "no BEL from the OSC sequence");
    }

    #[test]
    fn disconnected_state_shows_a_countdown_and_never_crashes() {
        let mut app = app();
        app.handle_msg(DaemonMsg::Disconnected {
            reason: "cannot connect to /run/x.sock: No such file or directory".to_string(),
            retry_in: 2,
        });
        let text = render_to_string(&mut app, 100, 30);
        assert!(text.contains("daemon not running"));
        assert!(text.contains("retry in"), "countdown visible: {text:?}");
        assert!(text.contains("press r to retry"));
        // still renders fine with no events and no stats
        let tiny = render_to_string(&mut app, 30, 8);
        assert!(tiny.contains("jevmon") || tiny.contains("small"));
    }

    #[test]
    fn no_color_theme_renders_no_colours() {
        let mut app = app();
        app.theme = Theme::monochrome();
        app.ingest_all(vec![payload(1, "msg", "42")]);
        app.handle_msg(DaemonMsg::Connected);
        app.handle_msg(DaemonMsg::Stats(Stats {
            model_status: "ready".to_string(),
            ..Stats::default()
        }));
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let buffer = terminal.backend().buffer();
        let area = buffer.area;
        for y in 0..area.height {
            for x in 0..area.width {
                let cell = buffer.cell((x, y)).expect("in-bounds cell");
                assert_eq!(cell.fg, Color::Reset, "fg at {x},{y}");
                assert_eq!(cell.bg, Color::Reset, "bg at {x},{y}");
            }
        }
        // ... and the text is still all there
        let text = buffer_text(buffer);
        assert!(text.contains("connected"));
        assert!(text.contains("msg"));
    }

    #[test]
    fn filter_chips_and_input_render() {
        let mut app = app();
        app.ingest_all(vec![payload(1, "curl crit", "7")]);
        app.handle_msg(DaemonMsg::Connected);
        app.cycle_severity();
        app.toggle_unlabeled();
        app.filters.text = "curl".to_string();
        app.rebuild();
        let text = render_to_string(&mut app, 110, 24);
        assert!(text.contains("≥ warn"), "severity chip");
        assert!(text.contains("unlabeled"), "unlabeled chip");
        assert!(text.contains("text: curl"), "text chip");

        app.open_filter();
        let text = render_to_string(&mut app, 110, 24);
        assert!(text.contains("filter▸"), "input line visible");
    }

    #[test]
    fn help_overlay_toggles() {
        let mut app = app();
        app.ingest_all(vec![payload(1, "m", "1")]);
        let closed = render_to_string(&mut app, 100, 30);
        assert!(!closed.contains("human input"));
        app.help = true;
        let open = render_to_string(&mut app, 100, 30);
        assert!(open.contains("human input"));
        assert!(open.contains("close this help"));
        assert!(open.contains("true-positive"));
    }

    #[test]
    fn empty_states_appear_for_filters_and_waiting() {
        let mut app = app();
        app.handle_msg(DaemonMsg::Connected);
        let waiting = render_to_string(&mut app, 100, 26);
        assert!(waiting.contains("waiting for events"));

        app.ingest_all(vec![payload(1, "curl", "1")]);
        app.filters.text = "nothing-matches".to_string();
        app.rebuild();
        let filtered = render_to_string(&mut app, 100, 26);
        assert!(filtered.contains("no events match"));
    }

    #[test]
    fn detail_lines_have_probability_bars_and_aligned_facts() {
        let event = payload(1, "a message", "42");
        let lines = detail_lines(&event, 90, &Theme::color());
        let text: String = lines
            .iter()
            .map(|line| {
                let mut row = String::new();
                for span in &line.spans {
                    row.push_str(span.content.as_ref());
                }
                row.push('\n');
                row
            })
            .collect();
        assert!(text.contains('#'), "summary id");
        assert!(text.contains("90%"), "top probability");
        assert!(text.contains("█"), "probability bar");
        assert!(text.contains("░"), "probability track");
        assert!(text.contains("verdict"), "fact key");
        assert!(text.contains("model: crit 0.90"), "fact value");
        assert!(text.contains("exe"), "context fact");
        assert!(text.contains("basename"), "context fact");
        // aligned facts: keys start at the same column, colons line up
        let exe = text.lines().find(|l| l.contains("/usr/bin/curl")).unwrap();
        let base = text.lines().find(|l| l.contains("basename")).unwrap();
        assert_eq!(exe.find("exe"), base.find("basename"));
        assert_eq!(exe.find(':'), base.find(':'));
    }

    #[test]
    fn list_shows_the_command_and_detail_the_full_command_and_exe() {
        let mut app = app();
        let mut event = payload(1, "judge: safe 0.90", "2410");
        event.command = "make -j8 all".to_string();
        let mut older = payload(2, "judge: from an older daemon", "2411");
        older.command = String::new();
        app.ingest_all(vec![event, older]);
        app.list.select_id(1);
        let text = render_to_string(&mut app, 130, 40);
        assert!(text.contains("command"), "column header: {text}");
        let rows = text.lines().filter(|l| l.contains("make -j8 all")).count();
        assert!(
            rows >= 2,
            "command in the list row and in the detail: {text}"
        );
        assert!(
            !text.lines().any(|l| l.contains("judge: safe")),
            "message replaced by the command"
        );
        assert!(
            text.contains("from an older daemon"),
            "falls back to the message"
        );

        let lines = detail_lines(&app.events[&1], 80, &Theme::color());
        let detail: String = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
                    + "\n"
            })
            .collect();
        assert!(
            detail.contains("command") && detail.contains("make -j8 all"),
            "{detail}"
        );
        assert!(
            detail.contains("exe") && detail.contains("/usr/bin/curl"),
            "{detail}"
        );
    }

    #[test]
    fn long_command_is_clipped_in_the_list_and_wrapped_in_the_detail() {
        let mut app = app();
        let mut event = payload(1, "m", "2410");
        event.command = format!("tool {}", "--flag=value ".repeat(30));
        app.ingest_all(vec![event]);
        let text = render_to_string(&mut app, 100, 40);
        for line in text.lines() {
            assert!(line.chars().count() <= 100, "row wider than the frame");
        }
        assert!(text.contains('…'), "clipped command marked: {text}");
        let lines = detail_lines(&app.events[&1], 60, &Theme::color());
        assert!(lines.len() > 8, "wrapped across several lines");
    }

    /// The foreground colour of the first cell of `needle` in the first row containing `row_key`.
    fn fg_of(buffer: &ratatui::buffer::Buffer, row_key: &str, needle: &str) -> Color {
        let area = buffer.area;
        for y in 0..area.height {
            let row: String = (0..area.width)
                .map(|x| buffer.cell((x, y)).map_or(" ", |c| c.symbol()))
                .collect();
            if !row.contains(row_key) {
                continue;
            }
            if let Some(at) = row.find(needle) {
                let x = row[..at].chars().count() as u16;
                return buffer.cell((x, y)).expect("cell").fg;
            }
        }
        panic!("`{needle}` not found in a row with `{row_key}`");
    }

    fn drawn(app: &mut App) -> ratatui::buffer::Buffer {
        let mut terminal = Terminal::new(TestBackend::new(130, 40)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        terminal.backend().buffer().clone()
    }

    fn bg_of(buffer: &ratatui::buffer::Buffer, row_key: &str, needle: &str) -> Color {
        let area = buffer.area;
        for y in 0..area.height {
            let row: String = (0..area.width)
                .map(|x| buffer.cell((x, y)).map_or(" ", |c| c.symbol()))
                .collect();
            if !row.contains(row_key) {
                continue;
            }
            if let Some(at) = row.find(needle) {
                let x = row[..at].chars().count() as u16;
                return buffer.cell((x, y)).expect("cell").bg;
            }
        }
        panic!("`{needle}` not found in a row with `{row_key}`");
    }

    fn with_class(id: u64, subject: &str, severity: i64) -> EventPayload {
        let mut event = payload(id, "m", subject);
        event.command = format!("cmd-{subject}");
        event.severity = severity;
        event.class = ["safe", "warn", "crit"][severity as usize].to_string();
        event
    }

    #[test]
    fn verdict_chip_colour_follows_the_class() {
        let mut app = app();
        app.ingest_all(vec![
            with_class(1, "2410", 0),
            with_class(2, "2411", 1),
            with_class(3, "2412", 2),
        ]);
        let buffer = drawn(&mut app);
        assert_eq!(bg_of(&buffer, "2410", "safe"), theme::SAFE_BG);
        assert_eq!(bg_of(&buffer, "2411", "warn"), theme::WARN_BG);
        assert_eq!(bg_of(&buffer, "2412", "crit"), theme::CRIT_BG);
    }

    #[test]
    fn red_appears_only_on_crit_rows() {
        for severity in 0..=1 {
            let mut app = app();
            app.ingest_all(vec![with_class(1, "2410", severity)]);
            app.handle_msg(DaemonMsg::Connected);
            let buffer = drawn(&mut app);
            let area = buffer.area;
            for y in 0..area.height {
                let row: String = (0..area.width)
                    .map(|x| buffer.cell((x, y)).map_or(" ", |c| c.symbol()))
                    .collect();
                if !row.contains("2410") {
                    continue;
                }
                for x in 0..area.width {
                    let cell = buffer.cell((x, y)).unwrap();
                    assert_ne!(cell.bg, theme::CRIT_BG, "severity {severity}");
                    assert_ne!(cell.fg, theme::RED, "severity {severity}");
                }
            }
        }
    }

    #[test]
    fn probability_labels_use_the_tier_of_their_class() {
        let mut app = app();
        let mut event = with_class(1, "2410", 1);
        event.probabilities = [
            ("safe".to_string(), 0.1),
            ("warn".to_string(), 0.6),
            ("crit".to_string(), 0.3),
        ]
        .into();
        app.ingest_all(vec![event]);
        app.list.select_id(1);
        let buffer = drawn(&mut app);
        assert_eq!(fg_of(&buffer, " 60%", "warn"), theme::YELLOW);
        assert_eq!(fg_of(&buffer, " 30%", "crit"), theme::RED);
        assert_eq!(fg_of(&buffer, " 10%", "safe"), theme::GREEN);
    }

    #[test]
    fn confidence_sits_next_to_the_class_and_the_frame_holds_at_every_width() {
        let mut app = app();
        let mut a = with_class(1, "2410", 2);
        a.command = "make -j8 all".to_string();
        app.ingest_all(vec![a]);
        for width in [30u16, 44, 60, 64, 80, 100, 120, 200] {
            let text = render_to_string(&mut app, width, 30);
            assert_eq!(text.lines().count(), 30);
            for line in text.lines() {
                assert!(
                    display_width(line) <= usize::from(width),
                    "row wider than {width}: {line:?}"
                );
            }
            if width >= 60 {
                assert!(text.contains("crit 0.90"), "{text}");
            }
        }
    }

    #[test]
    fn a_flattened_context_is_restructured_into_facts() {
        let flat = "process launch (tier 0 /proc poller) exe: /srv/demo/bin/x basename: x \
                    parents: bash <- sshd user(uid): 1001 cgroup: a.service \
                    argv(template): x --out <PATH> env: hour=9 fork_rate=2.1";
        let lines: Vec<String> = structure_context(flat)
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(lines[0], "process launch (tier 0 /proc poller)");
        assert!(
            lines.contains(&"exe: /srv/demo/bin/x".to_string()),
            "{lines:?}"
        );
        assert!(lines.contains(&"user(uid): 1001".to_string()), "{lines:?}");
        assert!(
            lines.contains(&"env: hour=9 fork_rate=2.1".to_string()),
            "{lines:?}"
        );
        assert_eq!(structure_context("a: b\nc: d"), "a: b\nc: d");

        let mut event = payload(1, "m", "2410");
        event.context = flat.to_string();
        let text: String = detail_lines(&event, 80, &Theme::color())
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
                    + "\n"
            })
            .collect();
        assert!(
            text.contains("exe") && text.contains("/srv/demo/bin/x"),
            "{text}"
        );
    }

    #[test]
    fn launch_facts_render_as_aligned_key_value_lines() {
        let flat = "process launch (tier 0 /proc poller) exe: /opt/demo/bin/x basename: x \
                    argv(template): x --port <NUM> argv(redacted): x --port 80 \
                    privileges: uid=1001 euid=0 capeff=000001ffffffffff nnp=0 seccomp=0 \
                    namespaces: differs(mnt,pid) [mnt=differs pid=differs] uid_map=remapped \
                    stdio: 0=null 1=pipe 2=pipe exe_state: deleted";
        let mut event = payload(1, "m", "2410");
        event.context = flat.to_string();
        let text: String = detail_lines(&event, 100, &Theme::color())
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
                    + "\n"
            })
            .collect();
        for needle in [
            "privileges",
            "euid=0",
            "namespaces",
            "differs(mnt,pid)",
            "uid_map=remapped",
            "stdio",
            "exe_state",
            "deleted",
        ] {
            assert!(text.contains(needle), "{needle} missing in\n{text}");
        }
    }

    #[test]
    fn wide_and_long_strings_do_not_break_the_frame() {
        let mut app = app();
        let mut event = payload(1, &"m".repeat(3_000), "1");
        event.subject = "日本語の件名が非常に長い場合の表示確認テスト".to_string();
        event.class = "カテゴリー".to_string();
        app.ingest_all(vec![event]);
        let text = render_to_string(&mut app, 80, 24);
        assert!(text.contains('日'), "wide subject shows");
        assert!(!text.contains('\u{1b}'));
        assert_eq!(
            text.lines().count(),
            24,
            "frame is exactly the terminal size"
        );
        for line in text.lines() {
            assert!(
                line.chars().count() <= 80,
                "row wider than the frame: {line:?}"
            );
        }
    }

    #[test]
    fn layout_toggle_cycles_without_panicking() {
        let mut app = app();
        app.ingest_all(vec![payload(1, "m", "1")]);
        for mode in [
            LayoutMode::Below,
            LayoutMode::Beside,
            LayoutMode::Hidden,
            LayoutMode::Below,
        ] {
            app.layout = mode;
            let text = render_to_string(&mut app, 90, 28);
            assert!(text.contains("events"), "table visible in {:?}", mode);
            if mode != LayoutMode::Hidden {
                assert!(text.contains("detail"), "detail visible in {:?}", mode);
            }
        }
    }

    #[test]
    fn command_words_get_their_own_colours() {
        let theme = Theme::color();
        let spans = command_spans("curl -fsS https://x/y | tee /tmp/o", &theme);
        let colour = |word: &str| {
            spans
                .iter()
                .find(|s| s.content == word)
                .and_then(|s| s.style.fg)
                .unwrap_or_else(|| panic!("{word}"))
        };
        assert_eq!(colour("curl"), theme::GREEN);
        assert_eq!(colour("-fsS"), theme::CYAN);
        assert_eq!(colour("https://x/y"), theme::ACCENT);
        assert_eq!(colour("tee"), theme::GREEN);
        let joined: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(joined, "curl -fsS https://x/y | tee /tmp/o");
    }

    #[test]
    fn columns_sum_to_their_width() {
        for width in [20u16, 40, 44, 60, 71, 72, 99, 100, 120, 200] {
            let columns = columns(width);
            let sum = columns.verdict + columns.time + columns.command + columns.label;
            assert_eq!(sum, width, "columns must tile width {width}");
        }
    }

    #[test]
    fn clip_and_cell_respect_display_width() {
        assert_eq!(clip("hello", 10), "hello");
        assert_eq!(clip("hello", 3), "he…");
        assert_eq!(clip("日本語", 3), "日…");
        assert_eq!(display_width(&cell("abcdef", 4)), 4);
        assert_eq!(cell("abcdef", 4), "ab… ");
        assert_eq!(cell("", 4), "    ");
        assert_eq!(cell("x", 0), "");
    }

    #[test]
    fn wrap_splits_on_display_width() {
        let wrapped = wrap(&"m".repeat(25), 10);
        assert_eq!(wrapped.len(), 3);
        assert_eq!(wrapped[0].len(), 10);
        assert_eq!(wrap("", 10), [""]);
    }
}

//! The TUI palette. Dark, cohesive, and degrading gracefully: with `NO_COLOR` (or `TERM=dumb`)
//! every colour collapses to the terminal default and only emphasis (bold/dim) survives.

use ratatui::style::{Color, Modifier, Style};

pub struct Theme {
    pub color: bool,
}

impl Default for Theme {
    fn default() -> Self {
        Self::from_env()
    }
}

impl Theme {
    pub fn from_env() -> Self {
        Theme {
            color: color_enabled(),
        }
    }

    /// A theme with colours enabled (tests, fixtures).
    pub fn color() -> Self {
        Theme { color: true }
    }

    /// A theme that never emits a colour (tests, `NO_COLOR`).
    pub fn monochrome() -> Self {
        Theme { color: false }
    }

    pub fn fg(&self, color: Color) -> Color {
        if self.color { color } else { Color::Reset }
    }

    pub fn bg(&self, color: Color) -> Option<Color> {
        if self.color { Some(color) } else { None }
    }

    /// Foreground-only style (modifiers always survive).
    pub fn text(&self, color: Color, modifiers: Modifier) -> Style {
        Style::default().fg(self.fg(color)).add_modifier(modifiers)
    }

    /// Foreground + background; without colour the background is dropped entirely.
    pub fn paint(&self, fg: Color, bg: Color) -> Style {
        let style = Style::default().fg(self.fg(fg));
        match self.bg(bg) {
            Some(bg) => style.bg(bg),
            None => style,
        }
    }

    pub fn paint_mod(&self, fg: Color, bg: Color, modifiers: Modifier) -> Style {
        self.paint(fg, bg).add_modifier(modifiers)
    }
}

/// `NO_COLOR` (https://no-color.org): any non-empty value switches colours off.
pub fn color_enabled() -> bool {
    if let Some(value) = std::env::var_os("NO_COLOR") {
        return value.is_empty();
    }
    if std::env::var_os("TERM").as_deref() == Some(std::ffi::OsStr::new("dumb")) {
        return false;
    }
    true
}

// -- palette ------------------------------------------------------------------------------------

pub const BG: Color = Color::Rgb(20, 22, 28);
pub const FG: Color = Color::Rgb(200, 204, 214);
pub const DIM: Color = Color::Rgb(108, 114, 128);
pub const ACCENT: Color = Color::Rgb(120, 176, 240);
pub const GREEN: Color = Color::Rgb(120, 190, 130);
pub const YELLOW: Color = Color::Rgb(220, 190, 96);
pub const RED: Color = Color::Rgb(226, 110, 110);
pub const MAGENTA: Color = Color::Rgb(192, 140, 220);
pub const CYAN: Color = Color::Rgb(120, 200, 200);
pub const SELECTED_BG: Color = Color::Rgb(52, 58, 72);
pub const CHIP_BG: Color = Color::Rgb(40, 44, 56);
pub const CRIT_BG: Color = Color::Rgb(190, 66, 66);
pub const WARN_BG: Color = Color::Rgb(176, 140, 52);
pub const SAFE_BG: Color = Color::Rgb(52, 116, 74);

/// Colour per verdict class tier: 0 safe green, 1 warn amber, 2 crit red. Red means crit and
/// nothing else. Out-of-range tiers clamp, so a future daemon cannot produce an unknown colour.
pub fn tier_color(tier: u8) -> Color {
    match tier {
        0 => GREEN,
        1 => YELLOW,
        _ => RED,
    }
}

/// The tier of a class name (`safe`, `warn`, `crit`); anything else reads as safe.
pub fn class_tier(class: &str) -> u8 {
    match class {
        "warn" => 1,
        "crit" => 2,
        _ => 0,
    }
}

/// The emphasis that keeps the tiers apart when colour is off: dim, plain, bold.
pub fn tier_modifier(tier: u8) -> Modifier {
    match tier {
        0 => Modifier::DIM,
        1 => Modifier::empty(),
        _ => Modifier::BOLD,
    }
}

impl Theme {
    /// A class label of the given tier on the chip background.
    pub fn tier_chip(&self, tier: u8) -> Style {
        self.paint_mod(tier_color(tier), CHIP_BG, self.tier_emphasis(tier))
    }

    /// A class label of the given tier on the default background (probability rows).
    pub fn tier_text(&self, tier: u8) -> Style {
        self.text(tier_color(tier), self.tier_emphasis(tier))
    }

    fn tier_emphasis(&self, tier: u8) -> Modifier {
        if self.color {
            // In colour the hue carries the tier; bold only marks the top one.
            if tier >= 2 {
                Modifier::BOLD
            } else {
                Modifier::empty()
            }
        } else {
            tier_modifier(tier)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_names_map_to_tiers() {
        assert_eq!(class_tier("safe"), 0);
        assert_eq!(class_tier("warn"), 1);
        assert_eq!(class_tier("crit"), 2);
        assert_eq!(class_tier("anything else"), 0);
    }

    #[test]
    fn red_is_only_for_crit() {
        assert_eq!(tier_color(0), GREEN);
        assert_eq!(tier_color(1), YELLOW);
        assert_eq!(tier_color(2), RED);
        assert_ne!(SAFE_BG, CRIT_BG);
        assert_ne!(WARN_BG, CRIT_BG);
    }

    #[test]
    fn tiers_stay_distinguishable_without_colour() {
        let mono = Theme::monochrome();
        let styles: Vec<Style> = (0..3).map(|t| mono.tier_chip(t)).collect();
        assert_ne!(styles[0], styles[1]);
        assert_ne!(styles[1], styles[2]);
        assert_ne!(styles[0], styles[2]);
        assert_eq!(styles[0].fg, Some(Color::Reset));
    }

    #[test]
    fn monochrome_theme_emits_no_colours() {
        let theme = Theme::monochrome();
        assert_eq!(theme.fg(RED), Color::Reset);
        assert_eq!(theme.bg(RED), None);
        let style = theme.paint(RED, CRIT_BG);
        assert_eq!(style.fg, Some(Color::Reset));
        assert_eq!(style.bg, None);
    }
}

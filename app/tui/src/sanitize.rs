//! Neutralise attacker-influenced text (argv, paths, subjects, messages) before it reaches a
//! terminal. A Rust port of `src/jevmon/sanitize.py`.
//!
//! Everything the daemon forwards is chosen by whoever generated the event: process arguments,
//! usernames, paths. This module strips ANSI/VT escape sequences entirely (so `ESC[2J`, an OSC
//! title set or a hyperlink can never reach the terminal), drops other C0/C1 control characters,
//! bidi overrides and zero-width/format characters, and bounds the result (head + tail, so padding
//! cannot push the interesting part out of view). The result is always plain text; no caller ever
//! interprets it as styling.

/// Cap for long fields (messages, contexts).
pub const SHOW_MAX: usize = 2_000;
/// Cap for any string that reaches the screen at all.
pub const DEFAULT_LIMIT: usize = 4_000;

/// Remove escapes and control/format characters, then cap at `limit` characters (head + tail).
/// Tabs/newlines collapse to spaces: this is the single-line form for subjects, messages, labels.
pub fn clean(text: &str, limit: usize) -> String {
    clean_with(text, limit, false)
}

/// Like [`clean`], but keeps `\n` as a line separator — for the model-facing context, whose
/// line structure is part of its meaning. Newlines never reach a terminal cell directly: the
/// renderer splits them into lines first.
pub fn clean_lines(text: &str, limit: usize) -> String {
    clean_with(text, limit, true)
}

fn clean_with(text: &str, limit: usize, keep_newlines: bool) -> String {
    if limit == 0 {
        return String::new();
    }
    // Bound the work on absurd inputs; keep both ends so a keyword cannot be padded out of view.
    let count = text.chars().count();
    let text = if count > 4 * limit {
        let head: String = text.chars().take(2 * limit).collect();
        let tail: String = text.chars().skip(count - 2 * limit).collect();
        format!("{head}{tail}")
    } else {
        text.to_string()
    };
    let text = strip_escapes(&text);
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if keep_newlines && c == '\n' {
            out.push('\n');
        } else if is_whitespace_control(c) {
            out.push(' ');
        } else if c.is_control() || is_format(c) {
            // C0/C1 controls (a leftover ESC included), bidi overrides, zero-widths, tags.
        } else {
            out.push(c);
        }
    }
    shorten(&out, limit)
}

/// Cap `text` at `limit` characters, keeping the head and tail around a marker.
pub fn shorten(text: &str, limit: usize) -> String {
    let count = text.chars().count();
    if count <= limit {
        return text.to_string();
    }
    let marker = format!(" [... {} chars ...] ", count - limit);
    if marker.chars().count() >= limit {
        return text.chars().take(limit).collect();
    }
    let keep = limit - marker.chars().count();
    let head = keep - keep / 4;
    let tail = keep - head;
    let head: String = text.chars().take(head).collect();
    let tail: String = text.chars().skip(count - tail).collect();
    format!("{head}{marker}{tail}")
}

/// Drop ANSI/VT escape sequences (CSI, OSC, DCS/SOS/PM/APC and two-character escapes), for both
/// the 7-bit `ESC` and the 8-bit C1 introducers. Anything malformed keeps the introducer, which
/// the control-character pass then removes: no ESC byte ever survives.
fn strip_escapes(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\u{1b}' => {
                i += 1;
                if i >= chars.len() {
                    break; // dangling ESC: dropped
                }
                match chars[i] {
                    '[' => i = skip_csi(&chars, i + 1),
                    ']' => i = skip_until_st(&chars, i + 1),
                    'P' | 'X' | '^' | '_' => i = skip_until_st(&chars, i + 1),
                    ' '..='/' => {
                        // ESC [ -/]* [0-~]: intermediates then one final byte.
                        while i < chars.len() && ('\u{20}'..='\u{2f}').contains(&chars[i]) {
                            i += 1;
                        }
                        i = (i + 1).min(chars.len());
                    }
                    _ => i += 1, // two-character escape (ESC 7, ESC =, ...)
                }
            }
            '\u{9b}' => i = skip_csi(&chars, i + 1), // 8-bit CSI
            '\u{9d}' => i = skip_until_st(&chars, i + 1), // 8-bit OSC
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// CSI: skip parameter/intermediate bytes and the final byte in `0x40..=0x7E`.
fn skip_csi(chars: &[char], mut i: usize) -> usize {
    while i < chars.len() && !('\u{40}'..='\u{7e}').contains(&chars[i]) {
        i += 1;
    }
    (i + 1).min(chars.len())
}

/// OSC/DCS/SOS/PM/APC: skip until BEL, ST (`ESC \` or C1 `ST`), or the end of the string.
fn skip_until_st(chars: &[char], mut i: usize) -> usize {
    while i < chars.len() {
        match chars[i] {
            '\u{7}' | '\u{9c}' => return i + 1,
            '\u{1b}' => {
                return if chars.get(i + 1) == Some(&'\\') {
                    i + 2
                } else {
                    i + 1
                };
            }
            _ => i += 1,
        }
    }
    i
}

/// Tabs/newlines and the Unicode line/paragraph separators collapse to a plain space.
fn is_whitespace_control(c: char) -> bool {
    matches!(
        c,
        '\t' | '\r' | '\n' | '\u{b}' | '\u{c}' | '\u{85}' | '\u{2028}' | '\u{2029}'
    )
}

/// Unicode format characters (category Cf): bidi controls and overrides, joiners, tag characters.
fn is_format(c: char) -> bool {
    matches!(
        c as u32,
        0x00ad
            | 0x0600..=0x0605
            | 0x061c
            | 0x06dd
            | 0x070f
            | 0x180e
            | 0x200b..=0x200f
            | 0x202a..=0x202e
            | 0x2060..=0x2064
            | 0x2066..=0x2069
            | 0x206a..=0x206f
            | 0xfe00..=0xfe0f
            | 0xfeff
            | 0xfff9..=0xfffb
            | 0x110bd
            | 0x110cd
            | 0x13430..=0x1343f
            | 0x1bca0..=0x1bca3
            | 0x1d173..=0x1d17a
            | 0xe0001
            | 0xe0020..=0xe007f
            | 0xe0100..=0xe01ef
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_csi_sequences() {
        assert_eq!(clean("\u{1b}[2Jok", 100), "ok");
        assert_eq!(clean("\u{1b}[31mred\u{1b}[0m", 100), "red");
        // An escape must not be able to split a keyword.
        assert_eq!(clean("Fai\u{1b}[0mled", 100), "Failed");
    }

    #[test]
    fn strips_osc_sequences() {
        assert_eq!(clean("\u{1b}]0;pwned title\u{7}ok", 100), "ok");
        assert_eq!(clean("\u{1b}]8;;http://evil\u{1b}\\ok", 100), "ok");
        assert_eq!(clean("\u{9d}0;title\u{7}ok", 100), "ok");
    }

    #[test]
    fn strips_dcs_and_two_char_escapes() {
        assert_eq!(clean("\u{1b}Pq\u{1b}\\ok", 100), "ok");
        assert_eq!(clean("a\u{1b}c", 100), "a"); // ESC c (RIS)
    }

    #[test]
    fn drops_lone_or_trailing_esc() {
        assert_eq!(clean("ab\u{1b}", 100), "ab");
        assert!(!clean("\u{1b}", 100).contains('\u{1b}'));
    }

    #[test]
    fn strips_8bit_c1_csi() {
        assert_eq!(clean("\u{9b}31mred\u{9b}0m", 100), "red");
    }

    #[test]
    fn drops_bidi_and_zero_width() {
        // RLO override, zero-width space, soft hyphen, ZWNBSP, variation selector.
        let dirty = "a\u{202e}b\u{200b}c\u{ad}d\u{feff}e\u{fe0f}f\u{2066}g\u{2069}";
        assert_eq!(clean(dirty, 100), "abcdefg");
    }

    #[test]
    fn whitespace_controls_become_spaces() {
        assert_eq!(clean("a\tb\nc", 100), "a b c");
        assert_eq!(clean("a\u{2028}b", 100), "a b");
    }

    #[test]
    fn bounds_long_input_keeping_both_ends() {
        let long = format!("HEAD{}", "x".repeat(10_000),);
        let out = clean(&format!("{long}TAIL"), 100);
        assert!(
            out.chars().count() <= 100,
            "got {} chars",
            out.chars().count()
        );
        assert!(out.contains("HEAD"));
        assert!(out.contains("TAIL"));
        assert!(out.contains("chars ..."));
    }

    #[test]
    fn truncates_wide_unicode_at_char_boundaries() {
        let wide = "日".repeat(500);
        let out = clean(&wide, 100);
        assert!(out.chars().count() <= 100);
        assert!(out.starts_with("日日"));
        assert!(
            out.chars()
                .all(|c| c == '日' || " .[]0123456789chars...".contains(c))
        );
    }

    #[test]
    fn torture_input_leaves_no_control_or_format_chars() {
        let dirty = "\u{1b}[31m\u{1b}]0;x\u{7}\u{9b}1m\u{1b}\u{7}\u{0}\u{1}\u{1f}\u{7f}\u{9f}\u{202e}\
                      \u{200f}\u{200b}\u{feff}\u{2066}ok\u{2069}\u{2028}end\u{1b}Pq\u{1b}\\";
        let out = clean(dirty, 100);
        assert!(out.contains("ok"));
        assert!(out.contains("end"));
        assert!(
            out.chars()
                .all(|c| !c.is_control() && !is_format(c) && !is_whitespace_control(c))
        );
        assert!(!out.contains('\u{1b}'));
    }

    #[test]
    fn shorten_is_idempotent_under_limit() {
        assert_eq!(shorten("short", 100), "short");
        let s = shorten(&"a".repeat(200), 50);
        assert_eq!(s.chars().count(), 50);
    }

    #[test]
    fn clean_collapses_newlines_but_clean_lines_keeps_structure() {
        assert_eq!(clean("a\nb", 100), "a b");
        assert_eq!(clean("a\r\nb", 100), "a  b"); // each control becomes a space (as in Python)
        assert_eq!(clean_lines("a\nb", 100), "a\nb");
        // ... while still stripping escapes and controls on both forms.
        assert_eq!(clean_lines("\x1b[2Ja\nb\u{202e}c", 100), "a\nbc");
        assert!(!clean_lines("a\u{9b}31m\nx", 100).contains('\u{9b}'));
        assert!(clean_lines("a\tb\n", 100).contains("a b\n"));
    }
}

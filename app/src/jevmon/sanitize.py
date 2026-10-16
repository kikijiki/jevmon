"""Neutralise attacker-influenced text (command lines, log lines, unit names) before it reaches
rules, the model or a terminal.

Process arguments, usernames, user agents and URLs are chosen by whoever is attacking us.
Everything downstream (rules, the model prompt, a UI, JSON output) gets the cleaned text:

- ANSI/VT escape sequences are dropped entirely (so `ESC[2J` or a title-set OSC never reaches
  the terminal, and cannot be used to split a keyword like `Fai<ESC>[0mled password`).
- Other control characters, C1 controls, format characters (zero width, bidi overrides) and
  lone surrogates are removed; whitespace controls and Unicode line separators become a space.
- Lines are capped: `SCAN_MAX` characters for rule matching (head + tail, so padding cannot push
  a keyword out of view), and `SHOW_MAX` for the model and for display.
"""

import re
import unicodedata

SCAN_MAX = 32_768  # what rules may see
SHOW_MAX = 2_000  # what the model and the UI may see
UNIT_MAX = 128

_ANSI = re.compile(
    r"""
    \x1b\[[0-?]*[ -/]*[@-~]          # CSI
    | \x9b[0-?]*[ -/]*[@-~]          # 8-bit CSI
    | \x1b\][^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)?  # OSC, BEL or ST terminated
    | \x9d[^\x07\x1b\x9c]*(?:\x07|\x9c)?
    | \x1b[PX^_][^\x1b\x9c]*(?:\x1b\\|\x9c)?        # DCS, SOS, PM, APC
    | \x1b[ -/]*[0-~]                # two-character escapes
    """,
    re.VERBOSE | re.DOTALL,
)
_SPACE = re.compile(r"[\t\r\n\v\f\u0085  ]")
_DROP = {"Cc", "Cf", "Cs"}


def clean(text: str, limit: int = SCAN_MAX) -> str:
    """Remove escapes and control/format characters, then cap at `limit` (head + tail)."""
    if len(text) > 4 * limit:  # bound the work on absurd inputs; keep both ends
        text = text[: 2 * limit] + text[-2 * limit :]
    text = _ANSI.sub("", text)
    text = _SPACE.sub(" ", text)
    if not text.isascii() or not text.isprintable():
        text = "".join(ch for ch in text if unicodedata.category(ch) not in _DROP)
    return shorten(text, limit)


def shorten(text: str, limit: int = SHOW_MAX) -> str:
    """Cap `text` at `limit` characters, keeping the head and tail around a marker."""
    if len(text) <= limit:
        return text
    marker = f" [...{len(text) - limit + 5} chars...] "
    keep = max(limit - len(marker), 0)
    head = keep - keep // 4
    return text[:head] + marker + text[len(text) - (keep - head) :]

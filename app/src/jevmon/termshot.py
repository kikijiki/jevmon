"""Rasterise a captured terminal screen (SGR-styled text) to a deterministic PNG.

`tmux capture-pane -p -e` returns one line per terminal row, styled only with SGR sequences. This
module parses those runs into (column, character, style) cells and paints them with Pillow: one
background rectangle per run that carries a non-default colour, then one glyph per cell at
`column * cell_width`, so a monospace font lands exactly on the terminal's grid.

The font is pinned (`JEVMON_MONO_FONT`, else a DejaVu Sans Mono found on the system) and the PNG
is written with a fixed compression level, so re-running the capture on the same input produces
the same bytes. Nothing here reads the machine: only the capture text goes in.
"""

import io
import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

# The TUI palette's own background/foreground (app/tui/src/theme.rs): `capture-pane -e` emits `39`/`49`
# for cells that carry no colour of their own, so those cells fall back to these.
DEFAULT_BG = "#14161c"
DEFAULT_FG = "#c8ccd6"

_SGR = re.compile(r"\x1b\[([0-9;]*)m")
_ESCAPE = re.compile(
    r"\x1b(?:\[[0-9;:?]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[()*+][0-9A-Za-z@#%]|[@-Z\\^_])"
)

_ANSI = (
    "#000000",
    "#cd3131",
    "#0dbc79",
    "#e5e510",
    "#2472c8",
    "#bc3fbc",
    "#11a8cd",
    "#e5e5e5",
)
_BRIGHT = (
    "#666666",
    "#f14c4c",
    "#23d18b",
    "#f5f543",
    "#3b8eea",
    "#d670d6",
    "#29b8db",
    "#ffffff",
)
_CUBE = (0, 95, 135, 175, 215, 255)

# Column ranges a terminal draws two cells wide (East Asian Wide/Fullwidth and the emoji planes).
_WIDE = (
    (0x1100, 0x115F),
    (0x2E80, 0x303E),
    (0x3041, 0x33FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xA000, 0xA4CF),
    (0xAC00, 0xD7A3),
    (0xF900, 0xFAFF),
    (0xFE10, 0xFE19),
    (0xFE30, 0xFE6F),
    (0xFF00, 0xFF60),
    (0xFFE0, 0xFFE6),
    (0x1F300, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x20000, 0x3FFFD),
)

Cell = tuple[int, str, "Style"]


@dataclass(frozen=True, slots=True)
class Style:
    """One cell's SGR state. A `None` colour means "the terminal default"."""

    fg: str | None = None
    bg: str | None = None
    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False
    reverse: bool = False


def char_width(char: str) -> int:
    """Terminal cells one character occupies: 0 for combining/control, 2 for wide, else 1."""
    if unicodedata.category(char) in ("Cc", "Cf", "Mn", "Me"):
        return 0
    code = ord(char)
    for low, high in _WIDE:
        if low <= code <= high:
            return 2
    return 1


def _colour256(index: int) -> str:
    index &= 0xFF
    if index < 16:
        return (_ANSI + _BRIGHT)[index]
    if index >= 232:
        level = 8 + 10 * (index - 232)
        return f"#{level:02x}{level:02x}{level:02x}"
    index -= 16
    red, green, blue = _CUBE[index // 36], _CUBE[(index // 6) % 6], _CUBE[index % 6]
    return f"#{red:02x}{green:02x}{blue:02x}"


def _extended(values: list[int], start: int) -> tuple[str | None, int]:
    """Parse `5;n` / `2;r;g;b` after a 38/48; returns the colour and the values it consumed."""
    if start >= len(values):
        return None, 0
    if values[start] == 5 and start + 1 < len(values):
        return _colour256(values[start + 1]), 2
    if values[start] == 2 and start + 3 < len(values):
        red, green, blue = values[start + 1] & 255, values[start + 2] & 255, values[start + 3] & 255
        return f"#{red:02x}{green:02x}{blue:02x}", 4
    return None, 1


def apply_sgr(style: Style, params: str) -> Style:
    """Fold one `CSI ... m` parameter list into a new `Style`."""
    if params == "":
        return Style()
    values = [int(part) if part.isdigit() else 0 for part in params.split(";")]
    state = style
    index = 0
    while index < len(values):
        value = values[index]
        if value == 0:
            state = Style()
        elif value == 1:
            state = replace(state, bold=True)
        elif value == 2:
            state = replace(state, dim=True)
        elif value == 3:
            state = replace(state, italic=True)
        elif value == 4:
            state = replace(state, underline=True)
        elif value == 7:
            state = replace(state, reverse=True)
        elif value == 22:
            state = replace(state, bold=False, dim=False)
        elif value == 23:
            state = replace(state, italic=False)
        elif value == 24:
            state = replace(state, underline=False)
        elif value == 27:
            state = replace(state, reverse=False)
        elif 30 <= value <= 37:
            state = replace(state, fg=_ANSI[value - 30])
        elif value == 39:
            state = replace(state, fg=None)
        elif 40 <= value <= 47:
            state = replace(state, bg=_ANSI[value - 40])
        elif value == 49:
            state = replace(state, bg=None)
        elif 90 <= value <= 97:
            state = replace(state, fg=_BRIGHT[value - 90])
        elif 100 <= value <= 107:
            state = replace(state, bg=_BRIGHT[value - 100])
        elif value in (38, 48):
            colour, consumed = _extended(values, index + 1)
            index += consumed
            if colour is not None:
                state = replace(state, fg=colour) if value == 38 else replace(state, bg=colour)
        index += 1
    return state


def parse_capture(capture: str, cols: int, rows: int) -> list[list[Cell]]:
    """Split a `capture-pane -e` dump into rows of (column, character, style)."""
    screen: list[list[Cell]] = []
    for line in capture.splitlines()[:rows]:
        cells: list[Cell] = []
        style = Style()
        column = 0
        index = 0
        while index < len(line) and column < cols:
            char = line[index]
            if char == "\x1b":
                sgr = _SGR.match(line, index)
                if sgr is not None:
                    style = apply_sgr(style, sgr.group(1))
                    index = sgr.end()
                    continue
                escape = _ESCAPE.match(line, index)
                index = escape.end() if escape is not None else index + 1
                continue
            index += 1
            width = char_width(char)
            if width == 0:
                continue
            if column + width > cols:
                break
            cells.append((column, char, style))
            column += width
        screen.append(cells)
    while len(screen) < rows:
        screen.append([])
    return screen


def _rgb(colour: str) -> tuple[int, int, int]:
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def _resolve(style: Style, default_fg: str, default_bg: str) -> tuple[str, str]:
    foreground = style.fg or default_fg
    background = style.bg or default_bg
    if style.reverse:
        return background, foreground
    return foreground, background


def _dim(colour: str, background: str) -> tuple[int, int, int]:
    """Blend a dimmed foreground toward the background - a fixed integer mix, so it is stable."""
    fg, bg = _rgb(colour), _rgb(background)
    return ((fg[0] * 2 + bg[0]) // 3, (fg[1] * 2 + bg[1]) // 3, (fg[2] * 2 + bg[2]) // 3)


def find_mono_font() -> Path:
    """The pinned monospace font: `$JEVMON_MONO_FONT`, else DejaVu Sans Mono on this machine."""
    candidates: list[Path] = []
    override = os.environ.get("JEVMON_MONO_FONT")
    if override:
        candidates.append(Path(override))
    try:  # fontconfig knows about fonts this list would miss
        done = subprocess.run(
            ["fc-match", "-f", "%{file}", "DejaVu Sans Mono"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if done.returncode == 0 and done.stdout.strip().endswith((".ttf", ".otf")):
            candidates.append(Path(done.stdout.strip()))
    except (FileNotFoundError, subprocess.SubprocessError):  # pragma: no cover - no fontconfig
        pass
    candidates += [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path.home() / ".nix-profile/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for prefix in ("/run/current-system/sw", str(Path.home() / ".nix-profile")):
        candidates.extend(sorted(Path(prefix).glob("share/fonts/**/DejaVuSansMono.ttf")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "screenshots: no DejaVu Sans Mono found - set JEVMON_MONO_FONT to a .ttf path "
        "(the flake dev shell sets it)"
    )


def _sibling(font: Path, suffix: str) -> Path | None:
    """`DejaVuSansMono.ttf` -> `DejaVuSansMono-Bold.ttf` next to it, when it exists.

    Only the same family qualifies: a glob would also find `DejaVuSans-Bold.ttf`, a proportional
    face that does not sit on the cell grid.
    """
    if suffix in font.stem:
        return font
    candidate = font.with_name(f"{font.stem}-{suffix}{font.suffix}")
    return candidate if candidate.exists() else None


def ansi_to_png(
    capture: str,
    *,
    cols: int,
    rows: int,
    cell_w: int = 9,
    cell_h: int = 18,
    font_size: int = 15,
    default_fg: str = DEFAULT_FG,
    default_bg: str = DEFAULT_BG,
    font: Path | None = None,
) -> bytes:
    """Rasterise a `capture-pane -e` dump to PNG bytes covering exactly `cols` x `rows` cells."""
    from PIL import Image, ImageDraw, ImageFont

    cols, rows = int(cols), int(rows)
    cell_w, cell_h, font_size = int(cell_w), int(cell_h), int(font_size)
    font_path = font or find_mono_font()
    regular = ImageFont.truetype(str(font_path), font_size)
    bold_path = _sibling(font_path, "Bold")
    bold = ImageFont.truetype(str(bold_path), font_size) if bold_path else regular
    oblique_path = _sibling(font_path, "Oblique") or _sibling(font_path, "Italic")
    italic = ImageFont.truetype(str(oblique_path), font_size) if oblique_path else regular
    ascent, _descent = regular.getmetrics()

    screen = parse_capture(capture, cols, rows)
    image = Image.new("RGB", (cols * cell_w, rows * cell_h), _rgb(default_bg))
    draw = ImageDraw.Draw(image)

    for row, cells in enumerate(screen):
        top = row * cell_h
        baseline = top + ascent
        for column, char, style in cells:
            foreground, background = _resolve(style, default_fg, default_bg)
            x = column * cell_w
            if background != default_bg:
                draw.rectangle(
                    [x, top, x + cell_w * max(char_width(char), 1) - 1, top + cell_h - 1],
                    fill=_rgb(background),
                )
            if char == " ":
                continue
            colour = _dim(foreground, background) if style.dim else _rgb(foreground)
            face = bold if style.bold else italic if style.italic else regular
            draw.text((x, baseline), char, font=face, fill=colour, anchor="ls")
            if style.underline:
                draw.line(
                    [x, baseline + 2, x + cell_w - 1, baseline + 2],
                    fill=colour,
                    width=1,
                )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


__all__ = [
    "DEFAULT_BG",
    "DEFAULT_FG",
    "Style",
    "ansi_to_png",
    "apply_sgr",
    "char_width",
    "find_mono_font",
    "parse_capture",
]

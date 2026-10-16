"""`jevmon.termshot`: a `capture-pane -e` dump becomes a deterministic, cell-aligned PNG."""

import io
import struct

from PIL import Image

from jevmon.termshot import (
    DEFAULT_BG,
    DEFAULT_FG,
    Style,
    ansi_to_png,
    apply_sgr,
    char_width,
    find_mono_font,
    parse_capture,
)

COLS, ROWS = 40, 4


def size(png: bytes) -> tuple[int, int]:
    assert png.startswith(b"\x89PNG\r\n\x1a\n"), "not a PNG"
    return struct.unpack(">II", png[16:24])


def test_sgr_parameters_fold_into_a_style():
    assert apply_sgr(Style(), "") == Style()
    assert apply_sgr(Style(), "0") == Style()
    assert apply_sgr(Style(), "1;31").bold and apply_sgr(Style(), "1;31").fg == "#cd3131"
    assert apply_sgr(Style(fg="#cd3131"), "39").fg is None
    assert apply_sgr(Style(), "48;2;20;30;40").bg == "#141e28"
    assert apply_sgr(Style(), "38;5;196").fg == "#ff0000"
    cleared = apply_sgr(apply_sgr(Style(), "1;7"), "22;27")
    assert not cleared.bold and not cleared.reverse


def test_capture_is_split_into_columns_and_styles():
    screen = parse_capture("\x1b[31mred\x1b[39m plain\nsecond", COLS, ROWS)
    assert len(screen) == ROWS  # padded to the pane height
    assert [(c, ch) for c, ch, _ in screen[0][:5]] == [
        (0, "r"),
        (1, "e"),
        (2, "d"),
        (3, " "),
        (4, "p"),
    ]
    assert "".join(ch for _c, ch, _s in screen[0]) == "red plain"
    assert screen[0][0][2].fg == "#cd3131" and screen[0][3][2].fg is None
    assert [ch for _c, ch, _s in screen[1]] == list("second")
    assert screen[2] == []


def test_unknown_escape_sequences_are_dropped_not_rendered():
    screen = parse_capture("\x1b[2Ja\x1b]0;title\x07b\x1b(Bc", COLS, 1)
    assert "".join(ch for _c, ch, _s in screen[0]) == "abc"


def test_wide_and_combining_characters_claim_the_right_columns():
    assert char_width("a") == 1 and char_width("日") == 2 and char_width("́") == 0
    screen = parse_capture("a日b", COLS, 1)
    assert [c for c, _ch, _s in screen[0]] == [0, 1, 3]


def test_the_monospace_font_is_found():
    font = find_mono_font()
    assert font.is_file() and font.suffix in (".ttf", ".otf")


def test_png_is_deterministic_and_sized_to_the_terminal():
    capture = "\x1b[1;34mbold blue\x1b[0m plain\x1b[7m reversed\x1b[0m"

    def render(text: str) -> bytes:
        return ansi_to_png(text, cols=COLS, rows=ROWS, cell_w=9, cell_h=18, font_size=15)

    first = render(capture)
    second = render(capture)
    assert first == second, "the same capture must rasterise to the same bytes"
    assert size(first) == (COLS * 9, ROWS * 18)
    assert render("other") != first, "different text, different pixels"


def test_every_colour_class_actually_paints_different_pixels():
    def render(text: str) -> bytes:
        return ansi_to_png(text, cols=12, rows=2, cell_w=9, cell_h=18, font_size=15)

    plain = render("X")
    coloured = render("\x1b[31mX\x1b[0m")
    reversed_video = render("\x1b[7mX\x1b[0m")
    bold = render("\x1b[1mX\x1b[0m")
    on_bg = render("\x1b[44m \x1b[0m")
    assert len({plain, coloured, reversed_video, bold, on_bg}) == 5
    assert DEFAULT_BG != DEFAULT_FG, "the palette itself must not be degenerate"


def test_the_default_background_fills_the_whole_frame():
    png = ansi_to_png("", cols=COLS, rows=ROWS, cell_w=9, cell_h=18, font_size=15)
    image = Image.open(io.BytesIO(png))
    assert image.size == (COLS * 9, ROWS * 18)
    assert image.convert("RGB").getpixel((0, 0)) == (0x14, 0x16, 0x1C)

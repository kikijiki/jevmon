from jevmon.sanitize import clean, shorten


def test_ansi_and_controls_are_removed_and_keywords_rejoined():
    assert clean("Fai\x1b[0mled password\x00") == "Failed password"
    assert "\x1b" not in clean("\x1b]0;title\x07rm -rf /")


def test_bidi_and_zero_width_are_removed():
    assert clean("pass" + chr(0x202E) + "w" + chr(0x200B) + "ord") == "password"


def test_shorten_keeps_head_and_tail():
    s = shorten("A" * 5000 + "TAIL", 200)
    assert len(s) <= 200 and s.startswith("A") and s.endswith("TAIL")

"""Masking helper: paths/numbers/hashes/uuids become placeholders, other tokens survive."""

from jevmon.templates import mask_token, masked_template


def test_paths_numbers_hashes_uuids_become_placeholders():
    assert mask_token("/usr/bin/ls") == "<PATH>"
    assert mask_token("/home/u/.cache/evil") == "<PATH>"
    assert mask_token("1234") == "<NUM>"
    assert mask_token("-3.5") == "<NUM>"
    assert mask_token("550e8400-e29b-41d4-a716-446655440000") == "<UUID>"
    assert mask_token("a" * 40) == "<HASH>"
    assert mask_token("Ab3+Zx9+Qq7+Ww2+Ee5+Rr8+Ty1+Uu4+Q1") == "<HASH>"
    assert mask_token("hello") == "hello"
    assert mask_token("--verbose") == "--verbose"


def test_masked_template_collapses_shapes():
    a = masked_template(["/usr/bin/python3", "/home/a/one.py", "42"])
    b = masked_template(["/usr/bin/python3", "/home/b/two.py", "7"])
    assert a == b == "<PATH> <PATH> <NUM>"
    assert masked_template(["/bin/true"]) == "<PATH>"


def test_template_is_bounded_and_sanitized():
    template = masked_template(["\x1b[31mred", "x" * 500])
    assert "\x1b" not in template and len(template) <= 300
    long = masked_template([f"arg{i}" for i in range(100)])
    assert long.endswith("...")

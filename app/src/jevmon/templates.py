"""Mask an argv into a template so that frequent shapes collapse and rare shapes stand out.

Paths, numbers, hashes and UUIDs become placeholders; other tokens stay literal (sanitized and
bounded). This is shape, not a signature list: it never decides anything, it only lets the learned
counts and the model see structure. Input is expected to be already redacted.
"""

import re

from jevmon.sanitize import clean, shorten

TOKEN_MAX = 64
TEMPLATE_MAX = 300

_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HEX = re.compile(r"^[0-9a-fA-F]{32,}$")
_B64 = re.compile(r"^[A-Za-z0-9+]{32,}={0,2}$")
_NUMBER = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?$")


def mask_token(token: str) -> str:
    """Replace a path/number/hash/uuid token with a placeholder; otherwise sanitize and bound."""
    if _UUID.fullmatch(token):
        return "<UUID>"
    if _HEX.fullmatch(token):
        return "<HASH>"
    if _B64.fullmatch(token) and any(ch in "+=" for ch in token):
        return "<HASH>"
    if _NUMBER.fullmatch(token):
        return "<NUM>"
    if "/" in token and not any(ch.isspace() for ch in token):
        return "<PATH>"
    return shorten(clean(token, TOKEN_MAX * 4), TOKEN_MAX)


def masked_template(argv: list[str] | tuple[str, ...], max_tokens: int = 24) -> str:
    """Join masked tokens into one bounded template string."""
    tokens = [mask_token(token) for token in list(argv)[:max_tokens]]
    if len(argv) > max_tokens:
        tokens.append("...")
    return shorten(" ".join(tokens), TEMPLATE_MAX)

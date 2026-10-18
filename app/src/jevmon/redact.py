"""Redact secrets from attacker-influenced process arguments and text, before storing or emitting.

Command lines routinely carry credentials: `mysql -pmypass`, `curl -u user:pw`, environment-style
`TOKEN=...`, `Authorization: Bearer ...`, cloud keys and long encoded payloads. Everything here
runs *before* an event is built, stored or printed.

Two entry points:

- `redact_arg` / `redact_argv` / `redact_argv_text`: per-argument redaction and bounds for argv.
- `redact_text`: the same secret rules for a free string (already-sanitized by `sanitize.clean`).

Rules (applied after `sanitize.clean`, which strips ANSI/control/format characters):

- `--password=...`/`--token ...`/`--secret`/`--api-key` flags (both attached and spaced forms);
- the `-p<pw>` short form (only when the value is not purely numeric, so `-p8080` survives);
- `KEY=value` where the key name looks secret-like;
- `Authorization: Bearer ...`;
- URL userinfo (`scheme://user:pass@host`);
- AWS-style access key ids, 32+ character hex blobs and 32+ character base64 blobs.

Per-argument and whole-argv lengths are bounded as well, so a hostile command line cannot push a
keyword out of view or blow up storage. Redaction is best-effort: it lowers exposure, it is not a
guarantee.
"""

import re

from jevmon.sanitize import clean, shorten

ARG_MAX = 512
ARGV_MAX = 4096
MAX_ARGS = 64

_SECRET_WORD = r"(?:password|passwd|token|secret|api[-_]?key|credential|auth)"
_FLAG = re.compile(
    rf"^(?P<flag>--?[A-Za-z0-9_.-]*{_SECRET_WORD}[A-Za-z0-9_.-]*)=(?P<val>.*)$", re.IGNORECASE
)
_FLAG_BARE = re.compile(rf"^--?[A-Za-z0-9_.-]*{_SECRET_WORD}[A-Za-z0-9_.-]*$", re.IGNORECASE)
_SHORT_PW = re.compile(r"^-p(?P<val>.+)$")

_SECRET_WORDS = {
    "password",
    "passwd",
    "token",
    "secret",
    "key",
    "apikey",
    "credential",
    "credentials",
    "auth",
    "authorization",
    "bearer",
}
_KEY_VALUE = re.compile(r"(?P<key>[A-Za-z0-9_.-]{1,64})=(?P<val>[^\s'\"]+)")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_BEARER = re.compile(r"(?i)(Authorization:\s*Bearer)\s+\S+")
_URL_USERINFO = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/\s:@]+(?::[^/\s@]+)?@")
_AWS_KEY_ID = re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA|AIPA|ANPA|ANVA|AGPA|A3T[A-Z0-9])[A-Z0-9]{16}\b")
_HEX_BLOB = re.compile(r"\b[0-9a-fA-F]{32,}\b")
# No `/` in the class: it would eat filesystem paths. Requires padding/`+` or a mixed
# upper/lower/digit token, so long lowercase path components survive.
_B64_BLOB = re.compile(r"\b[A-Za-z0-9+]{32,}={0,2}\b")

_MASK = "***"


def _secret_key(key: str) -> bool:
    """True if a `KEY=value` key names a secret: split on `_`/`-`/`.` and camelCase boundaries."""
    parts = re.split(r"[_.-]+", key)
    for part in parts:
        if part.lower() in _SECRET_WORDS:
            return True
    return any(w.lower() in _SECRET_WORDS for w in _CAMEL.findall(key))


def _mask_kv(m: re.Match[str]) -> str:
    key = m.group("key")
    return f"{key}={_MASK}" if _secret_key(key) else m.group(0)


def _redact_blobs(text: str) -> str:
    """Mask long encoded payloads. Base64 needs a digit or `+/=` to avoid eating long words."""
    text = _AWS_KEY_ID.sub(_MASK, text)
    text = _HEX_BLOB.sub(_MASK, text)

    def b64(m: re.Match[str]) -> str:
        token = m.group(0)
        if any(ch in "+=" for ch in token):
            return _MASK
        mixed = (
            any(ch.isdigit() for ch in token)
            and any(ch.isupper() for ch in token)
            and any(ch.islower() for ch in token)
        )
        return _MASK if mixed else token

    return _B64_BLOB.sub(b64, text)


def redact_text(text: str) -> str:
    """Sanitize then redact secrets in free text."""
    text = clean(text, ARGV_MAX * 4)
    text = _BEARER.sub(r"\1 " + _MASK, text)
    text = _URL_USERINFO.sub(r"\1" + _MASK + "@", text)
    text = _KEY_VALUE.sub(_mask_kv, text)
    text = _redact_blobs(text)
    return text


def redact_arg(arg: str) -> str:
    """Redact one argv element; sanitize and cap its length."""
    arg = clean(arg, ARG_MAX * 4)
    short = _SHORT_PW.match(arg)
    if short and not short.group("val").isdigit():
        return "-p" + _MASK
    arg = _FLAG.sub(lambda m: f"{m.group('flag')}={_MASK}", arg)
    arg = _BEARER.sub(r"\1 " + _MASK, arg)
    arg = _URL_USERINFO.sub(r"\1" + _MASK + "@", arg)
    arg = _KEY_VALUE.sub(_mask_kv, arg)
    arg = _redact_blobs(arg)
    return shorten(arg, ARG_MAX)


def redact_argv(
    argv: list[str], max_args: int = MAX_ARGS, total: int = ARGV_MAX, mask: bool = True
) -> list[str]:
    """Redact and bound a whole argv. A bare secret flag also masks the argument that follows it.

    `mask=False` only sanitizes (control characters) and bounds lengths: nothing is replaced by
    `***`. That is the model-facing form; it must never be stored or emitted.
    """
    out: list[str] = []
    i = 0
    while i < len(argv) and len(out) < max_args:
        arg = argv[i]
        if not mask:
            out.append(shorten(clean(arg, ARG_MAX * 4), ARG_MAX))
            i += 1
            continue
        if _FLAG_BARE.match(arg) and i + 1 < len(argv):
            out.append(shorten(clean(arg, ARG_MAX * 4), ARG_MAX))
            out.append(_MASK)
            i += 2
            continue
        out.append(redact_arg(arg))
        i += 1
    if i < len(argv):
        out.append("[...]")
    used = 0
    trimmed: list[str] = []
    for arg in out:
        if used + len(arg) + 1 > total and trimmed:
            trimmed.append("[...]")
            break
        trimmed.append(arg)
        used += len(arg) + 1
    return trimmed


def redact_argv_text(
    argv: list[str], max_args: int = MAX_ARGS, total: int = ARGV_MAX, mask: bool = True
) -> str:
    """Redact an argv and join it into one bounded, space-separated string for an event field."""
    return shorten(" ".join(redact_argv(argv, max_args, total, mask)), total)

"""E18: query TypeSafe's hosted Jev (OpenRouter decisions API) on lab contexts. Stdlib only.

    OPENROUTER_API_KEY=... python scripts/jev_probe.py run --rows rows.jsonl --out out.json \
        --variants A_flat_choice B_lines_choice C_flat_binary [--key-file PATH]
    python scripts/jev_probe.py show --rows rows.jsonl --variant E_json_choice   # print requests

Rows file: JSONL with `label`, `family`, `set` and the context text in `context` (and further text
fields such as `context_c` for a variant that reads another rendering). The key is read from
OPENROUTER_API_KEY or --key-file, never from this file. Everything sent is synthetic lab data
(scenarios run in a rootless podman lab); do not point this at real process launches.

Responses are cached one file per request under cache/jev/ (gitignored), keyed by the sha256 of the
request body, so reruns are free and a changed variant or model name misses the cache.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
CACHE_DIR = Path("cache/jev")
WORKERS = 8
RETRY_CODES = (408, 429, 500, 502, 503, 504, 524, 529)
CLASSES = ("safe", "warn", "crit")

# The category descriptions are the ones the local zero-shot Laya judge uses.
CRITERIA = {
    "safe": (
        "expected everyday activity here: shells, builds, package installs, scheduled jobs, ssh "
        "sessions and other development tooling; no action needed"
    ),
    "warn": (
        "out of the ordinary for this workstation but plausibly harmless: a one-off script, the "
        "first run of a newly installed tool, a debugger; worth a glance"
    ),
    "crit": (
        "looks hostile from the launch alone: downloading and running remote code, encoded or "
        "obfuscated commands, execution from a temporary, world-writable or hidden location, a "
        "shell spawned by an unexpected parent"
    ),
}
Q_CHOICE = {
    "verdict": {
        "type": "choice",
        "instructions": "Which of these categories best describes this process launch?",
        "criteria": CRITERIA,
    }
}
Q_BINARY = {
    "safe_to_run": {
        "type": "noul",
        "instructions": "Is this process launch expected everyday activity that needs no attention?",
        "criteria": {
            "true": "Normal development, build, package, scheduled or ssh activity.",
            "false": "Unusual or hostile.",
        },
    },
    "needs_review": {
        "type": "noul",
        "instructions": "Is this process launch out of the ordinary and worth a human glance?",
        "criteria": {"true": "Unusual for a workstation, plausibly harmless.", "false": "Routine."},
    },
    "hostile": {
        "type": "noul",
        "instructions": "Does this process launch look hostile from the launch alone?",
        "criteria": {
            "true": (
                "Fetch-and-run, encoded or obfuscated commands, execution from temp or hidden "
                "locations, unexpected parent shell."
            ),
            "false": "No sign of hostility.",
        },
    },
}
QUESTIONS = {"choice": Q_CHOICE, "binary": Q_BINARY}

_LABEL = re.compile(r"\s+(?=[A-Za-z_]+(?:\([a-z]+\))?: )")
_SEGMENT = re.compile(r"(?:^|\s)(exe|basename|argv\(template\)|argv\(redacted\)|argv): ")
_JSON_KEYS = {
    "exe": "exe",
    "basename": "basename",
    "argv(template)": "argv_template",
    "argv(redacted)": "argv",
    "argv": "argv",
}


def layout_flat(ctx: str) -> str:
    return ctx


def layout_lines(ctx: str) -> str:
    """One `label: value` segment per line (every ` word: ` boundary starts a line)."""
    return _LABEL.sub("\n", ctx).strip()


def segments(ctx: str) -> dict:
    """Split a rendered context into header, exe, basename, argv_template and argv.

    Each label is taken at its first occurrence, in order, so a `exe: ` inside an argument stays in
    the argv value. Segments the context does not carry are absent.
    """
    found = []
    pos = 0
    want = ["exe", "basename", "argv(template)", "argv(redacted)", "argv"]
    for m in _SEGMENT.finditer(ctx):
        label = m.group(1)
        if m.start() < pos or label not in want:
            continue
        want = want[want.index(label) + 1 :]
        if label == "argv(redacted)":
            want = []
        found.append((label, m.start(), m.end()))
        pos = m.end()
    out: dict = {}
    head = ctx[: found[0][1]].strip() if found else ctx.strip()
    if head:
        out["header"] = head
    for i, (label, _s, e) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(ctx)
        out[_JSON_KEYS[label]] = ctx[e:end].strip()
    return out


def layout_json(ctx: str) -> dict:
    return segments(ctx)


LAYOUTS = {"flat": layout_flat, "lines": layout_lines, "json": layout_json}

# name -> (text field of the row, layout, question set)
VARIANTS = {
    "A_flat_choice": ("context", "flat", "choice"),
    "B_lines_choice": ("context", "lines", "choice"),
    "C_flat_binary": ("context", "flat", "binary"),
    "D_raw_flat_choice": ("context_c", "flat", "choice"),
    "E_json_choice": ("context", "json", "choice"),
}


def build_request(row: dict, variant: str, model: str = MODEL) -> dict:
    field, layout, questions = VARIANTS[variant]
    return {
        "model": model,
        "state": LAYOUTS[layout](row[field]),
        "questions": QUESTIONS[questions],
    }


def request_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_key(key_file: str | None) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and key_file:
        key = Path(key_file).read_text().strip()
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY or pass --key-file")
    return key


def post(body: dict, key: str, tries: int = 5, timeout: float = 90.0) -> dict:
    """POST one decision request; {'response', 'latency_s'} or {'error', ...} after retries."""
    data = json.dumps(body).encode()
    err = "unknown"
    for attempt in range(tries):
        req = urllib.request.Request(
            URL,
            data=data,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return {
                    "response": json.loads(r.read()),
                    "latency_s": time.perf_counter() - started,
                }
        except urllib.error.HTTPError as e:
            err = f"http {e.code}: {e.read().decode(errors='replace')[:200]}"
            if e.code not in RETRY_CODES:
                break
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            err = f"{type(e).__name__}: {e}"
        time.sleep(2.0 * (attempt + 1))
    return {"error": err}


def cached_call(body: dict, key: str, cache_dir: Path | None = None) -> dict:
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{request_key(body)}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = post(body, key)
    if "error" not in out:
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(out))
        tmp.replace(path)
    return out


def run_all(
    rows: list[dict],
    variants: list[str],
    key: str,
    model: str = MODEL,
    workers: int = WORKERS,
    cache_dir: Path | None = None,
) -> dict:
    jobs = [(i, v) for i in range(len(rows)) for v in variants]

    def one(job):
        i, v = job
        return f"{i}|{v}", cached_call(build_request(rows[i], v, model), key, cache_dir)

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, jobs))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "show"))
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--variants", nargs="+", default=["A_flat_choice", "B_lines_choice"])
    ap.add_argument("--variant", default="A_flat_choice")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only")
    ap.add_argument("--key-file")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    rows = read_rows(args.rows)
    if args.limit:
        rows = rows[: args.limit]
    if args.mode == "show":
        for r in rows[:3]:
            print(json.dumps(build_request(r, args.variant, args.model), indent=1))
        return
    if not args.out:
        raise SystemExit("--out required")
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known {sorted(VARIANTS)}")
    res = run_all(
        rows, args.variants, load_key(args.key_file), args.model, args.workers, args.cache
    )
    args.out.write_text(json.dumps({"rows": rows, "res": res, "model": args.model}))
    errs = sum(1 for r in res.values() if "error" in r)
    cost = sum(r.get("response", {}).get("usage", {}).get("cost", 0) for r in res.values())
    lat = sorted(r["latency_s"] for r in res.values() if "latency_s" in r)
    print(f"{len(rows)} rows, {len(res)} calls, {errs} errors, cost ${cost:.4f}", file=sys.stderr)
    if lat:
        print(
            f"latency median (as recorded when fetched) {lat[len(lat) // 2]:.2f}s", file=sys.stderr
        )


if __name__ == "__main__":
    main()

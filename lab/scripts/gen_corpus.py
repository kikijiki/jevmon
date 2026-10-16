"""E19: generate the synthetic process-launch corpus.

    uv run --project lab python scripts/gen_corpus.py generate \
        --taxonomy ../lab/corpus_taxonomy.toml --out ../lab-data-gen-<date> \
        --per-family 20 --safe-model qwen3.8 --crit-model qwen3.8-uncensored

    uv run --project lab python scripts/gen_corpus.py qa --out ../lab-data-gen-<date> --model qwen3.8
    uv run --project lab python scripts/gen_corpus.py render --out ../lab-data-gen-<date>
    uv run --project lab python scripts/gen_corpus.py card --out ../lab-data-gen-<date>

Resumable: `generate` writes one `raw/<family>.jsonl` per family and skips families whose file
already has >= `--per-family` kept rows, so a crashed or Ctrl-C'd run restarts where it left off
(`--force` regenerates). Concurrency is 1 (sequential families) by default - the iGPU is shared
with the experiments agent; `--workers N` raises it, `--sleep S` adds a pause between calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gen_corpus_lib import Family, LLMClient, coerce_row, generate_family_rows, load_taxonomy
from gen_corpus_validate import apply_qa, attach_facts, dedupe_rows, render_rows, validate_rows
from jevmon_lab.manifest import split_of

RAW_DIR = "raw"
PROMPT_HASH_FILE = "prompt_hash.txt"


def prompt_hash() -> str:
    from gen_corpus_lib import GEN_SYSTEM, build_prompt

    sample = build_prompt(Family("x", "safe", "brief"), 5)
    return hashlib.sha256((GEN_SYSTEM + sample).encode()).hexdigest()[:16]


def model_for(family: Family, safe_model: str, crit_model: str) -> str:
    # qwen3.8-uncensored writes realistic hostile command lines without refusing; qwen3.8 covers
    # safe/warn variety (it is also fine on crit text, just more likely to refuse or soften it).
    return crit_model if family.label == "crit" else safe_model


def cmd_generate(args: argparse.Namespace) -> None:
    families = load_taxonomy(args.taxonomy)
    if args.only:
        wanted = set(args.only)
        families = [f for f in families if f.name in wanted]
    random.Random(19).shuffle(families)  # a partial run is class-balanced, not crit-first
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        families = [f for k, f in enumerate(families) if k % n == i]
    out = Path(args.out)
    (out / RAW_DIR).mkdir(parents=True, exist_ok=True)
    (out / PROMPT_HASH_FILE).write_text(prompt_hash())
    client = LLMClient()
    counts = Counter()
    for family in families:
        path = out / RAW_DIR / f"{family.name}.jsonl"
        existing = []
        if path.exists() and not args.force:
            existing = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        if len(existing) >= args.per_family:
            print(f"skip {family.name}: {len(existing)} rows already present")
            counts[family.label] += len(existing)
            continue
        model = model_for(family, args.safe_model, args.crit_model)
        need = args.per_family - len(existing)
        got: list[dict] = []
        attempts = 0
        while len(got) < need and attempts < args.max_attempts:
            attempts += 1
            batch_n = min(need - len(got), args.batch)
            try:
                items = generate_family_rows(client, model, family, batch_n)
            except Exception as e:  # noqa: BLE001 - OOM/timeout/bad JSON, back off and retry
                print(f"  {family.name}: attempt {attempts} failed ({e}); backing off")
                time.sleep(args.sleep * attempts)
                continue
            for item in items:
                row = coerce_row(item if isinstance(item, dict) else {}, family)
                if row is not None:
                    got.append(row.to_dict())
            time.sleep(args.sleep)
        rows = existing + got
        path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
        counts[family.label] += len(rows)
        print(f"{family.name} ({family.label}, model={model}): {len(rows)}/{args.per_family} rows")
    print("totals:", dict(counts))


def _load_all_raw(out: Path) -> list[dict]:
    rows = []
    for path in sorted((out / RAW_DIR).glob("*.jsonl")):
        rows += [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return rows


def cmd_validate(args: argparse.Namespace) -> None:
    out = Path(args.out)
    rows = _load_all_raw(out)
    result = validate_rows(rows)
    kept, dropped = dedupe_rows(result.kept)
    attach_facts(kept, seed=args.seed)
    for row in kept:
        row["split"] = split_of(row["family"])
    (out / "rejected.jsonl").write_text(
        "\n".join(json.dumps(r) for r in result.rejected) + ("\n" if result.rejected else "")
    )
    (out / "dataset_raw.jsonl").write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    print(
        f"raw {len(rows)} -> validated {len(result.kept)} "
        f"(rejected {len(result.rejected)}) -> deduped {len(kept)} (dropped {dropped})"
    )


def cmd_qa(args: argparse.Namespace) -> None:
    out = Path(args.out)
    rows = [json.loads(ln) for ln in (out / "dataset_raw.jsonl").read_text().splitlines() if ln]
    client = LLMClient()
    summary = apply_qa(rows, client, args.model, args.workers)
    (out / "dataset_raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "qa_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def cmd_jev(args: argparse.Namespace) -> None:
    """Third opinion from hosted Jev 1.13 (synthetic rows only; key never printed)."""
    from concurrent.futures import ThreadPoolExecutor

    from gen_corpus_eval import SHIPPED, jev_key, jev_label
    from jevmon.detect.context import ContextSettings, parse_exec_state, render_segments

    out = Path(args.out)
    rows = [json.loads(ln) for ln in (out / "dataset.jsonl").read_text().splitlines() if ln]
    key = jev_key()
    texts = [
        render_segments(
            parse_exec_state(r["context"]), ContextSettings(order=SHIPPED, model_max=1000)
        )
        for r in rows
    ]

    def one(i: int) -> str:
        for attempt in range(4):
            try:
                return jev_label(texts[i], key)
            except Exception:  # noqa: BLE001 - transient network/API errors
                time.sleep(2 * (attempt + 1))
        return ""

    with ThreadPoolExecutor(args.workers) as ex:
        labels = list(ex.map(one, range(len(rows))))
    for r, lab in zip(rows, labels, strict=True):
        r["jev_label"] = lab
        r["jev_agrees"] = (lab == r["label"]) if lab else None
    (out / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ok = [r for r in rows if r["jev_agrees"] is not None]
    print(
        f"jev labelled {len(ok)}/{len(rows)}; agreement with intent {sum(r['jev_agrees'] for r in ok) / max(1, len(ok)):.3f}"
    )


def cmd_render(args: argparse.Namespace) -> None:
    out = Path(args.out)
    rows = [json.loads(ln) for ln in (out / "dataset_raw.jsonl").read_text().splitlines() if ln]
    render_rows(rows)
    (out / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"rendered {len(rows)} rows -> {out / 'dataset.jsonl'}")


def cmd_card(args: argparse.Namespace) -> None:
    out = Path(args.out)
    rows = [json.loads(ln) for ln in (out / "dataset.jsonl").read_text().splitlines() if ln]
    by_class = Counter(r["label"] for r in rows)
    by_split = Counter(r["split"] for r in rows)
    by_family = Counter(r["family"] for r in rows)
    disputed = sum(1 for r in rows if r.get("disputed"))
    qa = (
        json.loads((out / "qa_summary.json").read_text())
        if (out / "qa_summary.json").exists()
        else {}
    )
    card = {
        "rows": len(rows),
        "families": len(by_family),
        "by_class": dict(by_class),
        "by_split": dict(by_split),
        "disputed_rows": disputed,
        "qa_agreement": qa,
        "prompt_hash": (out / PROMPT_HASH_FILE).read_text().strip()
        if (out / PROMPT_HASH_FILE).exists()
        else None,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "dataset_card.json").write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--taxonomy", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--per-family", type=int, default=20)
    g.add_argument("--batch", type=int, default=6)
    g.add_argument("--safe-model", default="qwen3.8")
    g.add_argument("--crit-model", default="qwen3.8-uncensored")
    g.add_argument("--only", nargs="*", default=[])
    g.add_argument("--max-attempts", type=int, default=4)
    g.add_argument("--sleep", type=float, default=1.0)
    g.add_argument("--force", action="store_true")
    g.add_argument("--shard", default="", help="i/n: this process takes every n-th family")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate")
    v.add_argument("--out", required=True)
    v.add_argument("--seed", type=int, default=19)
    v.set_defaults(func=cmd_validate)

    q = sub.add_parser("qa")
    q.add_argument("--out", required=True)
    q.add_argument("--model", default="qwen3.8")
    q.add_argument("--workers", type=int, default=2)
    q.set_defaults(func=cmd_qa)

    r = sub.add_parser("render")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_render)

    j = sub.add_parser("jev")
    j.add_argument("--out", required=True)
    j.add_argument("--workers", type=int, default=8)
    j.set_defaults(func=cmd_jev)

    c = sub.add_parser("card")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_card)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

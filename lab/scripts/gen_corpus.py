"""E19b: generate the synthetic process-launch corpus with multi-judge consensus.

Pipeline (each stage resumable, each writes its own progress/summary files):

    uv run --project lab python scripts/gen_corpus.py generate --taxonomy ../lab/corpus_taxonomy.toml \
        --out ../lab-data-gen2-<date> --per-family 8 --workers 8
    uv run --project lab python scripts/gen_corpus.py validate --out ../lab-data-gen2-<date>
    uv run --project lab python scripts/gen_corpus.py blind    --out ../lab-data-gen2-<date> --workers 2
    uv run --project lab python scripts/gen_corpus.py render   --out ../lab-data-gen2-<date>
    uv run --project lab python scripts/gen_corpus.py jev      --out ../lab-data-gen2-<date> --workers 8
    uv run --project lab python scripts/gen_corpus.py consensus --out ../lab-data-gen2-<date>
    uv run --project lab python scripts/gen_corpus.py card     --out ../lab-data-gen2-<date>
    uv run --project lab python scripts/gen_corpus.py review   --out ../lab-data-gen2-<date>

Generation goes through OpenRouter (pinned provider, deepseek/deepseek-v4.1-flash) for every
family; families it refuses or returns empty for fall back to the local qwen3.8-uncensored and
are listed in the summary. The generator cannot judge its own rows, so consensus is:
generating intent AND a blind local qwen3.8 pass AND hosted Jev 1.13. Rows that miss any of
the three keep `disputed: true` and are excluded from headline sets. Only synthetic text
leaves the box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gen_corpus_lib import (
    CLASS_DESCRIPTIONS,
    GEN_SYSTEM,
    OR_DEFAULT_MODEL,
    Family,
    LLMClient,
    OpenRouterClient,
    build_prompt,
    coerce_row,
    extract_json,
    load_taxonomy,
)
from gen_corpus_validate import attach_facts, dedupe_rows, qa_label, render_rows, validate_rows
from jevmon_lab.manifest import LABELS, split_of

RAW_DIR = "raw"
PROMPT_HASH_FILE = "prompt_hash.txt"
TELEMETRY_FILE = "gen_telemetry.jsonl"
# DigitalOcean list price for deepseek/deepseek-v4.1-flash output tokens, for the cost estimate.
OUTPUT_PRICE_PER_MTOK = 0.78
# Hosted third judge (OpenRouter decisions API). The key is read from the environment or
# /run/user/1000/secrets, never printed and never written to the data.
JEV_URL = "https://openrouter.ai/api/alpha/decisions"


def jev_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        key = Path("/run/user/1000/secrets/openrouter-api-key").read_text().strip()
    return key


def jev_label(text: str, key: str) -> str:
    body = {
        "model": "typesafe/jev-1.13",
        "state": text,
        "questions": {
            "verdict": {
                "type": "choice",
                "instructions": "Which of these categories best describes this process launch?",
                "criteria": CLASS_DESCRIPTIONS,
            }
        },
    }
    req = urllib.request.Request(
        JEV_URL,
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    probs = resp["answers"]["verdict"]["probabilities"]
    return max(LABELS, key=lambda k: float(probs.get(k, 0.0)))


def prompt_hash() -> str:
    sample = build_prompt(Family("x", "safe", "brief"), 5)
    return hashlib.sha256((GEN_SYSTEM + sample).encode()).hexdigest()[:16]


class TelemetryLog:
    """Thread-safe append of one JSON line per LLM call (tokens, finish_reason, provider)."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.empty = 0
        self.refusal = 0
        self.length_retry = 0
        self.providers: Counter = Counter()
        self.by_class_calls: Counter = Counter()
        self.by_class_empty: Counter = Counter()
        self.by_class_refusal: Counter = Counter()

    def record(
        self, family: str, label: str, client: str, model: str, tel: dict, rows_in_batch: int
    ) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "family": family,
            "label": label,
            "client": client,
            "model": model,
            "finish_reason": tel.get("finish_reason"),
            "empty": tel.get("empty"),
            "refusal": tel.get("refusal"),
            "prompt_tokens": tel.get("prompt_tokens"),
            "completion_tokens": tel.get("completion_tokens"),
            "provider": tel.get("provider"),
            "provider_retries": tel.get("provider_retries"),
            "length_retry": tel.get("length_retry"),
            "max_tokens": tel.get("max_tokens"),
            "rows_in_batch": rows_in_batch,
        }
        with self.lock:
            self.calls += 1
            self.prompt_tokens += int(tel.get("prompt_tokens") or 0)
            self.completion_tokens += int(tel.get("completion_tokens") or 0)
            if tel.get("empty"):
                self.empty += 1
            if tel.get("refusal"):
                self.refusal += 1
            if tel.get("length_retry"):
                self.length_retry += 1
            self.providers[tel.get("provider") or client] += 1
            self.by_class_calls[label] += 1
            if tel.get("empty"):
                self.by_class_empty[label] += 1
            if tel.get("refusal"):
                self.by_class_refusal[label] += 1
            with self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")

    def summary(self, fallback_families: list[str], rows: Counter, families: int) -> dict:
        return {
            "families": families,
            "rows_by_class": dict(rows),
            "llm_calls": self.calls,
            "tokens": {"prompt": self.prompt_tokens, "completion": self.completion_tokens},
            "cost_estimate_usd": round(self.completion_tokens * OUTPUT_PRICE_PER_MTOK / 1e6, 3),
            "empty_replies": self.empty,
            "refusals": self.refusal,
            "length_retries": self.length_retry,
            "per_class": {
                lab: {
                    "calls": self.by_class_calls.get(lab, 0),
                    "empty_rate": (
                        self.by_class_empty.get(lab, 0) / self.by_class_calls[lab]
                        if self.by_class_calls.get(lab)
                        else 0.0
                    ),
                    "refusal_rate": (
                        self.by_class_refusal.get(lab, 0) / self.by_class_calls[lab]
                        if self.by_class_calls.get(lab)
                        else 0.0
                    ),
                }
                for lab in ("safe", "warn", "crit")
            },
            "providers": dict(self.providers),
            "fallback_local_families": fallback_families,
        }


def _coerce_batch(items: object, family) -> list[dict]:
    rows = []
    if not isinstance(items, list):
        return rows
    for item in items:
        if isinstance(item, dict):
            row = coerce_row(item, family)
            if row is not None:
                rows.append(row.to_dict())
    return rows


def _gen_family(
    or_client: OpenRouterClient | None,
    local_client: LLMClient,
    local_model: str,
    family,
    out: Path,
    per_family: int,
    attempts: int,
    local_attempts: int,
    max_tokens: int,
    log: TelemetryLog,
) -> tuple[bool, int]:
    """One family: generate through OpenRouter (pinned provider), local uncensored only as a
    fallback when the family is really refused or every OpenRouter call errored. An empty or
    length-truncated reply is retried, not treated as a refusal (the client already escalates
    max_tokens once). Returns (used_local, rows)."""
    path = out / RAW_DIR / f"{family.name}.jsonl"
    existing: list[dict] = []
    if path.exists():
        existing = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    got = list(existing)
    used_local = False
    refused = False
    ok_calls = 0
    error_calls = 0

    if or_client is not None and len(got) < per_family:
        for _ in range(attempts):
            need = per_family - len(got)
            try:
                text, tel = or_client.chat(
                    OR_DEFAULT_MODEL,
                    build_prompt(family, need),
                    system=GEN_SYSTEM,
                    max_tokens=max_tokens,
                )
            except Exception as e:  # noqa: BLE001 - transport/provider failure, retry or fall back
                error_calls += 1
                log.record(
                    family.name,
                    family.label,
                    "openrouter",
                    OR_DEFAULT_MODEL,
                    {"provider": "openrouter-error", "error": str(e)[:160]},
                    need,
                )
                continue
            ok_calls += 1
            items = None
            try:
                items = extract_json(text)
            except ValueError:
                pass
            got += _coerce_batch(items, family)
            log.record(family.name, family.label, "openrouter", OR_DEFAULT_MODEL, tel, need)
            if tel.get("refusal"):
                refused = True
                break
            if len(got) >= per_family:
                break

    if len(got) < per_family and (refused or or_client is None or (ok_calls == 0 and error_calls)):
        used_local = True
        for _ in range(local_attempts):
            need = per_family - len(got)
            text = local_client.chat(
                local_model,
                build_prompt(family, need),
                system=GEN_SYSTEM,
                temperature=0.9,
                max_tokens=400 + 220 * need,
            )
            items = None
            try:
                items = extract_json(text)
            except ValueError:
                pass
            got += _coerce_batch(items, family)
            log.record(
                family.name,
                family.label,
                "local",
                local_model,
                {"empty": not text.strip(), "provider": "llama.cpp-local"},
                need,
            )
            if len(got) >= per_family:
                break

    rows = got[:per_family]
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    return used_local, len(rows)


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
    log = TelemetryLog(out / TELEMETRY_FILE)

    or_client = None if args.client == "local" else OpenRouterClient()
    local_client = LLMClient()
    counts: Counter = Counter()
    fallback: list[str] = []
    counts_lock = threading.Lock()

    def work(family) -> tuple[str, bool, int]:
        used_local, n = _gen_family(
            or_client,
            local_client,
            args.local_model,
            family,
            out,
            args.per_family,
            args.attempts,
            args.local_attempts,
            args.max_tokens,
            log,
        )
        with counts_lock:
            counts[family.label] += n
        return family.name, used_local, n

    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        results = list(ex.map(work, families))
    for name, used_local, n in results:
        if used_local:
            fallback.append(name)
        print(f"{name}: {n} rows{' (local fallback)' if used_local else ''}")
    summary = log.summary(sorted(fallback), counts, len(families))
    (out / "gen_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


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


def cmd_blind(args: argparse.Namespace) -> None:
    """Blind pass on the local server: the model has not seen the intended label.

    Resumable via blind_progress.jsonl (one line per completed row index)."""
    out = Path(args.out)
    rows = [json.loads(ln) for ln in (out / "dataset_raw.jsonl").read_text().splitlines() if ln]
    progress = out / "blind_progress.jsonl"
    done: set[int] = set()
    if progress.exists():
        done = {json.loads(ln)["i"] for ln in progress.read_text().splitlines() if ln.strip()}
    todo = [i for i, r in enumerate(rows) if i not in done and "qa_label" not in r]
    client = LLMClient()
    lock = threading.Lock()

    def work(i: int) -> None:
        label, conf = qa_label(client, args.model, rows[i])
        rows[i]["qa_label"] = label
        rows[i]["qa_confidence"] = conf
        rows[i]["blind_agrees"] = label == rows[i]["label"]
        with lock, progress.open("a") as f:
            f.write(json.dumps({"i": i}) + "\n")

    print(f"blind pass: {len(todo)} rows, {args.workers} workers, model {args.model}")
    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        list(ex.map(work, todo))
    (out / "dataset_raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    agree: Counter = Counter()
    total: Counter = Counter()
    for r in rows:
        if "qa_label" in r:
            total[r["label"]] += 1
            if r.get("blind_agrees"):
                agree[r["label"]] += 1
    summary = {
        "rows": len(rows),
        "blind_agreement_overall": sum(agree.values()) / max(1, sum(total.values())),
        "blind_agreement_per_class": {k: agree[k] / total[k] for k in total if total[k]},
    }
    (out / "blind_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def cmd_render(args: argparse.Namespace) -> None:
    out = Path(args.out)
    rows = [json.loads(ln) for ln in (out / "dataset_raw.jsonl").read_text().splitlines() if ln]
    render_rows(rows)
    (out / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"rendered {len(rows)} rows -> {out / 'dataset.jsonl'}")


def cmd_jev(args: argparse.Namespace) -> None:
    """Third judge: hosted Jev 1.13 (synthetic rows only; key never printed).

    Cost-saving rule: rows the blind local pass already disagrees with the intent cannot reach
    consensus, so Jev is only asked about the blind-agreeing rows (use --all to override).
    Resumable via jev_progress.jsonl."""
    from concurrent.futures import ThreadPoolExecutor

    from jevmon.detect.context import ContextSettings, parse_exec_state, render_segments

    out = Path(args.out)
    all_rows = [json.loads(ln) for ln in (out / "dataset.jsonl").read_text().splitlines() if ln]
    targets = [
        i
        for i, r in enumerate(all_rows)
        if "jev_label" not in r and (args.all or r.get("blind_agrees"))
    ]
    key = jev_key()
    texts = [
        render_segments(
            parse_exec_state(r["context"]),
            ContextSettings(order=("header", "exe", "template", "argv"), model_max=1000),
        )
        for r in all_rows
    ]
    progress = out / "jev_progress.jsonl"
    lock = threading.Lock()

    def work(i: int) -> str:
        for attempt in range(4):
            try:
                lab = jev_label(texts[i], key)
                with lock:
                    all_rows[i]["jev_label"] = lab
                    all_rows[i]["jev_agrees"] = lab == all_rows[i]["label"]
                    with progress.open("a") as f:
                        f.write(json.dumps({"i": i}) + "\n")
                return lab
            except Exception:  # noqa: BLE001 - transient network/API errors
                time.sleep(2 * (attempt + 1))
        with lock:
            all_rows[i]["jev_label"] = ""
            all_rows[i]["jev_agrees"] = None
            with progress.open("a") as f:
                f.write(json.dumps({"i": i}) + "\n")
        return ""

    print(f"jev pass: {len(targets)} rows, {args.workers} workers")
    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        list(ex.map(work, targets))
    (out / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in all_rows) + "\n")
    labelled = [r for r in all_rows if r.get("jev_agrees") is not None]
    agree = sum(1 for r in labelled if r["jev_agrees"])
    summary = {
        "rows_judged": len(labelled),
        "rows_total": len(all_rows),
        "jev_agreement_with_intent": agree / max(1, len(labelled)),
    }
    (out / "jev_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def cmd_tiebreak(args: argparse.Namespace) -> None:
    """Local uncensored model as the tie-break third judge, only for rows the hosted judge could
    not score (no `jev_label`). Resumable via tiebreak_progress.jsonl."""
    from gen_corpus_validate import qa_label as local_qa_label

    out = Path(args.out)
    rows = [json.loads(ln) for ln in (out / "dataset.jsonl").read_text().splitlines() if ln]
    progress = out / "tiebreak_progress.jsonl"
    done: set[int] = set()
    if progress.exists():
        done = {json.loads(ln)["i"] for ln in progress.read_text().splitlines() if ln.strip()}
    targets = [
        i
        for i, r in enumerate(rows)
        if i not in done and r.get("blind_agrees") and r.get("jev_agrees") is None
    ]
    client = LLMClient()
    lock = threading.Lock()

    def work(i: int) -> None:
        label, conf = local_qa_label(client, args.model, rows[i])
        with lock:
            rows[i]["unc_label"] = label
            rows[i]["unc_confidence"] = conf
            rows[i]["unc_agrees"] = label == rows[i]["label"]
            with progress.open("a") as f:
                f.write(json.dumps({"i": i}) + "\n")

    print(f"tiebreak pass: {len(targets)} rows, {args.workers} workers, model {args.model}")
    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        list(ex.map(work, targets))
    (out / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    judged = [r for r in rows if r.get("unc_agrees") is not None]
    agree = sum(1 for r in judged if r["unc_agrees"])
    summary = {
        "rows_judged": len(judged),
        "unc_agreement_with_intent": agree / max(1, len(judged)),
    }
    (out / "tiebreak_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def consensus_of(row: dict) -> bool:
    """A row is kept only if its generating intent, the blind local pass and a third judge (the
    hosted Jev, or the local uncensored tie-break when the hosted judge could not score it)
    all agree. This is the `disputed` rule and the headline-set filter."""
    third = bool(row.get("jev_agrees")) or bool(row.get("unc_agrees"))
    return bool(row.get("blind_agrees")) and third


def cmd_consensus(args: argparse.Namespace) -> None:
    """Consensus = generating intent AND the blind local pass AND (hosted Jev OR the local
    uncensored tie-break) agree. Disputed rows stay in the data, flagged, and are out of the
    headline sets."""
    out = Path(args.out)
    rows = [json.loads(ln) for ln in (out / "dataset.jsonl").read_text().splitlines() if ln]
    for r in rows:
        r["consensus"] = consensus_of(r)
        r["disputed"] = not r["consensus"]
    (out / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def rate(rows, key) -> dict:
        c: Counter = Counter()
        t: Counter = Counter()
        for r in rows:
            t[r["label"]] += 1
            if r.get(key):
                c[r["label"]] += 1
        return {k: c[k] / t[k] for k in t if t[k]}

    fams: dict = {}
    for r in rows:
        f = fams.setdefault(
            r["family"],
            {"label": r["label"], "rows": 0, "blind": 0, "jev": 0, "unc": 0, "consensus": 0},
        )
        f["rows"] += 1
        f["blind"] += int(bool(r.get("blind_agrees")))
        f["jev"] += int(bool(r.get("jev_agrees")))
        f["unc"] += int(bool(r.get("unc_agrees")))
        f["consensus"] += int(r["consensus"])
    for f in fams.values():
        f["blind_rate"] = f["blind"] / f["rows"]
        f["jev_rate"] = f["jev"] / f["rows"]
        f["unc_rate"] = f["unc"] / f["rows"]
        f["consensus_rate"] = f["consensus"] / f["rows"]
    zero = sorted(n for n, f in fams.items() if f["consensus"] == 0)
    report = {
        "rows": len(rows),
        "families": len(fams),
        "by_class": dict(Counter(r["label"] for r in rows)),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "blind_agreement_per_class_before": rate(rows, "blind_agrees"),
        "jev_agreement_per_class_before": rate(rows, "jev_agrees"),
        "unc_agreement_per_class_before": rate(rows, "unc_agrees"),
        "consensus_per_class_after": rate(rows, "consensus"),
        "consensus_overall_after": sum(1 for r in rows if r["consensus"]) / max(1, len(rows)),
        "consensus_target": 0.8,
        "target_met": sum(1 for r in rows if r["consensus"]) / max(1, len(rows)) >= 0.8,
        "disputed_rows": sum(1 for r in rows if r["disputed"]),
        "families_zero_consensus": zero,
        "surviving_families": sum(1 for f in fams.values() if f["consensus"] > 0),
        "family_agreement": fams,
    }
    (out / "consensus_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def cmd_card(args: argparse.Namespace) -> None:
    out = Path(args.out)
    rows = [json.loads(ln) for ln in (out / "dataset.jsonl").read_text().splitlines() if ln]
    by_class = Counter(r["label"] for r in rows)
    by_split = Counter(r["split"] for r in rows)
    by_family = Counter(r["family"] for r in rows)
    disputed = sum(1 for r in rows if r.get("disputed"))
    gen_summary = (
        json.loads((out / "gen_summary.json").read_text())
        if (out / "gen_summary.json").exists()
        else {}
    )
    consensus_report = (
        json.loads((out / "consensus_report.json").read_text())
        if (out / "consensus_report.json").exists()
        else {}
    )
    card = {
        "rows": len(rows),
        "families": len(by_family),
        "by_class": dict(by_class),
        "by_split": dict(by_split),
        "consensus_rows": sum(1 for r in rows if r.get("consensus")),
        "disputed_rows": disputed,
        "generation": gen_summary,
        "consensus": {
            k: consensus_report.get(k)
            for k in (
                "blind_agreement_per_class_before",
                "jev_agreement_per_class_before",
                "unc_agreement_per_class_before",
                "consensus_per_class_after",
                "consensus_overall_after",
                "target_met",
                "families_zero_consensus",
                "surviving_families",
            )
        },
        "class_descriptions": CLASS_DESCRIPTIONS,
        "prompt_hash": (out / PROMPT_HASH_FILE).read_text().strip()
        if (out / PROMPT_HASH_FILE).exists()
        else None,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "dataset_card.json").write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1))


def cmd_review(args: argparse.Namespace) -> None:
    """~150 rows for human labelling: across classes, spanning consensus and disputed, model
    labels (blind/jev) hidden so the reviewer does not anchor on them."""
    out = Path(args.out)
    rows = [json.loads(ln) for ln in (out / "dataset.jsonl").read_text().splitlines() if ln]
    rng = random.Random(14)
    per_class = max(1, args.n // 3)
    picked: list[int] = []
    for cls in ("safe", "warn", "crit"):
        cons = [i for i, r in enumerate(rows) if r["label"] == cls and r.get("consensus")]
        disp = [i for i, r in enumerate(rows) if r["label"] == cls and not r.get("consensus")]
        n_c = min(len(cons), round(per_class * 0.7))
        n_d = min(len(disp), per_class - n_c)
        picked += rng.sample(cons, n_c) + rng.sample(disp, n_d)
    picked = sorted(set(picked))
    cols = ("idx", "family", "split", "label", "consensus", "context")
    lines = ["\t".join(cols)]
    for i in picked:
        r = rows[i]
        ctx = r["context"].replace("\n", " ").replace("\t", " ")
        lines.append(
            "\t".join(
                (
                    str(i),
                    r["family"],
                    r["split"],
                    r["label"],
                    "true" if r.get("consensus") else "false",
                    ctx,
                )
            )
        )
    (out / "review.tsv").write_text("\n".join(lines) + "\n")
    print(f"review.tsv: {len(picked)} rows -> {out / 'review.tsv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--taxonomy", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--per-family", type=int, default=8)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--attempts", type=int, default=3, help="OpenRouter attempts per family")
    g.add_argument("--local-attempts", type=int, default=3, help="local fallback attempts")
    g.add_argument("--max-tokens", type=int, default=16000)
    g.add_argument("--client", choices=("openrouter", "local"), default="openrouter")
    g.add_argument("--local-model", default="qwen3.8-uncensored")
    g.add_argument("--only", nargs="*", default=[])
    g.add_argument("--shard", default="", help="i/n: this process takes every n-th family")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate")
    v.add_argument("--out", required=True)
    v.add_argument("--seed", type=int, default=19)
    v.set_defaults(func=cmd_validate)

    b = sub.add_parser("blind")
    b.add_argument("--out", required=True)
    b.add_argument("--model", default="qwen3.8")
    b.add_argument("--workers", type=int, default=2)
    b.set_defaults(func=cmd_blind)

    r = sub.add_parser("render")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_render)

    j = sub.add_parser("jev")
    j.add_argument("--out", required=True)
    j.add_argument("--workers", type=int, default=8)
    j.add_argument("--all", action="store_true", help="also judge rows the blind pass disputed")
    j.set_defaults(func=cmd_jev)

    t = sub.add_parser("tiebreak")
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="qwen3.8-uncensored")
    t.add_argument("--workers", type=int, default=2)
    t.set_defaults(func=cmd_tiebreak)

    c = sub.add_parser("consensus")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_consensus)

    cd = sub.add_parser("card")
    cd.add_argument("--out", required=True)
    cd.set_defaults(func=cmd_card)

    rv = sub.add_parser("review")
    rv.add_argument("--out", required=True)
    rv.add_argument("--n", type=int, default=150)
    rv.set_defaults(func=cmd_review)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Zero-shot sweep: question wording, category descriptions and context text format.

The lab dataset (`lab-data/dataset.jsonl`, exported by `jevmon-lab run --judge none`) already
holds every recorded context with its ground truth and family split. This script judges each
row through the shipped `LayaDecider` (the same path `jevmon-lab report --judge laya` and the
daemon use) under one variant at a time and scores train vs held-out families with
`jevmon_lab.metrics`, so every headline number carries the always-safe baseline and the
majority-class rate.

    just gpu-run-lab python lab/scripts/sweep_zero_shot.py --round 1
    just gpu-run-lab python lab/scripts/sweep_zero_shot.py --round 2 --question q_classify --categories c_rich
    just gpu-run-lab python lab/scripts/sweep_zero_shot.py --round 3 --format f_argv_first_lines
    just gpu-run-lab python lab/scripts/sweep_zero_shot.py --round 4 --question q_expected
    just gpu-run-lab python lab/scripts/sweep_zero_shot.py --round 5 --categories c_rich
    just gpu-run-lab python lab/scripts/sweep_zero_shot.py --round final --question ... --categories ... --format ...

Rounds pick by held-out macro-F1 (the held-out families are mostly `safe`, so accuracy
alone is the class mix talking). Results accumulate in `lab-data/sweep.json`.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from jevmon.config import load_config
from jevmon.detect.context import ContextSettings, parse_exec_state, render_segments
from jevmon.laya.decider import LayaConfig, LayaDecider
from jevmon_lab import metrics
from jevmon_lab.runlog import RunLog, class_table, score_scalars

CHUNK = 16  # contexts per decide_batch call, as in lab/dataset.py

QUESTIONS = {
    "q_expected": (
        "Is this process launch expected on a developer workstation that runs swarms of "
        "coding agents?"
    ),
    "q_classify": "How would you classify this process launch?",
    "q_best_fit": "Which of these categories best describes this process launch?",
    "q_triage": (
        "You are triaging a process launch on a developer workstation. Which category fits it?"
    ),
    "q_which": "Which category best describes this process launch?",
    "q_labels": "Classify this process launch as safe, warn or crit.",
    "q_categorise": "Categorise this process launch using the given categories.",
}
# Round 5 refines around the round-3 winner.
FOCUS_QUESTIONS = ("q_best_fit", "q_which", "q_labels", "q_categorise")

CATEGORY_SETS = {
    "c_shipped": {
        "safe": "normal operation for this workstation",
        "warn": "unexpected on this workstation but plausibly harmless",
        "crit": "possible attack, intrusion or abuse",
    },
    "c_rich": {
        "safe": (
            "expected everyday activity here: shells, builds, package installs, scheduled "
            "jobs, ssh sessions and other development tooling; no action needed"
        ),
        "warn": (
            "out of the ordinary for this workstation but plausibly harmless: a one-off "
            "script, the first run of a newly installed tool, a debugger; worth a glance"
        ),
        "crit": (
            "looks hostile from the launch alone: downloading and running remote code, "
            "encoded or obfuscated commands, execution from a temporary, world-writable or "
            "hidden location, a shell spawned by an unexpected parent"
        ),
    },
    "c_terse": {
        "safe": "benign, expected on this machine",
        "warn": "unusual but harmless",
        "crit": "malicious or abusive",
    },
}

_ARGV_FIRST = (
    "header",
    "template",
    "argv",
    "exe",
    "parents",
    "user",
    "seen",
    "surprisal",
    "struct",
    "env",
)
_ARGV_FIRST_NO_STATS = tuple(name for name in _ARGV_FIRST if name not in ("seen", "surprisal"))
_MINIMAL = ("header", "template", "argv")
_CMD = ("header", "exe", "template", "argv")
_FOCUS = ("header", "template", "argv", "exe", "struct")
FORMATS = {
    # f_base is the format stored contexts were captured with (every segment, one line).
    "f_base": {"order": ()},
    "f_lines": {"order": (), "sep": "\n"},
    "f_argv_first_lines": {"sep": "\n", "order": _ARGV_FIRST},
    "f_argv_first_lines_nostats": {"sep": "\n", "order": _ARGV_FIRST_NO_STATS},
    "f_argv_first_lines_nostats_short_argv": {
        "sep": "\n",
        "order": _ARGV_FIRST_NO_STATS,
        "argv_max": 160,
    },
    "f_argv_first": {"order": _ARGV_FIRST},
    "f_minimal": {"order": _MINIMAL},
    "f_cmd": {"order": _CMD},
    "f_minimal_lines": {"sep": "\n", "order": _MINIMAL},
    "f_focus_lines": {"sep": "\n", "order": _FOCUS},
    "f_cmd_lines": {"sep": "\n", "order": _CMD},
}
# The focused formats are round 4: what is left when only what the command did is shown.
FOCUS_FORMATS = ("f_minimal", "f_cmd", "f_minimal_lines", "f_focus_lines")


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("label")]


def segments_by_row(rows: list[dict]) -> list[list[tuple[str, str]]]:
    parsed = [parse_exec_state(row["context"]) for row in rows]
    captured = ContextSettings(order=(), sep=" ")  # the format the stored run rendered
    for row, segs in zip(rows, parsed, strict=True):
        if render_segments(segs, captured) != row["context"]:
            raise SystemExit(f"round-trip failed for signature {row['signature']}")
    return parsed


def build_decider() -> LayaDecider:
    cfg = load_config()
    base = LayaConfig(
        categories=dict(CATEGORY_SETS["c_shipped"]),
        attention_question=QUESTIONS["q_expected"],
        attack_min=1.0,
        offline=True,
        threads=cfg.judge.threads,
        threads_gpu=cfg.judge.threads_gpu,
        device=cfg.judge.device,
        dtype=cfg.judge.dtype,
    )
    decider = LayaDecider(base)
    decider.load()
    if decider.error:
        raise SystemExit(f"model failed to load: {decider.error}")
    return decider


def judge(
    decider: LayaDecider,
    question: str,
    categories: dict[str, str],
    states: list[str],
) -> list[str]:
    decider.reconfigure(
        replace(decider.config, attention_question=question, categories=dict(categories))
    )
    verdicts: list[str] = []
    for start in range(0, len(states), CHUNK):
        chunk = states[start : start + CHUNK]
        decisions = decider.decide_batch(chunk)
        verdicts.extend(d.label if d.available else "unavailable" for d in decisions)
    return verdicts


def scored_splits(rows: list[dict], verdicts: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for split in ("train", "heldout"):
        picked = [
            {"label": row["label"], "verdict": verdict}
            for row, verdict in zip(rows, verdicts, strict=True)
            if row["split"] == split
        ]
        out[split] = metrics.score(picked)
    return out


def summarize(name: str, result: dict[str, dict]) -> dict:
    held = result["heldout"]
    train = result["train"]
    recalls = held["per_class"]
    return {
        "variant": name,
        "heldout_macro_f1": held["macro_f1"],
        "heldout_accuracy": held["accuracy"],
        "train_macro_f1": train["macro_f1"],
        "heldout": held,
        "train": train,
        "recall_warn": recalls.get("warn", {}).get("recall", 0.0),
        "recall_crit": recalls.get("crit", {}).get("recall", 0.0),
        "precision_safe": recalls.get("safe", {}).get("precision", 0.0),
    }


def print_summary(summary: dict) -> None:
    held = summary["heldout"]
    print(
        f"{summary['variant']:<36} heldout macroF1 {held['macro_f1']:.3f} "
        f"(always-safe {held['always_safe_macro_f1']:.3f}) "
        f"acc {held['accuracy']:.3f} (majority {held['majority_rate']:.3f}) "
        f"train macroF1 {summary['train_macro_f1']:.3f} "
        f"recalls w/c {summary['recall_warn']:.2f}/{summary['recall_crit']:.2f} "
        f"safe-precision {summary['precision_safe']:.2f}"
    )


def save(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else []
    by_name = {row["variant"]: row for row in existing}
    for row in results:
        by_name[row["variant"]] = row
    path.write_text(json.dumps(list(by_name.values()), indent=1) + "\n")


def run(
    decider: LayaDecider,
    rows: list[dict],
    parsed: list[list[tuple[str, str]]],
    variants: list[tuple[str, str, str, str]],
    results_path: Path,
    track: RunLog | None = None,
) -> list[dict]:
    results: list[dict] = []
    for step, (q_name, c_name, f_name, label) in enumerate(variants):
        question = QUESTIONS[q_name]
        categories = CATEGORY_SETS[c_name]
        settings = replace(ContextSettings(), **FORMATS[f_name])
        rendered = [render_segments(segs, settings) for segs in parsed]
        started = time.perf_counter()
        verdicts = judge(decider, question, categories, rendered)
        elapsed = time.perf_counter() - started
        summary = summarize(label, scored_splits(rows, verdicts))
        summary["seconds"] = round(elapsed, 1)
        print_summary(summary)
        results.append(summary)
        if track is not None:
            track.log(
                step,
                **score_scalars("heldout", summary["heldout"]),
                **score_scalars("train", summary["train"]),
                seconds=elapsed,
            )
            track.table(
                "variants",
                [
                    {
                        "step": n,
                        "variant": r["variant"],
                        "heldout_macro_f1": round(r["heldout_macro_f1"], 4),
                        "heldout_accuracy": round(r["heldout_accuracy"], 4),
                        "train_macro_f1": round(r["train_macro_f1"], 4),
                        "seconds": r["seconds"],
                    }
                    for n, r in enumerate(results)
                ],
                step=step,
            )
    save(results_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="zero-shot judge sweep on the lab dataset")
    parser.add_argument("--data", type=Path, default=Path("lab-data/dataset.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("lab-data/sweep.json"))
    parser.add_argument("--round", choices=("1", "2", "3", "4", "5", "final"), default="1")
    parser.add_argument("--question", default="q_expected")
    parser.add_argument("--categories", default="c_shipped")
    parser.add_argument("--format", default="f_base")
    parser.add_argument("--no-track", action="store_true", help="skip experiment tracking")
    args = parser.parse_args()

    rows = load_rows(args.data)
    parsed = segments_by_row(rows)
    print(f"{len(rows)} labelled contexts loaded", flush=True)
    decider = build_decider()
    print(f"model: {decider.status}", flush=True)
    cfg = decider.config
    track = RunLog.start(
        "jevmon-finetune",
        time.strftime(f"zero-shot-sweep-round{args.round}-%Y%m%d-%H%M%S"),
        {
            "script": "sweep_zero_shot.py",
            "round": args.round,
            "question": args.question,
            "categories": args.categories,
            "format": args.format,
            "pooling": "laya decision head (zero-shot, no pooling choice)",
            "device": decider.active_device or cfg.device,
            "dtype": decider.active_dtype or cfg.dtype,
            "threads": cfg.threads,
            "seed": None,  # zero-shot inference: nothing is sampled
            "n_rows": len(rows),
            "chunk": CHUNK,
            "data": args.data,
        },
        enabled=not args.no_track,
    )

    if args.round == "1":
        variants = [(q, c, "f_base", f"{q}|{c}|f_base") for q in QUESTIONS for c in CATEGORY_SETS]
    elif args.round == "2":
        variants = [
            (args.question, args.categories, f, f"{args.question}|{args.categories}|{f}")
            for f in FORMATS
        ]
    elif args.round == "3":
        variants = [
            (q, c, args.format, f"{q}|{c}|{args.format}") for q in QUESTIONS for c in CATEGORY_SETS
        ]
    elif args.round == "4":
        variants = [
            (args.question, c, f, f"{args.question}|{c}|{f}")
            for c in CATEGORY_SETS
            for f in FOCUS_FORMATS
        ]
    elif args.round == "5":
        variants = [
            (q, args.categories, f, f"{q}|{args.categories}|{f}")
            for q in FOCUS_QUESTIONS
            for f in ("f_cmd", "f_cmd_lines")
        ]
    else:
        label = f"FINAL {args.question}|{args.categories}|{args.format}"
        results = run(
            decider,
            rows,
            parsed,
            [(args.question, args.categories, args.format, label)],
            args.out,
            track,
        )
        for split in ("heldout", "train"):
            print(f"\n### {split}\n")
            print("\n".join(metrics.lines(results[0][split])))
            track.table(f"per_class/{split}", class_table(results[0][split]))
        track.finish()
        return
    results = run(decider, rows, parsed, variants, args.out, track)
    track.finish()
    results.sort(key=lambda row: row["heldout_macro_f1"], reverse=True)
    print("\nranked by held-out macro-F1:")
    for row in results:
        print_summary(row)


if __name__ == "__main__":
    main()

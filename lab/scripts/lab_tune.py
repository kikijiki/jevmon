"""Model selection with train-family cross-validation, held-out families scored once.

The discipline the stage-3 experiment follows:
every choice - question wording, category descriptions, context format, epochs, learning rate,
class balancing - is made by grouped cross-validation INSIDE the train families (a family never
straddles a fold). Held-out families are scored once per final candidate and never fed back.
Every table carries the always-safe baseline and the majority-class rate, plus per-class
precision/recall with support.

    just gpu-run-lab python lab/scripts/lab_tune.py merge lab-data/r1 lab-data/r2 lab-data/r3 --out lab-data
    just gpu-run-lab python lab/scripts/lab_tune.py zero                  # zero-shot, picked by train-CV
    just gpu-run-lab python lab/scripts/lab_tune.py ft-cv --name lr1e-5 --encoder-lr 1e-5 --balance 4
    just gpu-run-lab python lab/scripts/lab_tune.py ft-final --name lr1e-5 --encoder-lr 1e-5 --epochs 3

Full fine-tune runs are capped by `lab-data/ft-runs.json` (`MAX_FT_RUNS`).
"""

import argparse
import gc
import itertools
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # shared helpers live next to this script

import numpy as np
import torch

from finetune_head import grouped_folds, load_rows, render_states
from jevmon.config import load_config
from jevmon.detect.context import ContextSettings, parse_exec_state, render_segments
from jevmon_lab import metrics
from jevmon_lab.runlog import RunLog, class_table, score_scalars
from sweep_zero_shot import CATEGORY_SETS, FORMATS, QUESTIONS, build_decider, judge

PROJECT = "jevmon-finetune"
MAX_FT_RUNS = 12
BASE_CHECKPOINT = "convaiinnovations/laya"
STAGE1_PICK = (
    "q_best_fit",
    "c_rich",
    "f_cmd",
)  # picked on held-out in stage 1: reported, not chosen

Q_GRID = ("q_best_fit", "q_which", "q_triage", "q_expected")
C_GRID = ("c_shipped", "c_rich", "c_terse")
F_GRID = ("f_cmd", "f_minimal", "f_cmd_lines", "f_base")


# --------------------------------------------------------------------------------- data


def cmd_merge(args: argparse.Namespace) -> None:
    """Concatenate runs into one dataset.jsonl + contexts.jsonl (same order, same format)."""
    args.out.mkdir(parents=True, exist_ok=True)
    data, ctxs = [], []
    for run in args.runs:
        data += [ln for ln in (run / "dataset.jsonl").read_text().splitlines() if ln.strip()]
        ctxs += [ln for ln in (run / "contexts.jsonl").read_text().splitlines() if ln.strip()]
    if len(data) != len(ctxs):
        raise SystemExit("dataset/contexts length mismatch")
    (args.out / "dataset.jsonl").write_text("\n".join(data) + "\n")
    (args.out / "contexts.jsonl").write_text("\n".join(ctxs) + "\n")
    print(f"merged {len(args.runs)} runs: {len(data)} contexts -> {args.out}")


def describe(rows: list[dict]) -> dict:
    out: dict = {}
    for split in ("train", "heldout"):
        part = [r for r in rows if r["split"] == split]
        by_label: dict = {}
        for r in part:
            by_label.setdefault(r["label"], {"rows": 0, "unique": set(), "families": set()})
            e = by_label[r["label"]]
            e["rows"] += 1
            e["unique"].add(r["context"])
            e["families"].add(r["family"])
        out[split] = {
            lab: {"rows": e["rows"], "unique": len(e["unique"]), "families": len(e["families"])}
            for lab, e in sorted(by_label.items())
        }
    return out


def print_description(rows: list[dict]) -> dict:
    info = describe(rows)
    for split, labs in info.items():
        print(f"\n{split}:")
        for lab, e in labs.items():
            print(
                f"  {lab:<20} rows {e['rows']:>4}  unique contexts {e['unique']:>4}  families {e['families']}"
            )
    return info


def score_split(rows: list[dict], verdicts: list[str]) -> dict:
    return metrics.score(
        [{"label": r["label"], "verdict": v} for r, v in zip(rows, verdicts, strict=True)]
    )


def show(title: str, result: dict) -> None:
    print(f"\n### {title}\n")
    print("\n".join(metrics.lines(result)))


def split_rows(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    rows = load_rows(path)
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] == "heldout"]
    if not train or not held:
        raise SystemExit("empty split")
    return rows, train, held


# ----------------------------------------------------------------------------- zero-shot


def cmd_zero(args: argparse.Namespace) -> None:
    rows, train, held = split_rows(args.data)
    print(f"{len(rows)} labelled rows: {len(train)} train / {len(held)} held-out")
    info = print_description(rows)
    parsed_train = [parse_exec_state(r["_raw"]) for r in train]
    parsed_held = [parse_exec_state(r["_raw"]) for r in held]
    folds = grouped_folds(train, 5)
    decider = build_decider()
    print(f"model: {decider.status}", flush=True)
    run = RunLog.start(
        PROJECT,
        time.strftime("zero-shot-train-cv-%Y%m%d-%H%M%S"),
        {
            "script": "lab_tune.py zero",
            "selection": "5-fold family-grouped CV inside train families; held-out scored once",
            "questions": list(Q_GRID),
            "categories": list(C_GRID),
            "formats": list(F_GRID),
            "device": decider.active_device or decider.config.device,
            "dtype": decider.active_dtype or decider.config.dtype,
            "threads": decider.config.threads,
            "seed": None,
            "n_train": len(train),
            "n_heldout": len(held),
            "support": json.dumps(info),
        },
        enabled=not args.no_track,
    )

    cache: dict[tuple[str, str, str, str], str] = {}

    def verdicts_for(q: str, c: str, f: str, parsed: list, rows_: list[dict]) -> list[str]:
        settings = replace(ContextSettings(), **FORMATS[f])
        states = [render_segments(segs, settings) for segs in parsed]
        todo = sorted({s for s in states if (q, c, f, s) not in cache})
        if todo:
            out = judge(decider, QUESTIONS[q], CATEGORY_SETS[c], todo)
            cache.update({(q, c, f, s): v for s, v in zip(todo, out, strict=True)})
        return [cache[(q, c, f, s)] for s in states]

    results = []
    for step, (q, c, f) in enumerate(itertools.product(Q_GRID, C_GRID, F_GRID)):
        started = time.perf_counter()
        verdicts = verdicts_for(q, c, f, parsed_train, train)
        fold_scores = [
            score_split([train[n] for n in fold], [verdicts[n] for n in fold])["macro_f1"]
            for fold in folds
        ]
        pooled = score_split(train, verdicts)
        cv = sum(fold_scores) / len(fold_scores)
        results.append({"config": (q, c, f), "cv": cv, "pooled": pooled})
        run.log(
            step,
            **{"cv/fold_mean_macro_f1": cv, **score_scalars("train_pooled", pooled)},
        )
        print(
            f"{q:<12} {c:<10} {f:<12} train-CV macroF1 {cv:.3f} pooled {pooled['macro_f1']:.3f} "
            f"(always-safe {pooled['always_safe_macro_f1']:.3f}) "
            f"[{time.perf_counter() - started:.0f}s]",
            flush=True,
        )
    results.sort(key=lambda r: -r["cv"])
    run.table(
        "train_cv_grid",
        [
            {
                "question": r["config"][0],
                "categories": r["config"][1],
                "format": r["config"][2],
                "cv_macro_f1": round(r["cv"], 4),
                "pooled_macro_f1": round(r["pooled"]["macro_f1"], 4),
                "always_safe_macro_f1": round(r["pooled"]["always_safe_macro_f1"], 4),
            }
            for r in results
        ],
    )
    best = results[0]["config"]
    print(f"\ntrain-CV pick: {best} (CV macro-F1 {results[0]['cv']:.3f})")
    report: dict = {"cv_pick": list(best), "stage1_pick": list(STAGE1_PICK)}
    # held-out: scored once for each of two pre-declared candidates (the CV pick and the stage-1
    # pick that was chosen on held-out and is therefore optimistic)
    for tag, config in (("cv_pick", best), ("stage1_pick", STAGE1_PICK)):
        verdicts = verdicts_for(*config, parsed_held, held)
        result = score_split(held, verdicts)
        train_res = score_split(train, verdicts_for(*config, parsed_train, train))
        show(f"held-out families: zero-shot {tag} {config}", result)
        show(f"train families: zero-shot {tag} {config} (no fitting: same as CV pooled)", train_res)
        report[tag] = {"heldout": result, "train": train_res}
        run.log(
            len(results) + (tag == "stage1_pick"),
            **score_scalars(f"heldout_{tag}", result),
            **score_scalars(f"train_{tag}", train_res),
        )
        run.table(f"per_class/heldout_{tag}", class_table(result))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=list) + "\n")
    run.finish()


# --------------------------------------------------------------------------- fine-tune


def ft_count() -> int:
    path = Path("lab-data/ft-runs.json")
    return len(json.loads(path.read_text())) if path.exists() else 0


def ft_log(entry: dict) -> None:
    path = Path("lab-data/ft-runs.json")
    runs = json.loads(path.read_text()) if path.exists() else []
    runs.append(entry)
    path.write_text(json.dumps(runs, indent=1) + "\n")


def questions_for(judge_cfg) -> dict:
    return {
        "attention": {
            "type": "choice",
            "instructions": judge_cfg.question,
            "criteria": dict(judge_cfg.categories),
        }
    }


def training_items(
    rows: list[dict], states: list[str], questions: dict, balance: int
) -> list[dict]:
    """Unique (state, label) rows, minority classes repeated up to `balance` times."""
    unique: dict[tuple[str, str], dict] = {}
    for row, state in zip(rows, states, strict=True):
        unique.setdefault(
            (state, row["label"]),
            {"state": state, "questions": questions, "expected": {"attention": row["label"]}},
        )
    items = list(unique.values())
    counts: dict[str, int] = {}
    for it in items:
        counts[it["expected"]["attention"]] = counts.get(it["expected"]["attention"], 0) + 1
    top = max(counts.values())
    out: list[dict] = []
    for it in items:
        repeat = 1
        if balance > 1:
            repeat = max(1, min(balance, round(top / counts[it["expected"]["attention"]])))
        out.extend([it] * repeat)
    return out


def make_config(args: argparse.Namespace, epochs: int):
    from laya.train import TrainConfig

    cfg = TrainConfig(
        epochs=epochs,
        micro_batch=8,
        grad_accum=1,
        loss="soft-ce",
        shuffle_options=("choice",),
        freeze_encoder=False,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        seed=0,
        log_every=0,
    )
    cfg.validate()
    return cfg


def fold_run(args, train_rows, val_rows, labels, questions, run, fold_no):
    """Train on `train_rows`, return val macro-F1 after each epoch (val never trained on)."""
    import laya.train as lt

    torch.set_num_threads(2)
    device = lt.resolve_device("auto")
    model, tok, cfg = lt.load_checkpoint(BASE_CHECKPOINT)
    model.to(device)
    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)
    settings = ContextSettings()
    tr_items = training_items(
        train_rows, render_states(train_rows, settings), questions, args.balance
    )
    va_states = render_states(val_rows, settings)
    items, _skipped = lt.items_from_rows(tok, tr_items, max_len, head_max_len, args.label_smoothing)
    va_rows = [
        {"state": s, "questions": questions, "expected": {"attention": r["label"]}}
        for r, s in zip(val_rows, va_states, strict=True)
    ]
    val_items, _ = lt.items_from_rows(tok, va_rows, max_len, head_max_len)
    if len(val_items) != len(val_rows):
        raise SystemExit("validation items lost in encoding")
    curve: list[dict] = []

    def on_epoch(epoch: int, loss: float) -> None:
        recs = lt.calibration_records(model, tok, val_items, device, max_len, head_max_len)
        verdicts = [labels[int(np.argmax(r[1]))] for r in recs]
        model.train()
        sc = score_split(val_rows, verdicts)
        curve.append(
            {
                "epoch": epoch + 1,
                "loss": loss,
                "macro_f1": sc["macro_f1"],
                "accuracy": sc["accuracy"],
            }
        )
        run.log(
            epoch,
            **{f"fold{fold_no}/val_macro_f1": sc["macro_f1"], f"fold{fold_no}/train_loss": loss},
        )
        print(
            f"  fold {fold_no} epoch {epoch + 1}: loss {loss:.3f} val macro-F1 {sc['macro_f1']:.3f}",
            flush=True,
        )

    lt.train_model(
        model,
        tok,
        items,
        make_config(args, args.max_epochs),
        device,
        max_len,
        head_max_len,
        on_epoch_end=on_epoch,
    )
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    return curve, len(items)


def cmd_ft_cv(args: argparse.Namespace) -> None:
    _rows, train, _held = split_rows(args.data)
    labels = tuple(load_config().judge.categories)
    questions = questions_for(load_config().judge)
    folds = grouped_folds(train, args.folds)
    if ft_count() + len(folds) > MAX_FT_RUNS:
        raise SystemExit(f"fine-tune budget: {ft_count()} used + {len(folds)} > {MAX_FT_RUNS}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    run = RunLog.start(
        PROJECT,
        f"full-finetune-cv-{args.name}-{time.strftime('%H%M%S')}",
        {
            "script": "lab_tune.py ft-cv",
            "name": args.name,
            "folds": len(folds),
            "max_epochs": args.max_epochs,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "balance_repeat_cap": args.balance,
            "loss": "soft-ce",
            "micro_batch": 8,
            "seed": 0,
            "device": "auto (iGPU)",
            "threads": 2,
            "n_train": len(train),
        },
        enabled=not args.no_track,
    )
    curves = []
    for k, val in enumerate(folds):
        keep = set(val)
        tr_rows = [r for n, r in enumerate(train) if n not in keep]
        va_rows = [train[n] for n in val]
        started = time.perf_counter()
        curve, n_items = fold_run(args, tr_rows, va_rows, labels, questions, run, k)
        ft_log(
            {"kind": "cv", "name": args.name, "fold": k, "seconds": time.perf_counter() - started}
        )
        print(f"fold {k}: {n_items} train items, {time.perf_counter() - started:.0f}s", flush=True)
        curves.append(curve)
    epochs = min(len(c) for c in curves)
    mean = [sum(c[e]["macro_f1"] for c in curves) / len(curves) for e in range(epochs)]
    best = int(np.argmax(mean)) + 1
    print(f"\ntrain-CV macro-F1 by epoch: {[round(m, 3) for m in mean]} -> best epoch {best}")
    for e, m in enumerate(mean):
        run.log(e, **{"cv/mean_val_macro_f1": m})
    run.table(
        "cv_by_epoch",
        [{"epoch": e + 1, "mean_val_macro_f1": round(m, 4)} for e, m in enumerate(mean)],
    )
    out = Path("lab-data/ft-cv")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(
        json.dumps(
            {"args": vars(args), "mean": mean, "best_epoch": best, "curves": curves},
            indent=1,
            default=str,
        )
        + "\n"
    )
    run.finish()


def cmd_ft_final(args: argparse.Namespace) -> None:
    import laya.train as lt

    _rows, train, held = split_rows(args.data)
    base = load_config()
    questions = questions_for(base.judge)
    if ft_count() + 1 > MAX_FT_RUNS:
        raise SystemExit("fine-tune budget exhausted")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    items = training_items(train, render_states(train, ContextSettings()), questions, args.balance)
    data_path = args.out.parent / f"{args.name}-train.jsonl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text("".join(json.dumps(i) + "\n" for i in items))
    cfg = make_config(args, args.epochs)
    run = RunLog.start(
        PROJECT,
        f"full-finetune-final-{args.name}-{time.strftime('%H%M%S')}",
        {
            "script": "lab_tune.py ft-final",
            "name": args.name,
            "epochs": args.epochs,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "balance_repeat_cap": args.balance,
            "loss": "soft-ce",
            "micro_batch": 8,
            "seed": 0,
            "device": "auto (iGPU)",
            "threads": 2,
            "n_train_items": len(items),
        },
        enabled=not args.no_track,
    )
    torch.set_num_threads(2)
    started = time.perf_counter()
    summary = lt.finetune(
        data=str(data_path),
        model_dir=BASE_CHECKPOINT,
        output_dir=str(args.out),
        config=cfg,
        device="auto",
    )
    ft_log({"kind": "final", "name": args.name, "seconds": time.perf_counter() - started})
    for e, loss in enumerate(summary["epoch_loss"]):
        run.log(e, **{"train/epoch_loss": float(loss)})
    gc.collect()
    from finetune_laya import judge_config

    tuned = __import__("jevmon.monitor", fromlist=["build_decider"]).build_decider(
        judge_config(base, args.out)
    )
    tuned.load()
    if tuned.error:
        raise SystemExit(f"tuned model failed: {tuned.error}")
    chunk = 16

    def decide(states: list[str]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(states), chunk):
            out += [d.label for d in tuned.decide_batch(states[i : i + chunk])]
        return out

    held_res = score_split(held, decide([r["context"] for r in held]))
    train_res = score_split(train, decide([r["context"] for r in train]))
    show(f"held-out families: full fine-tune {args.name}", held_res)
    show(f"train families: full fine-tune {args.name} (resubstitution)", train_res)
    run.log(
        args.epochs,
        **score_scalars("heldout", held_res),
        **score_scalars("train", train_res),
    )
    run.table("per_class/heldout", class_table(held_res))
    run.table("per_class/train_resubstitution", class_table(train_res))
    Path(f"lab-data/ft-final-{args.name}.json").write_text(
        json.dumps(
            {"args": vars(args), "heldout": held_res, "train": train_res}, indent=1, default=str
        )
        + "\n"
    )
    run.finish()


def cmd_head_cv(args: argparse.Namespace) -> None:
    """The frozen-embedding head scored on the same folds the fine-tune CV uses (comparator)."""
    from finetune_head import class_weights, embed, fit_head, predict

    _rows, train, _held = split_rows(args.data)
    labels = tuple(load_config().judge.categories)
    decider = build_decider()
    agent = decider._agent(decider._router)
    vectors = embed(agent, render_states(train, ContextSettings()), "mean")
    y = np.asarray([labels.index(r["label"]) for r in train])
    folds = grouped_folds(train, args.folds)
    run = RunLog.start(
        PROJECT,
        f"head-frozen-embeddings-cv{args.folds}-comparator-{time.strftime('%H%M%S')}",
        {
            "script": "lab_tune.py head-cv",
            "folds": args.folds,
            "pooling": "mean",
            "weight_decay": 0.01,
            "class_weight": "balanced",
            "n_train": len(train),
        },
        enabled=not args.no_track,
    )
    scores = []
    for k, val in enumerate(folds):
        keep = set(val)
        tr = [n for n in range(len(train)) if n not in keep]
        w, b = fit_head(vectors[tr], y[tr], class_weights(y[tr], True), 0.01)
        verdicts = predict(vectors[val], labels, w, b)
        sc = score_split([train[n] for n in val], verdicts)
        scores.append(sc["macro_f1"])
        run.log(k, **{"cv/fold_macro_f1": sc["macro_f1"]})
        print(f"fold {k}: macro-F1 {sc['macro_f1']:.3f}")
    mean = sum(scores) / len(scores)
    run.log(len(folds), **{"cv/mean_macro_f1": mean})
    print(f"head train-CV ({args.folds} folds, same folds as ft-cv): {mean:.3f}")
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("merge")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("lab-data"))
    p.set_defaults(fn=cmd_merge)
    p = sub.add_parser("head-cv")
    p.add_argument("--data", type=Path, default=Path("lab-data/dataset.jsonl"))
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--no-track", action="store_true")
    p.set_defaults(fn=cmd_head_cv)
    p = sub.add_parser("zero")
    p.add_argument("--data", type=Path, default=Path("lab-data/dataset.jsonl"))
    p.add_argument("--out", type=Path, default=Path("lab-data/zero-shot-cv.json"))
    p.add_argument("--no-track", action="store_true")
    p.set_defaults(fn=cmd_zero)
    for name, fn in (("ft-cv", cmd_ft_cv), ("ft-final", cmd_ft_final)):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=Path("lab-data/dataset.jsonl"))
        p.add_argument("--name", required=True)
        p.add_argument("--encoder-lr", type=float, default=1e-5)
        p.add_argument("--head-lr", type=float, default=1e-4)
        p.add_argument("--weight-decay", type=float, default=0.01)
        p.add_argument("--label-smoothing", type=float, default=0.0)
        p.add_argument("--balance", type=int, default=1, help="max repeat of minority classes")
        p.add_argument("--no-track", action="store_true")
        if name == "ft-cv":
            p.add_argument("--folds", type=int, default=3)
            p.add_argument("--max-epochs", type=int, default=5)
        else:
            p.add_argument("--epochs", type=int, required=True)
            p.add_argument("--out", type=Path, default=Path("models/lab-ft"))
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

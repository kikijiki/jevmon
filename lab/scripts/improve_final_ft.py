"""Held-out scoring of the final candidates (once each) on merged clean-label runs.

    python scripts/improve_final_ft.py lab-data-r6 lab-data-r7 --lora 6,3e-4,8 --full 6,2e-5 \
        --head-wd 0.1 --seeds 3 --export models/improve

Candidates (chosen on train-CV beforehand, passed explicitly): shipped-recipe head, CV-best head,
LoRA, full fine-tune (seeds), plus the stored shipped head, zero-shot and always-safe.
"""

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from improve_ft import FTSpec, Trainer, export_checkpoint, weights_for
from improve_lib import (
    LABELS,
    EmbeddingCache,
    Spec,
    fit_predict,
    laya_agent,
    load_rows,
    make_data,
    render,
    score_pair,
)
from jevmon.laya.head import TunedHead
from jevmon_lab.runlog import RunLog, class_table, score_scalars

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+", type=Path)
ap.add_argument("--lora", default="")
ap.add_argument("--full", default="")
ap.add_argument("--head-wd", type=float, default=0.03)
ap.add_argument("--head-z", action="store_true", help="z-score the CV-best head")
ap.add_argument("--skip-heads", action="store_true")
ap.add_argument("--skip-zero-shot", action="store_true")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--export", type=Path, default=None)
ap.add_argument("--tag", default="e12")
ap.add_argument("--project", default="jevmon-3class")
ap.add_argument("--checkpoint", default="english", help="encoder checkpoint (backbone)")
ap.add_argument("--stored", type=Path, default=None, help="a previously shipped head to score too")
ap.add_argument("--runs", type=Path, default=Path(os.environ.get("JEVMON_RUNS", "runs")))
args = ap.parse_args()

rows = []
for d in args.dirs:
    rows += load_rows(d / "dataset.jsonl")
data, texts = make_data(rows, lambda r: render(r, "shipped"))
tr, te = data.idx("train"), data.idx("heldout")
print(f"train {len(tr)} heldout {len(te)}", flush=True)
results: dict[str, list[dict]] = {}


def show(tag: str, s: dict, f1: list[float], acc: list[float], n_seeds: int) -> None:
    print(
        f"   [{tag}] n={s['n']} acc {np.mean(acc):.3f}+-{np.std(acc):.3f} "
        f"(always-safe {s['always_safe_accuracy']:.3f}, majority {s['majority_class']} "
        f"{s['majority_rate']:.3f})  macro-F1 {np.mean(f1):.3f}+-{np.std(f1):.3f} "
        f"(always-safe {s['always_safe_macro_f1']:.3f})  seeds {n_seeds}",
        flush=True,
    )
    for lab, v in s["per_class"].items():
        print(
            f"      {lab:6s} support {v['support']:3d} predicted {v['predicted']:3d} "
            f"P {v['precision']:.2f} R {v['recall']:.2f} F1 {v['f1']:.2f}"
        )


def record(name: str, config: dict, pairs: list[dict]) -> None:
    """`pairs`: one `score_pair` result per seed ({"rows", "unique"})."""
    results[name] = pairs
    print(f"\n## {name}", flush=True)
    for tag in ("rows", "unique"):
        scs = [p[tag] for p in pairs]
        show(
            tag,
            scs[0],
            [x["macro_f1"] for x in scs],
            [x["accuracy"] for x in scs],
            len(scs),
        )
    s, u = pairs[0]["rows"], pairs[0]["unique"]
    f1 = [p["rows"]["macro_f1"] for p in pairs]
    run = RunLog.start(
        args.project,
        f"{args.tag}-final-{name}-{time.strftime('%H%M%S')}",
        config,
        directory=args.runs,
    )
    run.log(
        0,
        **score_scalars("heldout", s),
        **{f"unique/{k}": v for k, v in score_scalars("heldout", u).items()},
        **{"heldout/f1_mean": float(np.mean(f1)), "heldout/f1_sd": float(np.std(f1))},
    )
    run.table("heldout_per_class", class_table(s))
    run.finish()


cache = EmbeddingCache(lambda: laya_agent(args.checkpoint), args.checkpoint)
x = cache.get(texts, "mean")
for name, spec in (
    ()
    if args.skip_heads
    else (
        ("head-shipped-recipe", Spec("logreg", wd=0.01, balanced=True)),
        (
            f"head-cv-best{'-z' if args.head_z else ''}-wd{args.head_wd}",
            Spec("logreg", wd=args.head_wd, balanced=True, standardize=args.head_z),
        ),
    )
):
    proba = fit_predict(spec, x[tr], data.y[tr], data.key[tr], x[te])
    record(name, {"spec": spec.name()}, [score_pair(data, te, proba.argmax(1))])

if args.stored is not None:
    head = TunedHead.load(args.stored)
    pred = head.probabilities(cache.get(texts, head.pooling)[te]).argmax(1)
    record("stored-shipped-head", {"head": str(args.stored)}, [score_pair(data, te, pred)])

specs = []
if args.lora:
    e, lr, r = args.lora.split(",")
    specs.append(("lora", FTSpec("lora", epochs=int(e), lr_enc=float(lr), rank=int(r))))
if args.full:
    e, lr = args.full.split(",")
    specs.append(("full", FTSpec("full", epochs=int(e), lr_enc=float(lr))))
if specs:
    trainer = Trainer(laya_agent(args.checkpoint))
    for name, spec in specs:
        scs = []
        for seed in range(args.seeds):
            s2 = replace(spec, seed=seed)
            w = weights_for(s2, data.y[tr], data.key[tr])
            proba = trainer.fit_predict(
                s2, [texts[i] for i in tr], data.y[tr], w, [texts[i] for i in te]
            )
            scs.append(score_pair(data, te, proba.argmax(1)))
            if seed == 0 and args.export is not None:
                weight, bias = export_checkpoint(trainer, args.export / f"ckpt-{name}", s2)
                TunedHead(labels=LABELS, weight=weight, bias=bias, pooling="mean").save(
                    args.export / f"head-{name}.safetensors"
                )
        record(f"{name}-{spec.name().replace('/', '_')}", {"spec": spec.name()}, scs)

if not args.skip_zero_shot:
    from jevmon.config import load_config
    from jevmon.monitor import build_decider

    decider = build_decider(load_config())
    decider.load()
    zs = [
        LABELS.index(d.label) for d in decider.decide_batch([data.rows[i]["context"] for i in te])
    ]
    record("zero-shot", {}, [score_pair(data, te, np.asarray(zs))])
    # the held-out majority/always-safe baselines sit in every record above (same rows)

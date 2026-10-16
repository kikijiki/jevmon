"""Score candidates on held-out families, once each, on a clean-label dataset.

    python scripts/improve_final.py lab-data-r6/dataset.jsonl [--runs DIR]

Candidates are fixed before looking at held-out: the shipped recipe (shipped format, mean pooling,
logreg wd 0.01 balanced), the train-CV best of the sweep, the control with a constant parent
segment, the stored shipped head file, and the zero-shot judge. Everything is tracked in Trackio
project `jevmon-improve`.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from improve_lib import (
    LABELS,
    EmbeddingCache,
    Spec,
    cv_scores,
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
ap.add_argument("data", type=Path)
ap.add_argument("--runs", type=Path, default=Path(os.environ.get("JEVMON_RUNS", "runs")))
ap.add_argument("--tag", default="r6")
ap.add_argument("--head", type=Path, default=Path("models/lab-head-v4.safetensors"))
args = ap.parse_args()

rows = load_rows(args.data)
cache = EmbeddingCache(laya_agent)
candidates = [
    ("shipped-recipe", "shipped", Spec("logreg", wd=0.01, balanced=True)),
    ("shipped-z", "shipped", Spec("logreg", wd=0.01, balanced=True, standardize=True)),
    ("parent-z", "parent", Spec("logreg", wd=0.1, balanced=True, standardize=True)),
]


def show(title: str, sc: dict) -> None:
    print(
        f"{title}: n={sc['n']} acc={sc['accuracy']:.3f} (always-safe {sc['always_safe_accuracy']:.3f},"
        f" majority {sc['majority_class']} {sc['majority_rate']:.3f}) macro-F1={sc['macro_f1']:.3f}"
        f" (always-safe {sc['always_safe_macro_f1']:.3f})"
    )
    for lab, v in sc["per_class"].items():
        print(
            f"   {lab:11s} support {v['support']:3d} predicted {v['predicted']:3d}"
            f" P {v['precision']:.2f} R {v['recall']:.2f} F1 {v['f1']:.2f}"
        )


for name, fmt, spec in candidates:
    data, texts = make_data(rows, lambda r, fmt=fmt: render(r, fmt))
    tr, te = data.idx("train"), data.idx("heldout")
    x = cache.get(texts, "mean")
    cv = cv_scores(spec, x, data, tr, repeats=5)
    run = RunLog.start(
        "jevmon-improve",
        f"{args.tag}-final-{name}-{time.strftime('%H%M%S')}",
        {
            "data": str(args.data),
            "format": fmt,
            "spec": spec.name(),
            "n_train": len(tr),
            "n_heldout": len(te),
        },
        directory=args.runs,
    )
    accs = []
    for seed in range(5):  # logreg is deterministic; bootstrap of the train rows gives the spread
        rng = np.random.default_rng(seed)
        idx = tr if seed == 0 else rng.choice(tr, size=len(tr), replace=True)
        proba = fit_predict(spec, x[idx], data.y[idx], data.key[idx], x[te])
        sc = score_pair(data, te, proba.argmax(1))["rows"]
        accs.append(sc["macro_f1"])
        if seed == 0:
            head_sc = sc
    print(
        f"\n## {name} [{fmt}] {spec.name()}  train-CV fold-mean {cv['fold_mean_f1']:.3f} rows {cv['rows_f1']:.3f}+-{cv['rows_f1_sd']:.3f}"
    )
    show("held-out", head_sc)
    print(f"   bootstrap(train rows) macro-F1 {np.mean(accs[1:]):.3f}+-{np.std(accs[1:]):.3f}")
    run.log(
        0,
        **score_scalars("heldout", head_sc),
        **{
            "cv/fold_mean_f1": cv["fold_mean_f1"],
            "boot/f1_mean": float(np.mean(accs[1:])),
            "boot/f1_sd": float(np.std(accs[1:])),
        },
    )
    run.table("heldout_per_class", class_table(head_sc))
    run.finish()

# stored shipped head and zero-shot on the same held-out rows
data, texts = make_data(rows, lambda r: render(r, "shipped"))
te = data.idx("heldout")
head = TunedHead.load(args.head)
x = cache.get(texts, head.pooling)
pred = head.probabilities(x[te]).argmax(1)
sc = score_pair(data, te, pred)["rows"]
print(f"\n## stored shipped head ({args.head})")
show("held-out", sc)
run = RunLog.start(
    "jevmon-improve",
    f"{args.tag}-final-stored-head-{time.strftime('%H%M%S')}",
    {"head": str(args.head)},
    directory=args.runs,
)
run.log(0, **score_scalars("heldout", sc))
run.finish()

from jevmon.config import load_config
from jevmon.monitor import build_decider

decider = build_decider(load_config())
decider.load()
verdicts = [d.label for d in decider.decide_batch([data.rows[i]["context"] for i in te])]
zs = [LABELS.index(v) for v in verdicts]
sc = score_pair(data, te, np.asarray(zs))["rows"]
print("\n## zero-shot (shipped config)")
show("held-out", sc)
run = RunLog.start(
    "jevmon-improve",
    f"{args.tag}-final-zero-shot-{time.strftime('%H%M%S')}",
    {},
    directory=args.runs,
)
run.log(0, **score_scalars("heldout", sc))
run.finish()

"""Train-CV sweep over heads and encoder training (LoRA / full) on merged clean-label runs.

    python scripts/improve_sweep.py lab-data-r6 lab-data-r7 --kind head|ft [--partitions 2]

Reads `dataset.jsonl` + `contexts.jsonl` of every directory, keeps designated (labelled) rows,
and scores each configuration on the same family-grouped partitions of the train families.
Every configuration is one Trackio run in `--project` (default `jevmon-3class`).
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

from improve_ft import FTSpec, Trainer, cv_ft
from improve_lib import (
    LABELS,
    EmbeddingCache,
    Spec,
    cv_scores,
    family_folds,
    laya_agent,
    load_rows,
    make_data,
    render,
)
from jevmon_lab.runlog import RunLog

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+", type=Path)
ap.add_argument("--kind", choices=("head", "ft"), required=True)
ap.add_argument("--partitions", type=int, default=2)
ap.add_argument("--folds", type=int, default=4)
ap.add_argument("--tag", default="s8")
ap.add_argument("--project", default="jevmon-3class")
ap.add_argument("--checkpoint", default="english", help="encoder checkpoint (backbone)")
ap.add_argument("--only", default="")
ap.add_argument("--runs", type=Path, default=Path(os.environ.get("JEVMON_RUNS", "runs")))
args = ap.parse_args()
sys.stdout.reconfigure(line_buffering=True)

rows = []
for d in args.dirs:
    rows += load_rows(d / "dataset.jsonl")
data, texts = make_data(rows, lambda r: render(r, "shipped"))
tr = data.idx("train")
print(
    f"labelled {len(rows)} train {len(tr)} heldout {len(data.idx('heldout'))} "
    f"train families {len(set(data.family[tr]))}",
    flush=True,
)
parts = [family_folds(data, tr, args.folds, 1000 + p) for p in range(args.partitions)]


def track(name: str, config: dict, result: dict) -> None:
    run = RunLog.start(
        args.project,
        f"{args.tag}-{name}-{time.strftime('%H%M%S')}",
        config,
        directory=args.runs,
    )
    run.log(0, **{f"cv/{k}": v for k, v in result.items() if isinstance(v, int | float)})
    run.finish()


if args.kind == "head":
    cache = EmbeddingCache(lambda: laya_agent(args.checkpoint), args.checkpoint)
    for pooling in ("mean", "mean+cls"):
        x = cache.get(texts, pooling)
        specs = [
            Spec("logreg", wd=w, balanced=True, standardize=z)
            for w in (0.003, 0.01, 0.03, 0.1)
            for z in (False, True)
        ]
        specs += [Spec("knn", k=k, norm=True) for k in (3, 7)] + [
            Spec("proto", temp=0.05, norm=True)
        ]
        specs += [
            Spec("mlp", hidden=h, dropout=0.3, wd=0.01, epochs=e, norm=True, seed=s)
            for h in (128, 512)
            for e in (60,)
            for s in (0, 1)
        ]
        for spec in specs:
            r = cv_scores(spec, x, data, tr, folds=args.folds, repeats=5)
            print(
                f"{pooling:9s} {spec.name():45s} rows {r['rows_f1']:.3f}+-{r['rows_f1_sd']:.3f} fold {r['fold_mean_f1']:.3f} rec {[round(r['unique_recall'][k], 2) for k in LABELS]}"
            )
            track(
                f"head-{pooling}-{spec.name().replace('/', '_')}",
                {"pooling": pooling, "spec": spec.name()},
                r,
            )
else:
    trainer = Trainer(laya_agent(args.checkpoint))
    grid = []
    for epochs, lr, rank in itertools.product((6, 10), (1e-4, 3e-4, 1e-3), (8,)):
        grid.append(FTSpec("lora", epochs=epochs, lr_enc=lr, rank=rank))
    for epochs, lr in itertools.product((6, 10, 14), (1e-5, 2e-5, 4e-5)):
        grid.append(FTSpec("full", epochs=epochs, lr_enc=lr))
    for spec in grid:
        if args.only and args.only not in spec.name():
            continue
        res = []
        for p, folds in enumerate(parts):
            for seed in (0, 1):
                from dataclasses import replace

                res.append(cv_ft(trainer, replace(spec, seed=seed), texts, data, tr, folds))
        out = {
            "unique_f1": float(np.mean([r["unique_f1"] for r in res])),
            "unique_f1_sd": float(np.std([r["unique_f1"] for r in res])),
            "fold_mean_f1": float(np.mean([r["fold_mean_f1"] for r in res])),
            "seconds": float(np.mean([r["seconds"] for r in res])),
            "n_fits": len(res),
        }
        print(
            f"{spec.name():55s} F1 {out['unique_f1']:.3f}+-{out['unique_f1_sd']:.3f} fold {out['fold_mean_f1']:.3f} ({out['seconds']:.0f}s/part)"
        )
        track(f"ft-{spec.name().replace('/', '_')}", {"spec": spec.name()}, out)

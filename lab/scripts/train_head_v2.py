"""Train and ship the stage-4 head: shipped format, mean pooling, logreg, class-balanced,
(text,label)-unique sample weights (so the 490-row idle_daemon line counts once). Selection was
train-CV only (docs/improve.md); this script only refits on all train rows and scores held-out
once, with a bootstrap spread for the held-out number (resampling train rows, same config).

    uv run python scripts/train_head_v2.py --out models/lab-head-v2.safetensors
"""

import argparse
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

FORMAT = "shipped"
POOLING = "mean"
SPEC_WD = 0.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("lab-data-merged/dataset.jsonl"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--sample", default="unique", choices=("unique", "rows", "sqrt"))
    args = ap.parse_args()
    spec = Spec("logreg", wd=SPEC_WD, balanced=True, sample=args.sample)

    if args.out is None:
        args.out = Path(f"models/lab-head-v2-{args.sample}.safetensors")
    rows = load_rows(args.data)
    data, texts = make_data(rows, lambda r: render(r, FORMAT))
    tr, te = data.idx("train"), data.idx("heldout")
    cache = EmbeddingCache(laya_agent)
    x = cache.get(texts, POOLING)

    run = RunLog.start(
        "jevmon-improve",
        time.strftime(f"head-v2-shipped-mean-logreg-{args.sample}-%Y%m%d-%H%M%S"),
        {
            "script": "train_head_v2.py",
            "data": str(args.data),
            "out": str(args.out),
            "format": FORMAT,
            "pooling": POOLING,
            "spec": spec.name(),
            "n_rows": len(rows),
            "n_families": len({r["family"] for r in rows}),
            "n_train": len(tr),
            "n_heldout": len(te),
        },
        enabled=not args.no_track,
    )

    cv = cv_scores(spec, x, data, tr, repeats=5)
    print(
        f"train-CV (5 repeats x 5 grouped folds): rows {cv['rows_f1']:.3f}+-{cv['rows_f1_sd']:.3f} "
        f"unique {cv['unique_f1']:.3f}+-{cv['unique_f1_sd']:.3f} fold-mean {cv['fold_mean_f1']:.3f}"
    )
    run.log(0, **{f"cv/{k}": v for k, v in cv.items() if not isinstance(v, dict)})

    # fit on every train row, score held-out once
    from improve_lib import _prep, sample_weights

    xtr, xte = _prep(spec, x[tr], x[te])
    w = sample_weights(spec, data.y[tr], data.key[tr], len(LABELS))
    w = w / max(w.mean(), 1e-12)
    import torch
    import torch.nn.functional as F

    xt, yt, wt = (
        torch.from_numpy(xtr),
        torch.from_numpy(data.y[tr]),
        torch.from_numpy(w.astype(np.float32)),
    )
    wgt = torch.zeros(len(LABELS), xtr.shape[1], requires_grad=True)
    b = torch.zeros(len(LABELS), requires_grad=True)
    opt = torch.optim.LBFGS([wgt, b], lr=1.0, line_search_fn="strong_wolfe", max_iter=200)

    def closure():
        opt.zero_grad()
        ce = F.cross_entropy(xt @ wgt.T + b, yt, reduction="none")
        loss = (ce * wt).sum() / wt.sum() + 0.5 * spec.wd * (wgt * wgt).sum()
        loss.backward()
        return loss

    opt.step(closure)
    head = TunedHead(
        labels=LABELS, weight=wgt.detach().numpy(), bias=b.detach().numpy(), pooling=POOLING
    )
    head.save(args.out)
    print(f"saved {args.out}")

    pred = head.probabilities(xte).argmax(1)
    held = score_pair(data, te, pred)
    print("\nheld-out families (scored once):")
    for split_name, sc in held.items():
        print(
            f"  {split_name}: n={sc['n']} accuracy={sc['accuracy']:.3f} "
            f"(always-safe {sc['always_safe_accuracy']:.3f}, majority {sc['majority_rate']:.3f}) "
            f"macro-F1={sc['macro_f1']:.3f} (always-safe {sc['always_safe_macro_f1']:.3f})"
        )
        for lab, v in sc["per_class"].items():
            print(
                f"    {lab:12s} support={v['support']:3d} predicted={v['predicted']:3d} "
                f"P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f}"
            )
    run.log(0, **score_scalars("heldout_rows", held["rows"]))
    run.log(0, **score_scalars("heldout_unique", held["unique"]))
    run.table("heldout_rows_per_class", class_table(held["rows"]))
    run.table("heldout_unique_per_class", class_table(held["unique"]))

    # bootstrap spread of the held-out number (resample TRAIN rows, same config, refit, re-score;
    # held-out itself is fixed and scored once per the rules above)
    rng = np.random.default_rng(0)
    boot_rows, boot_uniq = [], []
    for i in range(8):
        idx = rng.choice(tr, size=len(tr), replace=True)
        proba = fit_predict(spec, x[idx], data.y[idx], data.key[idx], x[te])
        p = proba.argmax(1)
        sc = score_pair(data, te, p)
        boot_rows.append(sc["rows"]["macro_f1"])
        boot_uniq.append(sc["unique"]["macro_f1"])
    print(
        f"\nheld-out bootstrap spread (8 train resamples, same config): "
        f"rows {np.mean(boot_rows):.3f}+-{np.std(boot_rows):.3f} "
        f"unique {np.mean(boot_uniq):.3f}+-{np.std(boot_uniq):.3f}"
    )
    run.log(
        0,
        **{
            "heldout_boot/rows_f1_mean": float(np.mean(boot_rows)),
            "heldout_boot/rows_f1_sd": float(np.std(boot_rows)),
            "heldout_boot/unique_f1_mean": float(np.mean(boot_uniq)),
            "heldout_boot/unique_f1_sd": float(np.std(boot_uniq)),
        },
    )
    run.finish()


if __name__ == "__main__":
    main()

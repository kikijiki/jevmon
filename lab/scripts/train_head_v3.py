"""Fit the shipped-recipe head on clean-label runs (train families only) and verify it end to end.

    python scripts/train_head_v3.py lab-data-r6 lab-data-r7 --out models/lab-head-v4.safetensors

Shipped format, mean pooling, logistic head, class-balanced, wd 0.01. Held-out is scored from the
saved file through the real `LayaDecider` (`[judge] tuned`), so what ships is what is measured.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

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
from jevmon.config import load_config
from jevmon.laya.head import TunedHead
from jevmon.monitor import build_decider

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+", type=Path)
ap.add_argument("--out", type=Path, default=Path("models/lab-head-v4.safetensors"))
ap.add_argument("--checkpoint", default="english", help="encoder checkpoint (backbone)")
args = ap.parse_args()
rows = []
for d in args.dirs:
    rows += load_rows(d / "dataset.jsonl")
data, texts = make_data(rows, lambda r: render(r, "shipped"))
tr, te = data.idx("train"), data.idx("heldout")
cache = EmbeddingCache(lambda: laya_agent(args.checkpoint), args.checkpoint)
x = cache.get(texts, "mean")
spec = Spec("logreg", wd=0.01, balanced=True)

# fit once more through the harness' own LBFGS to read weights back out
import torch
import torch.nn.functional as F

from improve_lib import sample_weights

w = sample_weights(spec, data.y[tr], data.key[tr], len(LABELS))
w = w / max(w.mean(), 1e-12)
xt, yt, wt = (
    torch.from_numpy(x[tr]),
    torch.from_numpy(data.y[tr]),
    torch.from_numpy(w.astype(np.float32)),
)
wg = torch.zeros(len(LABELS), x.shape[1], requires_grad=True)
b = torch.zeros(len(LABELS), requires_grad=True)
opt = torch.optim.LBFGS([wg, b], lr=1.0, line_search_fn="strong_wolfe", max_iter=200)


def closure():
    opt.zero_grad()
    ce = F.cross_entropy(xt @ wg.T + b, yt, reduction="none")
    loss = (ce * wt).sum() / wt.sum() + 0.5 * spec.wd * (wg * wg).sum()
    loss.backward()
    return loss


opt.step(closure)
TunedHead(labels=LABELS, weight=wg.detach().numpy(), bias=b.detach().numpy(), pooling="mean").save(
    args.out
)
print("saved", args.out)

# bootstrap spread of the held-out score (train rows resampled, same recipe)
f1s, accs = [], []
for seed in range(20):
    idx = np.random.default_rng(seed).choice(tr, size=len(tr), replace=True)
    p = fit_predict(spec, x[idx], data.y[idx], data.key[idx], x[te]).argmax(1)
    sc = score_pair(data, te, p)["rows"]
    f1s.append(sc["macro_f1"])
    accs.append(sc["accuracy"])
print(
    f"bootstrap x20: macro-F1 {np.mean(f1s):.3f}+-{np.std(f1s):.3f} acc {np.mean(accs):.3f}+-{np.std(accs):.3f}"
)

# through the real decider
cfg = load_config()
cfg = replace(cfg, judge=replace(cfg.judge, tuned=str(args.out)))
decider = build_decider(cfg)
decider.load()
print("decider:", decider.status)
verdicts = [d.label for d in decider.decide_batch([data.rows[i]["context"] for i in te])]
pred = np.asarray([LABELS.index(v) for v in verdicts])
direct = TunedHead.load(args.out).probabilities(x[te]).argmax(1)
sc = score_pair(data, te, pred)["rows"]
print(
    f"via LayaDecider: acc {sc['accuracy']:.3f} macro-F1 {sc['macro_f1']:.3f}; "
    f"identical to direct head: {(pred == direct).mean():.3f}"
)

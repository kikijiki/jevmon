"""Write the E13/E14/E15 frozen-backbone heads as loadable safetensors under lab/heads/.

Each head is the E15 variant-A recipe (RECIPES in settle_eval.py) fitted on the 206 train rows
from the cached backbone embeddings; a z-scoring step is folded into weight and bias so the file
is a plain linear head over the raw pooled embedding. The pool macro-F1 on the 420 held-out rows
is recomputed and stored in the metadata. Needs lab/cache/bb-*-512.pkl and the E15 captures.

    python lab/scripts/ship_heads.py --train D --old D --re D --fresh D --out lab/heads
"""

import argparse
import datetime
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file

from ctx_eval import load, sets_of, variant_data
from improve_lib import _prep, fit_predict, sample_weights, to_rows
from jevmon.laya.head import FORMAT_VERSION
from jevmon_lab import metrics
from jevmon_lab.manifest import LABELS
from settle_eval import RECIPES

HEADS = {
    "foundation-sec-8b": (
        "foundation-sec-8b",
        "fdtn-ai/Foundation-Sec-8B",
        "mean of 3/4-depth layer",
    ),
    "qwen3e-8b": ("qwen3e-8b", "Qwen/Qwen3-Embedding-8B", "mean of 3/4-depth layer"),
    "ettin-1b": ("ettin-1b", "jhu-clsp/ettin-encoder-1b", "mean of 3/4-depth layer"),
}


def fit_linear(spec, xtr, ytr, ktr):
    """Same objective as improve_lib._logreg; returns (weight, bias) on the raw features."""
    raw = xtr
    xs, _ = _prep(spec, xtr, xtr)
    w = sample_weights(spec, ytr, ktr, len(LABELS))
    w = w / max(w.mean(), 1e-12)
    xt, yt, wt = torch.from_numpy(xs), torch.from_numpy(ytr), torch.from_numpy(w.astype(np.float32))
    wg = torch.zeros(len(LABELS), xs.shape[1], requires_grad=True)
    b = torch.zeros(len(LABELS), requires_grad=True)
    opt = torch.optim.LBFGS([wg, b], lr=1.0, line_search_fn="strong_wolfe", max_iter=200)

    def closure():
        opt.zero_grad()
        ce = F.cross_entropy(xt @ wg.T + b, yt, reduction="none", label_smoothing=spec.smoothing)
        loss = (ce * wt).sum() / wt.sum() + 0.5 * spec.wd * (wg * wg).sum()
        loss.backward()
        return loss

    opt.step(closure)
    wn, bn = wg.detach().numpy(), b.detach().numpy()
    if spec.standardize:
        mu, sd = raw.mean(0, keepdims=True), raw.std(0, keepdims=True) + 1e-6
        wn = wn / sd
        bn = bn - (wn * mu).sum(1)
    return wn.astype(np.float32), bn.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("train", "old", "re", "fresh"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("lab/cache"))
    args = ap.parse_args()
    data = load(args)
    d, texts = variant_data(data, "A")
    tr = data.idx("train")
    pool = sets_of(data)["pool"]
    fams = sorted({data.family[i] for i in tr})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (src, model, layer) in HEADS.items():
        _s, var, spec, _m, _k, _r = RECIPES[name]
        store = pickle.loads((args.cache / f"bb-{src}-512.pkl").read_bytes())
        x = np.stack([store[t][var] for t in texts]).astype(np.float32)
        w, b = fit_linear(spec, x[tr], d.y[tr], d.key[tr])
        pred = (x @ w.T + b).argmax(1)
        ref = fit_predict(spec, x[tr], d.y[tr], d.key[tr], x).argmax(1)
        f1 = metrics.score(to_rows(data, pool, pred[pool]))["macro_f1"]
        f1_ref = metrics.score(to_rows(data, pool, ref[pool]))["macro_f1"]
        print(
            f"{name}: dim {x.shape[1]} pool macro-F1 {f1:.4f} (recipe {f1_ref:.4f}), "
            f"folded vs recipe agree {float((pred == ref).mean()):.4f}"
        )
        meta = {
            "format_version": FORMAT_VERSION,
            "labels": json.dumps(list(LABELS)),
            "pooling": "mean",
            "max_length": "512",
            "backbone": model,
            "layer": layer,
            "recipe": str(spec.name()) + (" standardize" if spec.standardize else ""),
            "train_rows": str(len(tr)),
            "train_families": str(len(fams)),
            "text_format": "E15 variant A: header, exe, template, argv(redacted)",
            "heldout_rows": str(len(pool)),
            "heldout_families": str(len({data.family[i] for i in pool})),
            "pool_macro_f1": f"{f1:.3f}",
            "date": datetime.date.today().isoformat(),
        }
        save_file({"weight": w, "bias": b}, str(args.out / f"{name}.safetensors"), metadata=meta)


if __name__ == "__main__":
    main()

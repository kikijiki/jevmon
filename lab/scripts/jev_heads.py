"""E18 local side: export the E15 captures as probe rows and dump per-row predictions of the
local heads (Laya recipe head, Foundation-Sec-8B head), both fitted on the E15 train rows with the
shipped variant-A text, so Jev is compared on exactly the same rows. Needs the E15 caches
(cache/emb-english-mean-512.pkl, cache/bb-foundation-sec-8b-512.pkl); run from the directory that
holds cache/. No model is loaded when the caches are complete.

    python scripts/jev_heads.py --train D --old D --re D --fresh D --rows-out rows.jsonl \
        --preds-out heads.json
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from ctx_eval import SPEC, features, load, render_variant, variant_data
from improve_lib import fit_predict
from jevmon_lab.manifest import LABELS
from settle_eval import RECIPES


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("train", "old", "re", "fresh"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--rows-out", type=Path, required=True)
    ap.add_argument("--preds-out", type=Path, required=True)
    args = ap.parse_args()
    data = load(args)
    d, texts = variant_data(data, "A")
    tr = data.idx("train")
    rows = []
    for r, t in zip(data.rows, texts, strict=True):
        assert t == render_variant(r, "A")
        rows.append(
            {
                "set": r["cap"],
                "label": r["label"],
                "family": r["family"],
                "scenario_id": r["scenario_id"],
                "context": t,
                "context_c": render_variant(r, "C"),
            }
        )
    args.rows_out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    out = {"labels": list(LABELS), "pred": {}, "proba": {}}
    x = features(texts)
    p = fit_predict(SPEC, x[tr], d.y[tr], d.key[tr], x)
    out["pred"]["laya"], out["proba"]["laya"] = p.argmax(1).tolist(), p.tolist()
    _src, var, spec, _mid, _kind, _role = RECIPES["foundation-sec-8b"]
    store = pickle.loads(Path("cache/bb-foundation-sec-8b-512.pkl").read_bytes())
    xf = np.stack([store[t][var] for t in texts]).astype(np.float32)
    p = fit_predict(spec, xf[tr], d.y[tr], d.key[tr], xf)
    out["pred"]["foundation-sec"], out["proba"]["foundation-sec"] = p.argmax(1).tolist(), p.tolist()
    args.preds_out.write_text(json.dumps(out))
    print(len(rows), "rows;", {k: len(v) for k, v in out["pred"].items()})


if __name__ == "__main__":
    main()

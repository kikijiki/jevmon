"""Frozen-feature backbones with a logistic head: train-CV sweep, then held-out once each.

    python scripts/backbone_eval.py cv    DIR... --names a,b,c [--project jevmon-backbones]
    python scripts/backbone_eval.py final DIR... --names a,b,c [--seeds 3]

Features come from `backbone_embed.py` (`cache/bb-NAME-512.pkl`; `laya` is the frozen Laya cache
`cache/emb-english-mean-512.pkl`). `cv` scores every (feature variant, wd, preprocessing) cell
with family-grouped CV on the train families only and writes `cache/bb-cv.json`. `final` takes
the CV-best cell of each backbone from that file (never looks at held-out to choose), fits on all
train rows and scores the held-out families once, with a family-cluster bootstrap of the held-out
macro-F1, a paired bootstrap difference against the Laya recipe head, and 20 train-row
resamples (as E12). The split and labels come from `improve_lib` unchanged.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from improve_lib import (
    LABELS,
    Spec,
    cv_scores,
    fit_predict,
    load_rows,
    make_data,
    render,
    to_rows,
)
from jevmon_lab import metrics
from jevmon_lab.runlog import RunLog

CV_JSON = Path("cache/bb-cv.json")
LAYA_SPEC = Spec("logreg", wd=0.01, balanced=True)


def features(name: str, texts: list[str]) -> dict[str, np.ndarray]:
    if name == "laya":
        store = pickle.loads(Path("cache/emb-english-mean-512.pkl").read_bytes())
        return {"mean": np.stack([store[t] for t in texts]).astype(np.float32)}
    store = pickle.loads(Path(f"cache/bb-{name}-512.pkl").read_bytes())
    variants = next(iter(store.values())).keys()
    return {v: np.stack([store[t][v] for t in texts]).astype(np.float32) for v in variants}


def grid() -> list[Spec]:
    out = []
    for wd in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
        out.append(Spec("logreg", wd=wd, balanced=True))
        out.append(Spec("logreg", wd=wd, balanced=True, norm=True))
        out.append(Spec("logreg", wd=wd, balanced=True, standardize=True))
    return out


def macro_f1(labels: np.ndarray, pred: np.ndarray) -> float:
    f1s = []
    for c in range(len(LABELS)):
        tp = float(((pred == c) & (labels == c)).sum())
        fp = float(((pred == c) & (labels != c)).sum())
        fn = float(((pred != c) & (labels == c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def cluster_boot(y, pred, fam, n=2000, seed=0, ref=None):
    """Family-cluster bootstrap of macro-F1 (and the paired difference to `ref` predictions)."""
    rng = np.random.default_rng(seed)
    fams = sorted(set(fam))
    members = {f: np.flatnonzero(fam == f) for f in fams}
    vals, diffs = [], []
    for _ in range(n):
        pick = rng.choice(len(fams), size=len(fams))
        idx = np.concatenate([members[fams[i]] for i in pick])
        a = macro_f1(y[idx], pred[idx])
        vals.append(a)
        if ref is not None:
            diffs.append(a - macro_f1(y[idx], ref[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    out = {"lo": float(lo), "hi": float(hi), "sd": float(np.std(vals))}
    if ref is not None:
        d = np.asarray(diffs)
        out |= {
            "diff_mean": float(d.mean()),
            "diff_lo": float(np.percentile(d, 2.5)),
            "diff_hi": float(np.percentile(d, 97.5)),
            "p_better": float((d > 0).mean()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("cv", "final"))
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--names", required=True)
    ap.add_argument("--project", default="jevmon-backbones")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tag", default="b13")
    ap.add_argument("--runs", type=Path, default=Path(os.environ.get("JEVMON_RUNS", "runs")))
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    rows = []
    for d in args.dirs:
        rows += load_rows(d / "dataset.jsonl")
    data, texts = make_data(rows, lambda r: render(r, "shipped"))
    tr, te = data.idx("train"), data.idx("heldout")
    print(f"train {len(tr)} heldout {len(te)} families {len(set(data.family[tr]))}", flush=True)
    names = args.names.split(",")
    cv = json.loads(CV_JSON.read_text()) if CV_JSON.exists() else {}

    if args.mode == "cv":
        for name in names:
            feats = features(name, texts)
            best = None
            for var, x in feats.items():
                for spec in grid():
                    r = cv_scores(spec, x, data, tr, folds=args.folds, repeats=args.repeats)
                    cell = f"{var}|{spec.name()}"
                    cv.setdefault(name, {})[cell] = r
                    if best is None or r["unique_f1"] > best[1]["unique_f1"]:
                        best = (cell, r)
                CV_JSON.write_text(json.dumps(cv))
            cell, r = best
            print(
                f"{name:20s} best {cell:40s} uniq {r['unique_f1']:.3f}+-{r['unique_f1_sd']:.3f} "
                f"rows {r['rows_f1']:.3f} rec {[round(r['unique_recall'][k], 2) for k in LABELS]}",
                flush=True,
            )
            run = RunLog.start(
                args.project,
                f"{args.tag}-cv-{name}-{time.strftime('%H%M%S')}",
                {"backbone": name, "best": cell},
                directory=args.runs,
            )
            run.log(0, **{f"cv/{k}": v for k, v in r.items() if isinstance(v, int | float)})
            run.finish()
        return

    # ---- final: held-out once per candidate, chosen by train-CV only
    def best_cell(name: str) -> tuple[str, Spec]:
        cell = max(cv[name], key=lambda c: cv[name][c]["unique_f1"])
        var, spec_name = cell.split("|", 1)
        spec = next(s for s in grid() if s.name() == spec_name)
        return var, spec

    xl = features("laya", texts)["mean"]
    ref_pred = fit_predict(LAYA_SPEC, xl[tr], data.y[tr], data.key[tr], xl[te]).argmax(1)
    ref_f1 = macro_f1(data.y[te], ref_pred)
    print(f"laya recipe head held-out macro-F1 {ref_f1:.3f}", flush=True)
    out = {}
    for name in names:
        if name == "laya":
            var, spec = "mean", LAYA_SPEC
        else:
            var, spec = best_cell(name)
        x = features(name, texts)[var]
        proba = fit_predict(spec, x[tr], data.y[tr], data.key[tr], x[te])
        pred = proba.argmax(1)
        sc = metrics.score(to_rows(data, te, pred))
        boot = cluster_boot(data.y[te], pred, data.family[te], ref=ref_pred)
        rng = np.random.default_rng(7)
        tr_f1 = []
        for _ in range(20):
            sub = rng.choice(tr, size=len(tr), replace=True)
            p2 = fit_predict(spec, x[sub], data.y[sub], data.key[sub], x[te]).argmax(1)
            tr_f1.append(macro_f1(data.y[te], p2))
        pc = {k: (v["precision"], v["recall"]) for k, v in sc["per_class"].items()}
        cvr = cv[name][f"{var}|{spec.name()}"] if name != "laya" else None
        out[name] = {
            "cell": f"{var}|{spec.name()}",
            "cv_unique_f1": cvr["unique_f1"] if cvr else None,
            "macro_f1": sc["macro_f1"],
            "accuracy": sc["accuracy"],
            "per_class_pr": pc,
            "boot": boot,
            "train_boot_mean": float(np.mean(tr_f1)),
            "train_boot_sd": float(np.std(tr_f1)),
        }
        print(
            f"{name:20s} {var}|{spec.name():36s} cv {out[name]['cv_unique_f1']} "
            f"held-out F1 {sc['macro_f1']:.3f} acc {sc['accuracy']:.3f} "
            f"CI [{boot['lo']:.2f},{boot['hi']:.2f}] diff vs laya {boot['diff_mean']:+.3f} "
            f"[{boot['diff_lo']:+.2f},{boot['diff_hi']:+.2f}] p>{boot['p_better']:.2f} "
            f"trainboot {np.mean(tr_f1):.3f}+-{np.std(tr_f1):.3f} "
            + " ".join(f"{k}:{p:.2f}/{r:.2f}" for k, (p, r) in pc.items()),
            flush=True,
        )
        run = RunLog.start(
            args.project,
            f"{args.tag}-final-{name}-{time.strftime('%H%M%S')}",
            {"backbone": name, "cell": out[name]["cell"]},
            directory=args.runs,
        )
        run.log(
            0,
            **{
                "heldout/macro_f1": sc["macro_f1"],
                "heldout/accuracy": sc["accuracy"],
                "heldout/ci_lo": boot["lo"],
                "heldout/ci_hi": boot["hi"],
                "heldout/diff_vs_laya": boot["diff_mean"],
                "heldout/trainboot_sd": float(np.std(tr_f1)),
            },
            **{f"heldout/recall_{k}": r for k, (_, r) in pc.items()},
            **{f"heldout/precision_{k}": p for k, (p, _) in pc.items()},
        )
        run.finish()
    Path("cache/bb-final.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

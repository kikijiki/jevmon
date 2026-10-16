"""Learning curves over the number of training families, family-grouped CV, train families only.

    python scripts/learning_curves.py lab-data-3c-r6 lab-data-3c-r7 --cache cache \
        --out curves.json [--runs DIR]

For every n a random pool of n train families is drawn (`--repeats` times, fixed seeds), cut into
4 family-grouped folds, and each method is scored on the pooled out-of-fold rows. A model in a
fold therefore sees about 0.75 n families. The same pools and folds are used for every method.
Embeddings come from the on-disk cache (frozen Laya, shipped context format, mean pooling) and
are computed only for texts the cache has not seen. Held-out families are never read: the
scores use `Data.idx("train")` rows only. Results go to a JSON file and, one run per method, to
Trackio project `jevmon-curves3` with step = n (per-class F1 metrics `class/0_safe`,
`class/1_warn`, `class/2_crit`).
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

import improve_lib as lib
from improve_lib import LABELS, Spec, family_folds, fit_predict, load_rows, make_data, render
from jevmon_lab import metrics
from jevmon_lab.runlog import RunLog

NS = (6, 10, 15, 20, 25, 30, 36, 42, 49)
METHODS = {
    "logistic head": Spec("logreg", wd=0.01, balanced=True),
    "mlp head": Spec("mlp", balanced=True),
    "kNN": Spec("knn", k=5, balanced=True),
    "prototype": Spec("proto", temp=0.05, balanced=True),
}


def load_data(dirs: list[Path]):
    rows = []
    for d in dirs:
        rows += load_rows(d if d.suffix == ".jsonl" else d / "dataset.jsonl")
    return make_data(rows, lambda r: render(r, "shipped"))


def zero_shot(data, texts, path: Path) -> np.ndarray:
    """Zero-shot verdict (class index) per row, cached by context text."""
    store = json.loads(path.read_text()) if path.exists() else {}
    ctx = [r["context"] for r in data.rows]
    missing = sorted({c for c in ctx if c not in store})
    if missing:
        from jevmon.config import load_config
        from jevmon.monitor import build_decider

        decider = build_decider(load_config())
        decider.load()
        for c, d in zip(missing, decider.decide_batch(missing), strict=True):
            store[c] = d.label
        path.write_text(json.dumps(store))
    return np.asarray([LABELS.index(store[c]) for c in ctx])


def summarise(data, idx, pred) -> dict:
    rows = lib.to_rows(data, idx, pred)
    sc = metrics.score(rows)
    out = {"macro_f1": sc["macro_f1"], "accuracy": sc["accuracy"]}
    for lab in LABELS:
        out[lab] = sc["per_class"].get(lab, {}).get("f1", 0.0)
    return out


def curves(data, x, methods: dict[str, Spec], fixed: dict[str, np.ndarray], repeats: int) -> dict:
    """{method: {n: {metric: [per-repeat values]}}}; `fixed` maps a name to per-row predictions."""
    train = data.idx("train")
    fams = sorted(set(data.family[train]))
    res: dict = {m: {} for m in [*methods, *fixed]}
    for n in sorted({*(n for n in NS if n <= len(fams)), len(fams)}):
        for rep in range(repeats):
            rng = np.random.default_rng(1000 * n + rep)
            pool = set(rng.choice(fams, size=n, replace=False))
            idx = train[np.isin(data.family[train], list(pool))]
            parts = family_folds(data, idx, 4, rep)
            pooled = np.concatenate(parts)
            preds = {m: np.zeros(len(data.y), dtype=int) for m in methods}
            for val in parts:
                tr = np.setdiff1d(idx, val)
                for m, spec in methods.items():
                    p = fit_predict(spec, x[tr], data.y[tr], data.key[tr], x[val])
                    preds[m][val] = p.argmax(1)
            for m in methods:
                s = summarise(data, pooled, preds[m][pooled])
                for k, v in s.items():
                    res[m].setdefault(n, {}).setdefault(k, []).append(v)
            for m, pr in fixed.items():
                s = summarise(data, pooled, pr[pooled])
                for k, v in s.items():
                    res[m].setdefault(n, {}).setdefault(k, []).append(v)
        print(
            f"n={n}: " + " ".join(f"{m} {np.mean(res[m][n]['macro_f1']):.3f}" for m in res),
            flush=True,
        )
    return res


def embeddings(texts, cache_dir: Path, *, only_cached: bool = False):
    lib.CACHE_DIR = cache_dir
    path = cache_dir / "emb-english-mean-512.pkl"
    if only_cached:
        store = pickle.loads(path.read_bytes())
        if any(t not in store for t in texts):
            return None
    return lib.EmbeddingCache(lib.laya_agent).get(texts, "mean")


def log_runs(
    project: str, res: dict, runs: Path, config: dict, *, per_class_of: str | None
) -> None:
    for m, by_n in res.items():
        run = RunLog.start(project, m, {**config, "method": m}, directory=runs)
        for n, v in by_n.items():
            run.log(
                n,
                **{
                    "curve/macro_f1": float(np.mean(v["macro_f1"])),
                    "curve/macro_f1_sd": float(np.std(v["macro_f1"])),
                    "curve/accuracy": float(np.mean(v["accuracy"])),
                },
            )
        run.finish()
    if per_class_of:
        run = RunLog.start(
            project, "head per-class", {**config, "method": per_class_of}, directory=runs
        )
        for n, v in res[per_class_of].items():
            run.log(
                n, **{f"class/{i}_{lab}": float(np.mean(v[lab])) for i, lab in enumerate(LABELS)}
            )
        run.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("curves.json"))
    ap.add_argument("--project", default="jevmon-curves3")
    ap.add_argument("--runs", type=Path, default=Path(os.environ.get("TRACKIO_DIR", "runs")))
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    data, texts = load_data(args.dirs)
    x = embeddings(texts, args.cache)
    fixed = {
        "zero-shot": zero_shot(data, texts, args.cache / "zero-shot-verdicts.json"),
        "always safe": np.zeros(len(data.y), dtype=int),
    }
    res = curves(data, x, METHODS, fixed, args.repeats)
    config = {
        "data": [str(d) for d in args.dirs],
        "folds": 4,
        "repeats": args.repeats,
        "families": sorted(NS),
        "n_train_rows": len(data.idx("train")),
        "labels": "three-class",
    }
    out = {"three_class": res}

    args.out.write_text(json.dumps(out, indent=1))
    if not args.no_track:
        log_runs(args.project, res, args.runs, config, per_class_of="logistic head")


if __name__ == "__main__":
    main()

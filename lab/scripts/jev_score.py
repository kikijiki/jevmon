"""E18: score hosted-Jev probe outputs against local heads. Descriptive, nothing pre-registered.

    python scripts/jev_score.py --rows rows.jsonl --heldout out_heldout.json --train out_train.json \
        --heads heads.json --out results.json

`rows.jsonl` and `heads.json` come from jev_heads.py (all E15 rows, train first), the two probe
outputs from jev_probe.py. Macro-F1 over safe/warn/crit, family-cluster bootstrap (10000
resamples, seed 14, the E14/E15 procedure, paired on the same resamples), per-class precision and
recall, safe recall, reliability of Jev's confidence, and a logistic head fitted on Jev's
probabilities with the train rows.
"""

import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ("safe", "warn", "crit")
SETS = ("old", "re", "fresh", "new", "pool")
BOOT = 10000
SEED = 14


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int = 3) -> float:
    f1s = []
    for c in range(k):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def per_class(y: np.ndarray, pred: np.ndarray, k: int = 3) -> list[tuple[float, float]]:
    """[(precision, recall)] per class; 0.0 where a class is never predicted or absent."""
    out = []
    for c in range(k):
        tp = float(((pred == c) & (y == c)).sum())
        p = tp / max(float((pred == c).sum()), 1.0)
        r = tp / max(float((y == c).sum()), 1.0)
        out.append((p, r))
    return out


def set_index(sets: np.ndarray, name: str) -> np.ndarray:
    if name == "new":
        return np.flatnonzero(np.isin(sets, ("re", "fresh")))
    if name == "pool":
        return np.flatnonzero(np.isin(sets, ("old", "re", "fresh")))
    return np.flatnonzero(sets == name)


def boot_draws(fam: np.ndarray, n: int = BOOT, seed: int = SEED) -> list[np.ndarray]:
    """Row-index draws for a family-cluster bootstrap (families resampled, all rows kept)."""
    rng = np.random.default_rng(seed)
    fams = sorted(set(fam.tolist()))
    members = [np.flatnonzero(fam == f) for f in fams]
    return [
        np.concatenate([members[i] for i in rng.integers(0, len(fams), len(fams))])
        for _ in range(n)
    ]


def boot_paired(y, pred_a, pred_b, draws) -> dict:
    """CI of macro-F1 for `a`, and CI of a - b on the same resamples."""
    va = np.asarray([macro_f1(y[d], pred_a[d]) for d in draws])
    vb = np.asarray([macro_f1(y[d], pred_b[d]) for d in draws])
    diff = va - vb
    return {
        "lo": float(np.percentile(va, 2.5)),
        "hi": float(np.percentile(va, 97.5)),
        "diff_lo": float(np.percentile(diff, 2.5)),
        "diff_hi": float(np.percentile(diff, 97.5)),
        "p_better": float((diff > 0).mean()),
    }


def probs(res: dict, i: int, variant: str) -> np.ndarray | None:
    """Choice probabilities (safe, warn, crit) of row i, or None if the call failed."""
    r = res.get(f"{i}|{variant}")
    if not r or "error" in r:
        return None
    a = r["response"]["answers"]["verdict"]["probabilities"]
    return np.asarray([a[c] for c in CLASSES], dtype=float)


def binary_features(res: dict, i: int, variant: str) -> np.ndarray | None:
    r = res.get(f"{i}|{variant}")
    if not r or "error" in r:
        return None
    a = r["response"]["answers"]
    return np.asarray([a[q]["noul"] for q in ("safe_to_run", "needs_review", "hostile")], float)


def feature_matrix(out: dict, variant: str) -> np.ndarray:
    fn = binary_features if variant.endswith("binary") else probs
    rows = [fn(out["res"], i, variant) for i in range(len(out["rows"]))]
    if any(r is None for r in rows):
        raise ValueError(f"missing responses for {variant}")
    return np.stack(rows)


def confidences(out: dict, variant: str) -> np.ndarray:
    return np.asarray(
        [
            out["res"][f"{i}|{variant}"]["response"]["answers"]["verdict"]["confidence"]
            for i in range(len(out["rows"]))
        ],
        float,
    )


def calibration(conf: np.ndarray, correct: np.ndarray, bins: int = 5) -> dict:
    """Equal-count reliability bins and expected calibration error."""
    order = np.argsort(conf, kind="stable")
    parts = np.array_split(order, bins)
    table = []
    ece = 0.0
    for p in parts:
        if not len(p):
            continue
        c, a = float(conf[p].mean()), float(correct[p].mean())
        table.append({"n": len(p), "mean_conf": c, "accuracy": a})
        ece += len(p) / len(conf) * abs(c - a)
    return {"bins": table, "ece": float(ece)}


def fit_logreg(x: np.ndarray, y: np.ndarray, c: float = 1.0, k: int = 3) -> np.ndarray:
    """Multinomial logistic regression (L2, weight 1/(2c) on the weights, unpenalised intercept).

    Returns a predict function's parameters packed as (k, d+1).
    """
    import torch

    xt, yt = torch.tensor(x, dtype=torch.float64), torch.tensor(y)
    w = torch.zeros(k, x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        nll = torch.nn.functional.cross_entropy(xt @ w.T + b, yt, reduction="sum")
        loss = c * nll + 0.5 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.cat([w.detach(), b.detach()[:, None]], dim=1).numpy()


def predict_logreg(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (x @ params[:, :-1].T + params[:, -1]).argmax(1)


def labels_of(rows: list[dict]) -> np.ndarray:
    return np.asarray([CLASSES.index(r["label"]) for r in rows])


def evaluate(preds: dict[str, np.ndarray], y, fam, sets, refs: tuple[str, ...]) -> dict:
    """{set: {model: stats}} with paired bootstrap against each model in `refs`."""
    table: dict = {}
    for s in SETS:
        idx = set_index(sets, s)
        draws = boot_draws(fam[idx])
        yy = y[idx]
        table[s] = {"rows": len(idx), "families": len(set(fam[idx].tolist())), "models": {}}
        for name, p in preds.items():
            pp = p[idx]
            pc = per_class(yy, pp)
            entry = {
                "macro_f1": macro_f1(yy, pp),
                "accuracy": float((pp == yy).mean()),
                "per_class": {c: list(pc[j]) for j, c in enumerate(CLASSES)},
                "safe_recall": pc[0][1],
                "vs": {},
            }
            for ref in refs:
                if ref != name:
                    entry["vs"][ref] = boot_paired(yy, pp, preds[ref][idx], draws)
            table[s]["models"][name] = entry
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--heads", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["A_flat_choice", "B_lines_choice"])
    ap.add_argument("--extra", nargs="*", default=["D_raw_flat_choice", "E_json_choice"])
    args = ap.parse_args()
    allrows = [json.loads(ln) for ln in args.rows.read_text().splitlines() if ln]
    held = [i for i, r in enumerate(allrows) if r["set"] != "train"]
    out = json.loads(args.heldout.read_text())
    train = json.loads(args.train.read_text())
    assert [r["context"] for r in out["rows"]] == [allrows[i]["context"] for i in held]
    heads = json.loads(args.heads.read_text())
    y = labels_of(out["rows"])
    fam = np.asarray([r["family"] for r in out["rows"]])
    sets = np.asarray([r["set"] for r in out["rows"]])
    preds = {
        "laya head": np.asarray(heads["pred"]["laya"])[held],
        "foundation-sec head": np.asarray(heads["pred"]["foundation-sec"])[held],
    }
    for v in (*args.variants, *args.extra):
        preds[f"jev {v}"] = feature_matrix(out, v).argmax(1)
    ytr = labels_of(train["rows"])
    for v in ("A_flat_choice", "B_lines_choice", "C_flat_binary"):
        params = fit_logreg(feature_matrix(train, v), ytr)
        preds[f"jev {v} + head"] = predict_logreg(params, feature_matrix(out, v))
    result = {"table": evaluate(preds, y, fam, sets, ("laya head", "foundation-sec head"))}
    cal = {}
    for v in args.variants:
        x, conf = feature_matrix(out, v), confidences(out, v)
        correct = (x.argmax(1) == y).astype(float)
        cal[v] = {
            "confidence_field": calibration(conf, correct),
            "max_probability": calibration(x.max(1), correct),
            "accuracy": float(correct.mean()),
        }
    result["calibration"] = cal
    args.out.write_text(json.dumps(result, indent=1))
    for s in SETS:
        t = result["table"][s]
        print(f"\n{s} rows {t['rows']} families {t['families']}")
        for name, e in t["models"].items():
            d = " ".join(
                f"vs {k.split()[0]} {v['diff_lo']:+.2f}..{v['diff_hi']:+.2f}"
                for k, v in e["vs"].items()
            )
            print(f"  {name:32s} F1 {e['macro_f1']:.3f} safeR {e['safe_recall']:.2f} {d}")


if __name__ == "__main__":
    main()

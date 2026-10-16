"""E15: what the model sees (redacted or raw argv, with or without the argv template).

    python scripts/ctx_eval.py show   --capture DIR...          # render + overlap check, no model
    python scripts/ctx_eval.py embed  --train D --old D --re D --fresh D
    python scripts/ctx_eval.py cv     --train D --old D --re D --fresh D     # train-CV only
    python scripts/ctx_eval.py score  --train D --old D --re D --fresh D     # held-out, once
    python scripts/ctx_eval.py reference --old-train D... --old-old D... --re D --fresh D

Variants (pre-registered in research-log E15; all use the shipped Laya head recipe):

    A  header exe template argv   redacted argv   label `argv(redacted):`   (shipped)
    B  header exe argv            redacted argv
    C  header exe argv            raw argv        label `argv:`
    D  header exe template argv   raw argv, template of the raw argv

The captures carry `context` (state built from the redacted argv) and `context_raw` (state built
from the unmasked, sanitized and bounded argv by the lab-only RawExecSource); both describe the
same process. `score` refuses to run twice (it writes cache/ctx-final.json) unless --force.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from backbone_eval import macro_f1
from improve_lib import (
    EmbeddingCache,
    Spec,
    cv_scores,
    fit_predict,
    laya_agent,
    make_data,
    to_rows,
)
from jevmon.detect.context import ContextSettings, parse_exec_state, render_segments
from jevmon_lab import metrics
from jevmon_lab.manifest import LABELS, split_of
from jevmon_lab.runlog import RunLog

FINAL_JSON = Path("cache/ctx-final.json")
BOOT = 10000
SEED = 14
SPEC = Spec("logreg", wd=0.01, balanced=True)
VARIANTS = {
    "A": (("header", "exe", "template", "argv"), False),
    "B": (("header", "exe", "argv"), False),
    "C": (("header", "exe", "argv"), True),
    "D": (("header", "exe", "template", "argv"), True),
}
RED_LABEL = " argv(redacted): "


def render_variant(row: dict, name: str) -> str:
    order, raw = VARIANTS[name]
    state = row["_raw_argv"] if raw else row["_redacted"]
    text = render_segments(parse_exec_state(state), ContextSettings(order=order, model_max=1000))
    if raw:
        text = text.replace(RED_LABEL, " argv: ", 1)
    return text


def load_capture(path: Path) -> list[dict]:
    """Labelled rows of a capture dir with both the redacted and the raw state attached."""
    rows = [json.loads(ln) for ln in (path / "dataset.jsonl").read_text().splitlines() if ln]
    raw = [json.loads(ln) for ln in (path / "contexts.jsonl").read_text().splitlines() if ln]
    assert len(rows) == len(raw), path
    out = []
    for row, c in zip(rows, raw, strict=True):
        if not row.get("label"):
            continue
        row["_redacted"] = c["context"]
        row["_raw_argv"] = c.get("context_raw") or ""
        row["split"] = split_of(row["family"])
        assert row.get("designated") in (True, None), row["scenario_id"]
        out.append(row)
    return out


def load(args):
    rows = []
    for r in load_capture(args.train):
        if r["split"] == "train":
            r["cap"] = "train"
            rows.append(r)
    train_fams = {r["family"] for r in rows}
    for cap, d in (("old", args.old), ("re", args.re), ("fresh", args.fresh)):
        for r in load_capture(d):
            if r["split"] != "heldout":
                continue
            assert r["family"] not in train_fams, r["family"]
            r["cap"] = cap
            rows.append(r)
    assert all(r["_raw_argv"] for r in rows), "capture without context_raw"
    data, _ = make_data(rows, lambda r: r["_redacted"])
    data.extra["cap"] = np.asarray([r["cap"] for r in rows])
    return data


def variant_data(data, name: str):
    texts = [render_variant(r, name) for r in data.rows]
    keyed = type(data)(
        rows=data.rows,
        split=data.split,
        family=data.family,
        y=data.y,
        key=np.asarray([f"{t}\x00{r['label']}" for t, r in zip(texts, data.rows, strict=True)]),
        extra=data.extra,
    )
    return keyed, texts


def sets_of(data):
    cap = data.extra["cap"]
    return {
        "old": np.flatnonzero(cap == "old"),
        "re": np.flatnonzero(cap == "re"),
        "fresh": np.flatnonzero(cap == "fresh"),
        "new": np.flatnonzero(np.isin(cap, ("re", "fresh"))),
        "pool": np.flatnonzero(cap != "train"),
    }


def safe_recall(data, idx, pred):
    m = data.y[idx] == LABELS.index("safe")
    return float((pred[idx][m] == LABELS.index("safe")).mean())


def boot(data, preds, idx, ref="A", n=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    fam = data.family[idx]
    fams = sorted(set(fam))
    members = [np.flatnonzero(fam == f) for f in fams]
    draws = [
        np.concatenate([members[i] for i in rng.integers(0, len(fams), len(fams))])
        for _ in range(n)
    ]
    y = data.y[idx]
    base = np.asarray([macro_f1(y[d], preds[ref][idx][d]) for d in draws])
    out = {}
    for name, pred in preds.items():
        p = pred[idx]
        vals = np.asarray([macro_f1(y[d], p[d]) for d in draws])
        e = {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}
        if name != ref:
            diffs = vals - base
            e |= {
                "diff_lo": float(np.percentile(diffs, 2.5)),
                "diff_hi": float(np.percentile(diffs, 97.5)),
                "p_better": float((diffs > 0).mean()),
            }
        out[name] = e
    return out


def features(texts):
    return EmbeddingCache(laya_agent).get(texts, "mean")


def cmd_show(args):
    data = load(args)
    for name in VARIANTS:
        _, texts = variant_data(data, name)
        print(f"{name}: {len(set(texts))} unique texts of {len(texts)}")
    differs = sum(r["_redacted"] != r["_raw_argv"] for r in data.rows)
    print(f"rows where raw state differs from redacted: {differs}/{len(data.rows)}")
    for r in data.rows:
        if r["_redacted"] != r["_raw_argv"]:
            print("  A:", render_variant(r, "A")[:200])
            print("  D:", render_variant(r, "D")[:200])
            break


def cmd_embed(args):
    data = load(args)
    for name in VARIANTS:
        features(variant_data(data, name)[1])


def cmd_cv(args):
    data = load(args)
    tr = data.idx("train")
    out = {}
    for name in VARIANTS:
        d, texts = variant_data(data, name)
        x = features(texts)
        sc = cv_scores(SPEC, x, d, tr, folds=4, repeats=5, base_seed=0)
        out[name] = sc
        print(
            f"{name}: train-CV unique macro-F1 {sc['unique_f1']:.3f} +- {sc['unique_f1_sd']:.3f}"
            f"  rows {sc['rows_f1']:.3f}",
            flush=True,
        )
    Path("cache/ctx-cv.json").write_text(json.dumps(out, indent=1))


def cmd_score(args):
    if FINAL_JSON.exists() and not args.force:
        raise SystemExit(f"{FINAL_JSON} exists: held-out is scored once (use --force)")
    data = load(args)
    cv = json.loads(Path("cache/ctx-cv.json").read_text())
    tr = data.idx("train")
    sets = sets_of(data)
    for k, v in sets.items():
        print(f"  {k:6s} rows {len(v):4d} families {len(set(data.family[v])):3d}")
    preds = {}
    for name in VARIANTS:
        d, texts = variant_data(data, name)
        x = features(texts)
        preds[name] = fit_predict(SPEC, x[tr], d.y[tr], d.key[tr], x).argmax(1)
    table = {}
    for s, idx in sets.items():
        b = boot(data, preds, idx)
        table[s] = {}
        for name in VARIANTS:
            sc = metrics.score(to_rows(data, idx, preds[name][idx]))
            table[s][name] = {
                "macro_f1": sc["macro_f1"],
                "accuracy": sc["accuracy"],
                "per_class": {k: [v["precision"], v["recall"]] for k, v in sc["per_class"].items()},
                "safe_recall": safe_recall(data, idx, preds[name]),
                "boot": b[name],
                "rows": len(idx),
                "families": len(set(data.family[idx])),
            }
    dec = decide(table, cv)
    FINAL_JSON.write_text(json.dumps({"table": table, "cv": cv, "decision": dec}, indent=1))
    for s in sets:
        for name in VARIANTS:
            t = table[s][name]
            bb = t["boot"]
            d = (
                f" diff [{bb['diff_lo']:+.3f},{bb['diff_hi']:+.3f}] p>{bb['p_better']:.2f}"
                if "diff_lo" in bb
                else ""
            )
            pc = " ".join(f"{k}:{p:.2f}/{r:.2f}" for k, (p, r) in t["per_class"].items())
            print(
                f"{s:6s} {name} F1 {t['macro_f1']:.3f} [{bb['lo']:.2f},{bb['hi']:.2f}]"
                f" acc {t['accuracy']:.3f}{d} safeR {t['safe_recall']:.2f} | {pc}"
            )
    print(json.dumps(dec, indent=1))
    for name in VARIANTS:
        run = RunLog.start(
            args.project,
            f"e15-final-{name}-{time.strftime('%H%M%S')}",
            {"variant": name, "recipe": SPEC.name(), "segments": ",".join(VARIANTS[name][0])},
            directory=args.runs,
        )
        flat = {"train_cv_unique_f1": cv[name]["unique_f1"]}
        for s in sets:
            t = table[s][name]
            flat[f"{s}/macro_f1"] = t["macro_f1"]
            flat[f"{s}/safe_recall"] = t["safe_recall"]
            flat[f"{s}/ci_lo"] = t["boot"]["lo"]
            flat[f"{s}/ci_hi"] = t["boot"]["hi"]
        run.log(0, **flat)
        run.finish()


def decide(table, cv):
    """Pre-registered rule (research-log E15)."""
    out = {}
    for x in ("B", "C", "D"):
        pool = table["pool"]
        b = pool[x]["boot"]
        diff = pool[x]["macro_f1"] - pool["A"]["macro_f1"]
        cv_diff = cv[x]["unique_f1"] - cv["A"]["unique_f1"]
        drop = pool["A"]["safe_recall"] - pool[x]["safe_recall"]
        rule1 = diff > 0 and b["diff_lo"] > 0
        rule2 = cv_diff >= 0.03 and diff >= 0.03 and drop <= 0.03
        out[x] = {
            "pool_diff": diff,
            "pool_diff_ci": [b["diff_lo"], b["diff_hi"]],
            "cv_diff": cv_diff,
            "safe_recall_drop_pool": drop,
            "rule1": bool(rule1),
            "rule2": bool(rule2),
            "adopt": bool(rule1 or rule2),
        }
    passing = [x for x in out if out[x]["adopt"]]
    out["winner"] = max(passing, key=lambda x: cv[x]["unique_f1"]) if passing else "A"
    out["b_ge_a"] = bool(
        cv["B"]["unique_f1"] >= cv["A"]["unique_f1"]
        and table["pool"]["B"]["macro_f1"] >= table["pool"]["A"]["macro_f1"]
    )
    return out


def cmd_reference(args):
    """Variant A on the old (E12/E14) captures: reproduces the E14 laya row, not a decision."""
    rows = []
    for d in args.old_train + args.old_old:
        for r in load_ref(d):
            r["cap"] = "train" if r["split"] == "train" else "old"
            rows.append(r)
    for cap, d in (("re", args.re), ("fresh", args.fresh)):
        for r in load_ref(d):
            r["cap"] = cap
            rows.append(r)
    data, texts = make_data(
        rows,
        lambda r: render_segments(
            parse_exec_state(r["_redacted"]),
            ContextSettings(order=VARIANTS["A"][0], model_max=1000),
        ),
    )
    data.extra["cap"] = np.asarray([r["cap"] for r in rows])
    x = features(texts)
    tr = np.flatnonzero(data.extra["cap"] == "train")
    pred = fit_predict(SPEC, x[tr], data.y[tr], data.key[tr], x).argmax(1)
    for s, idx in sets_of(data).items():
        sc = metrics.score(to_rows(data, idx, pred[idx]))
        print(f"reference A on old captures, {s}: rows {len(idx)} macro-F1 {sc['macro_f1']:.3f}")


def load_ref(path: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in (path / "dataset.jsonl").read_text().splitlines() if ln]
    ctx = [json.loads(ln) for ln in (path / "contexts.jsonl").read_text().splitlines() if ln]
    out = []
    for row, c in zip(rows, ctx, strict=True):
        if not row.get("label"):
            continue
        row["label"] = {"routine": "safe", "unusual": "warn", "suspicious": "crit"}.get(
            row["label"], row["label"]
        )
        row["_redacted"] = c["context"]
        row["split"] = split_of(row["family"])
        out.append(row)
    return out


def cmd_stage1(args):
    """Stage 1: score each variant's text with models that already exist, as they are.

    Zero-shot Laya (shipped category descriptions, no head), the shipped head lab-head-v4, and the
    frozen-backbone heads of E14 (fitted on the OLD capture's train rows in the old format, the
    recipes of settle_eval.RECIPES) applied to the variant text. A head trained on the old format
    is out-of-distribution on new text, so a drop there is not proof the information is useless;
    zero-shot is the fairer information test. Nothing is fitted on the new capture here.
    """
    from dataclasses import replace as _replace

    from jevmon.config import load_config
    from jevmon.laya.head import TunedHead, resolve_head
    from jevmon.monitor import build_decider
    from settle_eval import RECIPES

    data = load(args)
    sets = {"train": data.idx("train"), **sets_of(data)}
    old = []
    for d in args.old_train:
        for r in load_ref(d):
            if r["split"] == "train":
                old.append(r)
    old_data, old_texts = make_data(
        old,
        lambda r: render_segments(
            parse_exec_state(r["_redacted"]),
            ContextSettings(order=VARIANTS["A"][0], model_max=1000),
        ),
    )
    result: dict[str, dict] = {}

    def record(model: str, variant: str, pred: np.ndarray) -> None:
        for s, idx in sets.items():
            sc = metrics.score(to_rows(data, idx, pred[idx]))
            result.setdefault(model, {}).setdefault(variant, {})[s] = sc["macro_f1"]

    texts_of = {v: variant_data(data, v)[1] for v in VARIANTS}
    # shipped head (lab-head-v4), through the same embedder the daemon uses
    head = TunedHead.load(resolve_head("lab-head-v4"))
    for v, texts in texts_of.items():
        p = head.probabilities(features(texts)).argmax(1)
        record("shipped head v4", v, p)
    # zero-shot Laya, shipped descriptions
    cfg = load_config()
    cfg = _replace(cfg, judge=_replace(cfg.judge, tuned=""))
    decider = build_decider(cfg)
    decider.load()
    if decider.error:
        raise SystemExit(f"model failed to load: {decider.error}")
    for v, texts in texts_of.items():
        labels = []
        for start in range(0, len(texts), 32):
            labels += [
                d.label if d.available else "safe"
                for d in decider.decide_batch(texts[start : start + 32])
            ]
        record("zero-shot laya", v, np.asarray([LABELS.index(x) for x in labels]))
    # E14 frozen backbones fitted on the old format
    from backbone_embed import embed

    for name, (src, var, spec, model_id, kind, _role) in RECIPES.items():
        if name in ("laya", "laya-wd03"):
            continue
        store = __import__("pickle").loads(Path(f"cache/bb-{src}-512.pkl").read_bytes())
        xtr = np.stack([store[t][var] for t in old_texts]).astype(np.float32)
        for v, texts in texts_of.items():
            uniq = sorted(set(texts))
            vec = embed(src, model_id, uniq, kind=kind, max_length=512, device="cuda")[var]
            ix = {t: i for i, t in enumerate(uniq)}
            xte = np.stack([vec[ix[t]] for t in texts]).astype(np.float32)
            p = fit_predict(spec, xtr, old_data.y, old_data.key, xte).argmax(1)
            record(f"{name} (old-format head)", v, p)
    Path("cache/ctx-stage1.json").write_text(json.dumps(result, indent=1))
    for model, per in result.items():
        for v, row in per.items():
            base = per["A"]
            cells = " ".join(f"{s} {row[s]:.3f}({row[s] - base[s]:+.3f})" for s in sets)
            print(f"{model:34s} {v}: {cells}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("show", "embed", "cv", "score", "reference", "stage1"))
    for k in ("train", "old", "re", "fresh"):
        ap.add_argument(f"--{k}", type=Path)
    ap.add_argument("--old-train", nargs="+", type=Path)
    ap.add_argument("--old-old", nargs="+", type=Path)
    ap.add_argument("--project", default="jevmon-context")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--runs", type=Path, default=Path(os.environ.get("JEVMON_RUNS", "runs")))
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    {
        "show": cmd_show,
        "embed": cmd_embed,
        "cv": cmd_cv,
        "score": cmd_score,
        "reference": cmd_reference,
        "stage1": cmd_stage1,
    }[args.mode](args)


if __name__ == "__main__":
    main()

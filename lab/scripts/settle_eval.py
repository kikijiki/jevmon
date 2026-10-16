"""E14: settle Foundation-Sec-8B vs the Laya head on a second capture and fresh families.

    python scripts/settle_eval.py embed  --old D6 D7 --recapture DIR --fresh DIR     # features only
    python scripts/settle_eval.py score  --old D6 D7 --recapture DIR --fresh DIR     # once, writes JSON
    python scripts/settle_eval.py latency --old D6 D7 --recapture DIR --fresh DIR

Recipes are frozen in RECIPES below and in research-log E14 before `score` runs; `score` refuses
to run twice (it writes cache/settle-final.json and aborts if it exists, unless --force). Training
rows are always the original r6+r7 train split (205 rows / 37 families). Sets scored:

    old     the original held-out capture (130 rows, 22 families), as in E12 / E13
    re      a second capture of the same held-out scenarios (same families)
    fresh   new held-out-only families (never in training)
    new     re + fresh (the "new held-out" of the decision rule)
    pool    old + re + fresh (family clusters shared between old and re)

Bootstrap: family-cluster (families resampled with replacement, all rows of a family kept),
paired against the Laya recipe head on the same resamples.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from backbone_embed import embed
from backbone_eval import macro_f1
from improve_lib import (
    EmbeddingCache,
    Spec,
    fit_predict,
    laya_agent,
    load_rows,
    make_data,
    render,
    to_rows,
)
from jevmon_lab import metrics
from jevmon_lab.manifest import LABELS, split_of
from jevmon_lab.runlog import RunLog

FINAL_JSON = Path("cache/settle-final.json")
BOOT = 10000

# name -> (feature source, variant, Spec, model id, kind, role)
RECIPES = {
    "laya": (
        "laya",
        "mean",
        Spec("logreg", wd=0.01, balanced=True),
        None,
        None,
        "reference (ships)",
    ),
    "laya-wd03": ("laya", "mean", Spec("logreg", wd=0.03, balanced=True), None, None, "variant"),
    "foundation-sec-8b": (
        "foundation-sec-8b",
        "mean75",
        Spec("logreg", wd=0.001, balanced=True),
        "fdtn-ai/Foundation-Sec-8B",
        "decoder",
        "candidate",
    ),
    "qwen3e-8b": (
        "qwen3e-8b",
        "mean75",
        Spec("logreg", wd=0.0003, balanced=True, standardize=True),
        "Qwen/Qwen3-Embedding-8B",
        "decoder",
        "context",
    ),
    "ettin-1b": (
        "ettin-1b",
        "mean75",
        Spec("logreg", wd=0.0003, balanced=True, standardize=True),
        "jhu-clsp/ettin-encoder-1b",
        "encoder",
        "context",
    ),
}


def load(args):
    rows = []
    for d in args.old:
        for r in load_rows(d / "dataset.jsonl"):
            r["cap"] = "train" if r["split"] == "train" else "old"
            rows.append(r)
    train_fams = {r["family"] for r in rows if r["cap"] == "train"}
    old_fams = {r["family"] for r in rows if r["cap"] == "old"}
    for r in load_rows(args.recapture / "dataset.jsonl"):
        r["cap"] = "re"
        rows.append(r)
    for r in load_rows(args.fresh / "dataset.jsonl"):
        r["cap"] = "fresh"
        rows.append(r)
    for r in rows:
        if r["cap"] in ("re", "fresh"):
            assert split_of(r["family"]) == "heldout", r["family"]
            assert r["family"] not in train_fams, r["family"]
        if r["cap"] == "fresh":
            assert r["family"] not in old_fams, r["family"]
        # the designated-driver rule: only the driver process carries the label
        assert r.get("designated") in (True, None), r["scenario_id"]
    data, texts = make_data(rows, lambda r: render(r, "shipped"))
    data.extra["cap"] = np.asarray([r["cap"] for r in rows])
    return data, texts


def features(name: str, texts: list[str]) -> np.ndarray:
    src, var, _spec, model_id, kind, _role = RECIPES[name]
    if src == "laya":
        return EmbeddingCache(laya_agent).get(texts, "mean")
    return embed(src, model_id, texts, kind=kind, max_length=512, device="cuda")[var]


def boot_sets(data, pred_by_name, sets, n=BOOT, seed=14):
    rng = np.random.default_rng(seed)
    out = {}
    for sname, idx in sets.items():
        fam = data.family[idx]
        fams = sorted(set(fam))
        members = [np.flatnonzero(fam == f) for f in fams]
        draws = [
            np.concatenate([members[i] for i in rng.integers(0, len(fams), len(fams))])
            for _ in range(n)
        ]
        y = data.y[idx]
        ref = pred_by_name["laya"][idx]
        res = {}
        for name, pred in pred_by_name.items():
            p = pred[idx]
            vals = np.asarray([macro_f1(y[d], p[d]) for d in draws])
            entry = {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}
            if name != "laya":
                diffs = vals - np.asarray([macro_f1(y[d], ref[d]) for d in draws])
                entry |= {
                    "diff_lo": float(np.percentile(diffs, 2.5)),
                    "diff_hi": float(np.percentile(diffs, 97.5)),
                    "p_better": float((diffs > 0).mean()),
                }
            res[name] = entry
        out[sname] = res
    return out


def safe_recall(data, idx, pred):
    m = data.y[idx] == LABELS.index("safe")
    return float((pred[idx][m] == LABELS.index("safe")).mean())


def verdict(point, boots, srec):
    """Pre-registered decision rule (see research-log E14)."""
    c = "foundation-sec-8b"
    new_diff = point["new"][c] - point["new"]["laya"]
    old_diff = point["old"][c] - point["old"]["laya"]
    b = boots["new"][c]
    rule1 = new_diff > 0 and b["diff_lo"] > 0
    drop_new = srec["new"]["laya"] - srec["new"][c]
    drop_pool = srec["pool"]["laya"] - srec["pool"][c]
    rule2 = new_diff >= 0.05 and old_diff >= 0.05 and drop_new <= 0.03 and drop_pool <= 0.03
    return {
        "new_diff": new_diff,
        "old_diff": old_diff,
        "new_diff_ci": [b["diff_lo"], b["diff_hi"]],
        "safe_recall_drop_new": drop_new,
        "safe_recall_drop_pool": drop_pool,
        "rule1_ci_excludes_zero": bool(rule1),
        "rule2_fallback": bool(rule2),
        "foundation_sec_wins": bool(rule1 or rule2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("embed", "score", "latency"))
    ap.add_argument("--old", nargs="+", type=Path, required=True)
    ap.add_argument("--recapture", type=Path, required=True)
    ap.add_argument("--fresh", type=Path, required=True)
    ap.add_argument("--project", default="jevmon-settle")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--runs", type=Path, default=Path(os.environ.get("JEVMON_RUNS", "runs")))
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    data, texts = load(args)
    cap = data.extra["cap"]
    tr = np.flatnonzero(cap == "train")
    sets = {
        "old": np.flatnonzero(cap == "old"),
        "re": np.flatnonzero(cap == "re"),
        "fresh": np.flatnonzero(cap == "fresh"),
        "new": np.flatnonzero(np.isin(cap, ("re", "fresh"))),
        "pool": np.flatnonzero(cap != "train"),
    }
    print(f"train rows {len(tr)} families {len(set(data.family[tr]))}")
    for k, v in sets.items():
        print(
            f"  {k:6s} rows {len(v):4d} families {len(set(data.family[v])):3d} "
            f"classes {np.bincount(data.y[v], minlength=3).tolist()}"
        )

    if mode_is(args, "embed"):
        for name in RECIPES:
            features(name, texts)
        return

    if args.mode == "latency":
        latency(texts)
        return

    if FINAL_JSON.exists() and not args.force:
        raise SystemExit(f"{FINAL_JSON} exists: held-out is scored once (use --force to redo)")
    pred = {}
    proba = {}
    for name, (_s, _v, spec, *_rest) in RECIPES.items():
        x = features(name, texts)
        p = fit_predict(spec, x[tr], data.y[tr], data.key[tr], x)
        proba[name] = p
        pred[name] = p.argmax(1)
    boots = boot_sets(data, pred, {k: v for k, v in sets.items()})
    point = {s: {} for s in sets}
    srec = {s: {} for s in sets}
    table = {}
    for s, idx in sets.items():
        table[s] = {}
        for name in RECIPES:
            sc = metrics.score(to_rows(data, idx, pred[name][idx]))
            point[s][name] = sc["macro_f1"]
            srec[s][name] = safe_recall(data, idx, pred[name])
            table[s][name] = {
                "macro_f1": sc["macro_f1"],
                "accuracy": sc["accuracy"],
                "per_class": {k: [v["precision"], v["recall"]] for k, v in sc["per_class"].items()},
                "boot": boots[s][name],
                "rows": len(idx),
                "families": len(set(data.family[idx])),
            }
    dec = verdict(point, boots, srec)
    fam_acc = {}
    for f in sorted(set(data.family[sets["new"]])):
        m = sets["new"][data.family[sets["new"]] == f]
        fam_acc[f] = {
            "label": data.rows[m[0]]["label"],
            "n": len(m),
            **{n: float((pred[n][m] == data.y[m]).mean()) for n in ("laya", "foundation-sec-8b")},
        }
    out = {"table": table, "decision": dec, "family_accuracy_new": fam_acc}
    FINAL_JSON.write_text(json.dumps(out, indent=1))
    for s in sets:
        for name in RECIPES:
            t = table[s][name]
            b = t["boot"]
            d = (
                f" diff [{b['diff_lo']:+.3f},{b['diff_hi']:+.3f}] p>{b['p_better']:.2f}"
                if "diff_lo" in b
                else ""
            )
            pc = " ".join(f"{k}:{p:.2f}/{r:.2f}" for k, (p, r) in t["per_class"].items())
            print(
                f"{s:6s} {name:18s} F1 {t['macro_f1']:.3f} [{b['lo']:.2f},{b['hi']:.2f}] "
                f"acc {t['accuracy']:.3f}{d} | {pc}"
            )
    print(json.dumps(dec, indent=1))
    for name in RECIPES:
        run = RunLog.start(
            args.project,
            f"e14-final-{name}-{time.strftime('%H%M%S')}",
            {"candidate": name, "recipe": RECIPES[name][2].name(), "role": RECIPES[name][5]},
            directory=args.runs,
        )
        flat = {}
        for s in sets:
            t = table[s][name]
            flat[f"{s}/macro_f1"] = t["macro_f1"]
            flat[f"{s}/accuracy"] = t["accuracy"]
            flat[f"{s}/ci_lo"] = t["boot"]["lo"]
            flat[f"{s}/ci_hi"] = t["boot"]["hi"]
            if "diff_lo" in t["boot"]:
                flat[f"{s}/diff_lo"] = t["boot"]["diff_lo"]
                flat[f"{s}/diff_hi"] = t["boot"]["diff_hi"]
            for k, (p, r) in t["per_class"].items():
                flat[f"{s}/precision_{k}"] = p
                flat[f"{s}/recall_{k}"] = r
        run.log(0, **flat)
        run.finish()


def mode_is(args, mode: str) -> bool:
    return args.mode == mode


def latency(texts: list[str]) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from jevmon.laya.head import embed_states

    uniq = sorted(set(texts))[:96]
    model_id = RECIPES["foundation-sec-8b"][3]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def run(batch):
        ids = [
            tok(t, add_special_tokens=False)["input_ids"][:511] + [tok.eos_token_id] for t in batch
        ]
        w = max(map(len, ids))
        inp = torch.tensor([[pad] * (w - len(i)) + i for i in ids], device="cuda")
        att = torch.tensor([[0] * (w - len(i)) + [1] * len(i) for i in ids], device="cuda")
        with torch.inference_mode():
            model(input_ids=inp, attention_mask=att, output_hidden_states=True)
        torch.cuda.synchronize()

    def bench(fn, label):
        fn(uniq[:4])
        for bs in (1, 32):
            t = []
            for s in range(0, len(uniq), bs):
                a = time.perf_counter()
                fn(uniq[s : s + bs])
                t.append((time.perf_counter() - a) * 1000 / len(uniq[s : s + bs]))
            print(
                f"{label} batch {bs}: p50 {np.percentile(t, 50):.0f} ms p95 {np.percentile(t, 95):.0f} ms"
            )

    bench(run, "foundation-sec-8b")
    print(f"  gpu memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    model = None
    torch.cuda.empty_cache()
    agent = laya_agent()
    bench(
        lambda b: embed_states(agent, b, pooling="mean", max_length=512, batch_size=len(b)), "laya"
    )


if __name__ == "__main__":
    main()

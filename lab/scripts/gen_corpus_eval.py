"""E19 stage 1 and learning curve on the synthetic corpus (see research-log E19).

    python scripts/gen_corpus_eval.py stage1 --gen D [--jev-sample 500]
    python scripts/gen_corpus_eval.py curve  --gen D --train T --old O --re R --fresh F

`stage1` scores existing models on the new rows (nothing is fitted on them): zero-shot Laya,
the shipped head `lab-head-v4`, and optionally hosted Jev 1.13 on a sample (synthetic data only;
key from OPENROUTER_API_KEY or /run/user/1000/secrets/openrouter-api-key, never printed).
`curve` fits the E15 head recipe (Laya mean embeddings, logistic wd 0.01 class-balanced) on the
original 205 train rows plus growing slices of non-disputed synthetic train-split rows
(by whole families) and scores on the ORIGINAL held-out sets with the E15 family-cluster
bootstrap (10000 resamples, seed 14). Rows of `disputed` QA are excluded from headline numbers.
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from backbone_eval import macro_f1
from ctx_eval import boot, features, sets_of
from ctx_eval import load as load_ctx
from improve_lib import Spec, fit_predict
from jevmon.detect.context import ContextSettings, parse_exec_state, render_segments
from jevmon_lab import metrics
from jevmon_lab.manifest import LABELS

SHIPPED = ("header", "exe", "template", "argv")
SPEC = Spec("logreg", wd=0.01, balanced=True)
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
CRITERIA = {
    "safe": "expected everyday activity here: shells, builds, package installs, scheduled jobs, ssh sessions and other development tooling; no action needed",
    "warn": "out of the ordinary for this workstation but plausibly harmless: a one-off script, the first run of a newly installed tool, a debugger; worth a glance",
    "crit": "looks hostile from the launch alone: downloading and running remote code, encoded or obfuscated commands, execution from a temporary, world-writable or hidden location, a shell spawned by an unexpected parent",
}


def load_gen(path: Path, mode: str = "qa") -> list[dict]:
    """mode: qa = intent agrees with the blind second pass (headline), strict = intent, second
    pass and Jev all agree, all = every row including disputed."""
    rows = [json.loads(ln) for ln in (path / "dataset.jsonl").read_text().splitlines() if ln]
    if mode == "qa":
        rows = [r for r in rows if not r.get("disputed")]
    elif mode == "strict":
        rows = [r for r in rows if not r.get("disputed") and r.get("jev_label") == r["label"]]
    for r in rows:
        r["text"] = render_segments(
            parse_exec_state(r["context"]), ContextSettings(order=SHIPPED, model_max=1000)
        )
    return rows


def score(rows: list[dict], pred: np.ndarray) -> float:
    y = np.asarray([LABELS.index(r["label"]) for r in rows])
    return metrics.score(
        [{"label": LABELS[a], "verdict": LABELS[b]} for a, b in zip(y, pred, strict=True)]
    )["macro_f1"]


def jev_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        key = Path("/run/user/1000/secrets/openrouter-api-key").read_text().strip()
    return key


def jev_label(text: str, key: str) -> str:
    body = {
        "model": "typesafe/jev-1.13",
        "state": text,
        "questions": {
            "verdict": {
                "type": "choice",
                "instructions": "Which of these categories best describes this process launch?",
                "criteria": CRITERIA,
            }
        },
    }
    req = urllib.request.Request(
        JEV_URL,
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    probs = resp["answers"]["verdict"]["probabilities"]
    return max(LABELS, key=lambda k: float(probs.get(k, 0.0)))


def cmd_stage1(args: argparse.Namespace) -> None:
    from dataclasses import replace

    from jevmon.config import load_config
    from jevmon.laya.head import TunedHead, resolve_head
    from jevmon.monitor import build_decider

    rows = load_gen(args.gen, args.mode)
    texts = [r["text"] for r in rows]
    out: dict = {"rows": len(rows)}
    head = TunedHead.load(resolve_head("lab-head-v4"))
    head_pred = head.probabilities(features(texts)).argmax(1)
    cfg = load_config()
    cfg = replace(cfg, judge=replace(cfg.judge, tuned=""))
    decider = build_decider(cfg)
    decider.load()
    zs = []
    for i in range(0, len(texts), 32):
        zs += [d.label if d.available else "safe" for d in decider.decide_batch(texts[i : i + 32])]
    zs_pred = np.asarray([LABELS.index(x) for x in zs])
    for split in ("train", "heldout"):
        idx = [i for i, r in enumerate(rows) if r["split"] == split]
        sub = [rows[i] for i in idx]
        out[split] = {
            "rows": len(idx),
            "zero_shot_laya": score(sub, zs_pred[idx]),
            "head_v4": score(sub, head_pred[idx]),
        }
    if args.jev_sample:
        rng = random.Random(14)
        idx = rng.sample(range(len(rows)), min(args.jev_sample, len(rows)))
        key = jev_key()
        jev = {i: jev_label(texts[i], key) for i in idx}
        jp = np.asarray([LABELS.index(jev[i]) for i in idx])
        sub = [rows[i] for i in idx]
        out["jev_sample"] = {
            "rows": len(idx),
            "jev_macro_f1": score(sub, jp),
            "head_macro_f1": score(sub, head_pred[idx]),
            "head_jev_agreement": float((head_pred[idx] == jp).mean()),
        }
        for r_i, lab in jev.items():
            rows[r_i]["jev_label"] = lab
    print(json.dumps(out, indent=1))
    (args.gen / f"stage1-{args.mode}.json").write_text(json.dumps(out, indent=1))


def cmd_curve(args: argparse.Namespace) -> None:
    data = load_ctx(args)  # original train + old/re/fresh held-out (E15 sets)
    base_tr = data.idx("train")
    from ctx_eval import variant_data

    d, real_texts = variant_data(data, "A")
    spec = SPEC
    if args.backbone == "laya":
        feat = features
    else:
        from backbone_embed import embed
        from settle_eval import RECIPES

        _src, var, spec, model_id, kind, _role = RECIPES[args.backbone]

        def feat(texts):
            return embed(args.backbone, model_id, texts, kind=kind, max_length=512, device="cuda")[
                var
            ]

    x_real = feat(real_texts)
    gen = load_gen(args.gen, args.mode)
    fam_train = sorted({r["family"] for r in gen if r["split"] == "train"})
    random.Random(14).shuffle(fam_train)
    gen_train = [r for r in gen if r["split"] == "train"]
    x_gen = feat([r["text"] for r in gen_train])
    y_gen = np.asarray([LABELS.index(r["label"]) for r in gen_train])
    key_gen = np.asarray([f"{r['text']}\x00{r['label']}" for r in gen_train])
    sets = sets_of(data)
    steps = (
        [0, 50, 100, len(fam_train)]
        if args.backbone != "laya"
        else [0, 25, 50, 100, 200, len(fam_train)]
    )
    preds = {}
    for k in steps:
        fams = set(fam_train[:k])
        sel = np.asarray([i for i, r in enumerate(gen_train) if r["family"] in fams], dtype=int)
        xtr = np.concatenate([x_real[base_tr], x_gen[sel]]) if len(sel) else x_real[base_tr]
        ytr = np.concatenate([d.y[base_tr], y_gen[sel]]) if len(sel) else d.y[base_tr]
        ktr = np.concatenate([d.key[base_tr], key_gen[sel]]) if len(sel) else d.key[base_tr]
        preds[f"+{k}fam({len(sel)}rows)"] = fit_predict(spec, xtr, ytr, ktr, x_real).argmax(1)
    ref = next(iter(preds))
    table = {}
    for s, idx in sets.items():
        b = boot(d, preds, idx, ref=ref)
        table[s] = {
            name: {
                "macro_f1": macro_f1(d.y[idx], p[idx]),
                "ci": [b[name]["lo"], b[name]["hi"]],
                **({"diff_vs_0": [b[name]["diff_lo"], b[name]["diff_hi"]]} if name != ref else {}),
            }
            for name, p in preds.items()
        }
    print(json.dumps(table, indent=1))
    (
        args.gen / f"curve-{args.backbone}-{args.mode}.json"
        if args.backbone != "laya"
        else f"curve-{args.mode}.json"
    ).write_text(json.dumps(table, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage1")
    s.add_argument("--gen", type=Path, required=True)
    s.add_argument("--jev-sample", type=int, default=0)
    s.add_argument("--mode", default="qa", choices=("qa", "strict", "all"))
    s.set_defaults(func=cmd_stage1)
    c = sub.add_parser("curve")
    c.add_argument("--gen", type=Path, required=True)
    for n in ("train", "old", "re", "fresh"):
        c.add_argument(f"--{n}", type=Path, required=True)
    c.add_argument("--mode", default="qa", choices=("qa", "strict", "all"))
    c.add_argument("--backbone", default="laya")
    c.add_argument("--project", default="jevmon")
    c.add_argument("--runs", default=None)
    c.set_defaults(func=cmd_curve)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

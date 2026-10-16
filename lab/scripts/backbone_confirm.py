"""Second check for a frozen-feature candidate: fresh family-grouped CV partitions (train only)
and per-context latency of the backbone forward pass.

    python scripts/backbone_confirm.py DIR... --name foundation-sec-8b --model fdtn-ai/Foundation-Sec-8B
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from backbone_eval import LAYA_SPEC, features, grid
from improve_lib import cv_scores, load_rows, make_data, render

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+", type=Path)
ap.add_argument("--name", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--cell", required=True, help="variant|spec name, as printed by backbone_eval")
ap.add_argument("--latency", action="store_true")
args = ap.parse_args()
sys.stdout.reconfigure(line_buffering=True)
rows = []
for d in args.dirs:
    rows += load_rows(d / "dataset.jsonl")
data, texts = make_data(rows, lambda r: render(r, "shipped"))
tr = data.idx("train")
var, spec_name = args.cell.split("|", 1)
spec = next(s for s in grid() if s.name() == spec_name)
x = features(args.name, texts)[var]
xl = features("laya", texts)["mean"]
for base in (100, 200):
    a = cv_scores(spec, x, data, tr, folds=4, repeats=5, base_seed=base)
    b = cv_scores(LAYA_SPEC, xl, data, tr, folds=4, repeats=5, base_seed=base)
    print(
        f"fresh partitions seeds {base}-{base + 4}: {args.name} {a['unique_f1']:.3f}+-{a['unique_f1_sd']:.3f}"
        f" vs laya recipe {b['unique_f1']:.3f}+-{b['unique_f1_sd']:.3f}",
        flush=True,
    )

if args.latency:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    uniq = sorted(set(texts))
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

    run(uniq[:4])
    for bs in (1, 32):
        t = []
        for s in range(0, min(len(uniq), 96), bs):
            a = time.perf_counter()
            run(uniq[s : s + bs])
            t.append((time.perf_counter() - a) * 1000 / len(uniq[s : s + bs]))
        print(
            f"latency batch {bs}: p50 {np.percentile(t, 50):.0f} ms p95 {np.percentile(t, 95):.0f} ms per context",
            flush=True,
        )
    print(f"gpu memory allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

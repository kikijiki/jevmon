"""LoRA sequence classifier on a decoder backbone (last-token head), train-CV then held-out.

    python scripts/backbone_lora.py cv    DIR... --model Qwen/Qwen3-8B --name qwen3-8b \
        --cells 6,3e-4,8;10,1e-3,8 [--partitions 1]
    python scripts/backbone_lora.py final DIR... --model Qwen/Qwen3-8B --name qwen3-8b \
        --cell 6,3e-4,8 --seeds 3

Same loop as `improve_ft.py` (own LoRA, no peft, class-balanced CE, AdamW, one-cycle schedule,
bf16 autocast) with the encoder replaced by a decoder stack: left padding, end token appended,
last-token state -> dropout -> linear over the three classes. Selection by family-grouped
train-CV, held-out scored once per candidate over `--seeds` seeds, and a family-cluster
bootstrap of the seed-mean predictions is not meaningful, so the report is seed mean +- sd of
the macro-F1 plus the seed-0 per-class precision/recall.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from improve_ft import FTSpec, LoRALinear, weights_for
from improve_lib import LABELS, family_folds, load_rows, make_data, render, score_pair
from jevmon_lab.runlog import RunLog, class_table, score_scalars

torch.set_num_threads(2)
DEPTH_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class DecoderTrainer:
    def __init__(
        self, model_id: str, device: str = "cuda", max_length: int = 256, pooling: str = "mean"
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16)
        for p in model.parameters():
            p.requires_grad = False
        self.model = model.to(device)
        self.device = device
        self.max_length = max_length
        self.pooling = pooling
        self.eos = self.tok.eos_token_id
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos
        self.dim = model.config.hidden_size

    def _batch(self, texts: list[str]):
        ids = [
            self.tok(t, add_special_tokens=False)["input_ids"][: self.max_length - 1] + [self.eos]
            for t in texts
        ]
        w = max(len(i) for i in ids)
        inp = torch.tensor([[self.pad] * (w - len(i)) + i for i in ids], device=self.device)
        att = torch.tensor([[0] * (w - len(i)) + [1] * len(i) for i in ids], device=self.device)
        return inp, att

    def _feats(self, texts: list[str]) -> torch.Tensor:
        inp, att = self._batch(texts)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = self.model(input_ids=inp, attention_mask=att).last_hidden_state
        if self.pooling == "last":
            f = h[:, -1].float()
        else:
            m = att.unsqueeze(-1).to(h.dtype)
            f = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float()
        return F.layer_norm(f, (f.shape[-1],))

    def _add_lora(self, rank: int, alpha: float, dropout: float) -> list[torch.nn.Parameter]:
        params = []
        for _n, module in list(self.model.named_modules()):
            for cname, child in list(module.named_children()):
                if isinstance(child, torch.nn.Linear) and cname in DEPTH_MODULES:
                    lora = LoRALinear(child, rank, alpha, dropout).to(self.device)
                    # peft's default: small kaiming-uniform A (the encoder-sized init of
                    # improve_ft explodes on a decoder's massive-activation residual stream)
                    torch.nn.init.kaiming_uniform_(lora.a, a=5**0.5)
                    setattr(module, cname, lora)
                    params += [lora.a, lora.b]
        return params

    def _drop_lora(self) -> None:
        for _n, module in list(self.model.named_modules()):
            for cname, child in list(module.named_children()):
                if isinstance(child, LoRALinear):
                    setattr(module, cname, child.base)

    def fit_predict(self, spec: FTSpec, train_texts, y, weights, test_texts) -> np.ndarray:
        torch.manual_seed(spec.seed)
        rng = np.random.default_rng(spec.seed)
        lora = self._add_lora(spec.rank, spec.alpha, spec.dropout)
        head = torch.nn.Sequential(
            torch.nn.Dropout(0.1), torch.nn.Linear(self.dim, len(LABELS))
        ).to(self.device)
        groups = [
            {"params": head.parameters(), "lr": spec.lr_head, "weight_decay": spec.wd},
            {"params": lora, "lr": spec.lr_enc, "weight_decay": spec.wd},
        ]
        opt = torch.optim.AdamW(groups)
        n = len(train_texts)
        steps = max(1, spec.epochs * ((n + spec.batch - 1) // spec.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=[g["lr"] for g in groups],
            total_steps=steps,
            pct_start=0.15,
            anneal_strategy="linear",
        )
        yt = torch.from_numpy(y).to(self.device)
        wt = torch.from_numpy(weights.astype(np.float32)).to(self.device)
        self.model.train()
        try:
            for _epoch in range(spec.epochs):
                head.train()
                total = 0.0
                order = rng.permutation(n)
                for s in range(0, n, spec.batch):
                    idx = order[s : s + spec.batch]
                    logits = head(self._feats([train_texts[i] for i in idx]))
                    ce = F.cross_entropy(logits, yt[idx], reduction="none")
                    loss = (ce * wt[idx]).sum() / wt[idx].sum()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                    opt.step()
                    sched.step()
                    total += float(loss.item()) * len(idx)
                print(f"    epoch {_epoch} loss {total / n:.3f}", flush=True)
            self.model.eval()
            head.eval()
            out = []
            with torch.no_grad():
                for s in range(0, len(test_texts), 16):
                    out.append(
                        torch.softmax(head(self._feats(test_texts[s : s + 16])), -1).cpu().numpy()
                    )
            return np.concatenate(out)
        finally:
            self._drop_lora()
            del opt, head, lora
            torch.cuda.empty_cache()


def parse_cell(text: str) -> FTSpec:
    e, lr, r = text.split(",")
    return FTSpec("lora", epochs=int(e), lr_enc=float(lr), rank=int(r), batch=8, alpha=2.0 * int(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("cv", "final"))
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cells", default="6,3e-4,8")
    ap.add_argument("--cell", default="6,3e-4,8")
    ap.add_argument("--partitions", type=int, default=1)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--pooling", choices=("mean", "last"), default="mean")
    ap.add_argument("--project", default="jevmon-backbones")
    ap.add_argument("--tag", default="b13")
    ap.add_argument("--runs", type=Path, default=Path(os.environ.get("JEVMON_RUNS", "runs")))
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    rows = []
    for d in args.dirs:
        rows += load_rows(d / "dataset.jsonl")
    data, texts = make_data(rows, lambda r: render(r, "shipped"))
    tr, te = data.idx("train"), data.idx("heldout")
    trainer = DecoderTrainer(args.model, max_length=args.max_length, pooling=args.pooling)

    if args.mode == "cv":
        parts = [family_folds(data, tr, args.folds, 1000 + p) for p in range(args.partitions)]
        for cell in args.cells.split(";"):
            spec = parse_cell(cell)
            f1s, secs = [], []
            for folds in parts:
                preds = np.zeros(len(data.y), dtype=int)
                started = time.perf_counter()
                for val in folds:
                    trn = np.setdiff1d(tr, val)
                    w = weights_for(spec, data.y[trn], data.key[trn])
                    proba = trainer.fit_predict(
                        spec, [texts[i] for i in trn], data.y[trn], w, [texts[i] for i in val]
                    )
                    preds[val] = proba.argmax(1)
                pooled = np.concatenate(folds)
                sc = score_pair(data, pooled, preds[pooled])
                f1s.append(sc["unique"]["macro_f1"])
                secs.append(time.perf_counter() - started)
            print(
                f"{args.name} lora {cell:14s} CV unique-F1 {np.mean(f1s):.3f} "
                f"(parts {[round(f, 3) for f in f1s]}) {np.mean(secs):.0f}s/part",
                flush=True,
            )
            run = RunLog.start(
                args.project,
                f"{args.tag}-cv-lora-{args.name}-{cell}-{time.strftime('%H%M%S')}",
                {"backbone": args.name, "cell": cell, "kind": "lora"},
                directory=args.runs,
            )
            run.log(0, **{"cv/unique_f1": float(np.mean(f1s)), "cv/seconds": float(np.mean(secs))})
            run.finish()
        return

    spec = parse_cell(args.cell)
    pairs = []
    for seed in range(args.seeds):
        s2 = replace(spec, seed=seed)
        w = weights_for(s2, data.y[tr], data.key[tr])
        proba = trainer.fit_predict(
            s2, [texts[i] for i in tr], data.y[tr], w, [texts[i] for i in te]
        )
        pair = score_pair(data, te, proba.argmax(1))
        pairs.append(pair)
        pc = pair["rows"]["per_class"]
        print(
            f"{args.name} lora {args.cell} seed {seed}: held-out F1 {pair['rows']['macro_f1']:.3f} "
            f"acc {pair['rows']['accuracy']:.3f} "
            + " ".join(f"{k}:{v['precision']:.2f}/{v['recall']:.2f}" for k, v in pc.items()),
            flush=True,
        )
        np.save(f"cache/bb-lora-{args.name}-seed{seed}.npy", proba)
    f1 = [p["rows"]["macro_f1"] for p in pairs]
    print(
        f"{args.name} lora {args.cell}: macro-F1 {np.mean(f1):.3f} +- {np.std(f1):.3f}", flush=True
    )
    run = RunLog.start(
        args.project,
        f"{args.tag}-final-lora-{args.name}-{time.strftime('%H%M%S')}",
        {"backbone": args.name, "cell": args.cell, "kind": "lora", "seeds": args.seeds},
        directory=args.runs,
    )
    run.log(
        0,
        **score_scalars("heldout", pairs[0]["rows"]),
        **{"heldout/f1_mean": float(np.mean(f1)), "heldout/f1_sd": float(np.std(f1))},
    )
    run.table("heldout_per_class", class_table(pairs[0]["rows"]))
    run.finish()
    Path(f"cache/bb-lora-{args.name}.json").write_text(json.dumps({"f1": f1, "cell": args.cell}))


if __name__ == "__main__":
    main()

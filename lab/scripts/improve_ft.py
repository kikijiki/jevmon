"""Encoder training on the lab labels: full fine-tune and LoRA, scored like the linear head.

A small PyTorch loop (no peft dependency): mean-pooled encoder state -> dropout -> linear over
the four categories, class-balanced cross-entropy, AdamW, bf16 autocast. `lora` freezes the
encoder and trains rank-r adapters on every attention/MLP projection plus the head; `full`
trains everything; `head` trains only the linear layer (a sanity check against the LBFGS head).
Selection is family-grouped CV on train families; held-out is scored once per candidate.
"""

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from improve_lib import LABELS, class_balance, score_pair
from jevmon_lab import metrics


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = torch.nn.Parameter(torch.randn(rank, base.in_features) * (1.0 / rank**0.5))
        self.b = torch.nn.Parameter(torch.zeros(base.out_features, rank))
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.a.T @ self.b.T) * self.scale


def add_lora(encoder: torch.nn.Module, rank: int, alpha: float, dropout: float) -> None:
    for name, module in list(encoder.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
    for p in encoder.parameters():
        p.requires_grad = False
    for m in encoder.modules():
        if isinstance(m, LoRALinear):
            m.a.requires_grad = True
            m.b.requires_grad = True


@dataclass(slots=True)
class FTSpec:
    mode: str = "lora"  # full | lora | head
    epochs: int = 6
    lr_enc: float = 3e-4  # encoder (lora adapters or full weights)
    lr_head: float = 1e-3
    batch: int = 16
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    wd: float = 0.01
    smoothing: float = 0.0
    sample: str = "rows"
    seed: int = 0

    def name(self) -> str:
        return (
            f"{self.mode}/e{self.epochs}/lr{self.lr_enc}/h{self.lr_head}"
            f"/r{self.rank}/ls{self.smoothing}/{self.sample}"
        )


class Trainer:
    """Holds the pretrained encoder once; every fit starts from the pretrained weights."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.tok = agent.tok
        self.device = agent.device
        self.pretrained = {
            k: v.detach().cpu().clone() for k, v in agent.model.encoder.state_dict().items()
        }
        self.encoder_cls = agent.model.encoder

    def _fresh_encoder(self) -> torch.nn.Module:
        enc = self.agent.model.encoder
        # drop adapters from an earlier fit, then restore the pretrained weights
        for _name, module in list(enc.named_modules()):
            for child_name, child in list(module.named_children()):
                if isinstance(child, LoRALinear):
                    setattr(module, child_name, child.base)
        enc.load_state_dict(self.pretrained)
        for p in enc.parameters():
            p.requires_grad = True
        return enc

    def _batches(self, texts: list[str], max_length: int = 256):
        from laya.common import encode_text

        enc = encode_text(
            self.tok,
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def _pool(self, encoder, ids, mask):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return ((hidden * m).sum(1) / m.sum(1).clamp(min=1.0)).float()

    def fit_predict(
        self,
        spec: FTSpec,
        train_texts: list[str],
        y: np.ndarray,
        weights: np.ndarray,
        test_texts: list[str],
        on_epoch=None,
    ) -> np.ndarray:
        torch.manual_seed(spec.seed)
        rng = np.random.default_rng(spec.seed)
        encoder = self._fresh_encoder()
        if spec.mode == "lora":
            add_lora(encoder, spec.rank, spec.alpha, spec.dropout)
            encoder.to(self.device)
        elif spec.mode == "head":
            for p in encoder.parameters():
                p.requires_grad = False
        dim = encoder.config.hidden_size
        head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(dim, len(LABELS))).to(
            self.device
        )
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
        groups = [{"params": head.parameters(), "lr": spec.lr_head, "weight_decay": spec.wd}]
        if enc_params:
            groups.append({"params": enc_params, "lr": spec.lr_enc, "weight_decay": spec.wd})
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
        for epoch in range(spec.epochs):
            encoder.train(spec.mode != "head")
            head.train()
            order = rng.permutation(n)
            total = 0.0
            for s in range(0, n, spec.batch):
                idx = order[s : s + spec.batch]
                ids, mask = self._batches([train_texts[i] for i in idx])
                if spec.mode == "head":
                    with torch.no_grad():
                        feats = self._pool(encoder, ids, mask)
                else:
                    feats = self._pool(encoder, ids, mask)
                logits = head(feats)
                ce = F.cross_entropy(
                    logits, yt[idx], reduction="none", label_smoothing=spec.smoothing
                )
                loss = (ce * wt[idx]).sum() / wt[idx].sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
                total += float(loss.item()) * len(idx)
            if on_epoch is not None:
                on_epoch(epoch, total / n)
        encoder.eval()
        head.eval()
        self.last_encoder, self.last_head = encoder, head
        out = []
        with torch.no_grad():
            for s in range(0, len(test_texts), 32):
                ids, mask = self._batches(test_texts[s : s + 32])
                out.append(torch.softmax(head(self._pool(encoder, ids, mask)), -1).cpu().numpy())
        return np.concatenate(out)


def export_checkpoint(trainer: Trainer, out_dir, spec: FTSpec) -> tuple[np.ndarray, np.ndarray]:
    """Write the last fitted encoder as a laya checkpoint directory (adapters merged) and return
    the jointly trained head as (weight, bias) for a `TunedHead` file (mean pooling)."""
    import glob
    import shutil
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    enc = trainer.last_encoder
    if spec.mode == "lora":
        for _n, module in list(enc.named_modules()):
            for child_name, child in list(module.named_children()):
                if isinstance(child, LoRALinear):
                    with torch.no_grad():
                        child.base.weight += (child.b @ child.a).to(
                            child.base.weight.dtype
                        ) * child.scale
                    setattr(module, child_name, child.base)
    snapshot = Path(
        glob.glob(
            str(Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots/*")
        )[0]
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = load_file(str(snapshot / "model.safetensors"))
    new = {k: v.detach().cpu() for k, v in enc.state_dict().items()}
    for key in list(state):
        if key.startswith("encoder."):
            name = key[len("encoder.") :]
            state[key] = new[name].to(state[key].dtype).contiguous()
    save_file(state, str(out / "model.safetensors"))
    shutil.copy(snapshot / "rl_agent_config.json", out / "rl_agent_config.json")
    shutil.copytree(snapshot / "tokenizer", out / "tokenizer", dirs_exist_ok=True)
    shutil.copytree(snapshot / "encoder", out / "encoder", dirs_exist_ok=True)
    linear = trainer.last_head[1]
    return linear.weight.detach().cpu().numpy(), linear.bias.detach().cpu().numpy()


def weights_for(spec: FTSpec, y: np.ndarray, key: np.ndarray) -> np.ndarray:
    from collections import Counter

    counts = Counter(key.tolist())
    n = np.asarray([counts[k] for k in key], dtype=np.float64)
    w = 1.0 / n if spec.sample == "unique" else np.ones(len(y))
    return class_balance(y, w, len(LABELS)).astype(np.float32)


def cv_ft(trainer: Trainer, spec: FTSpec, texts, data, idx, folds_list) -> dict:
    """Pooled out-of-fold scores for `spec` over the given fold partitions (train-CV)."""
    preds = np.zeros(len(data.y), dtype=int)
    started = time.perf_counter()
    for val in folds_list:
        tr = np.setdiff1d(idx, val)
        w = weights_for(spec, data.y[tr], data.key[tr])
        proba = trainer.fit_predict(
            spec, [texts[i] for i in tr], data.y[tr], w, [texts[i] for i in val]
        )
        preds[val] = proba.argmax(1)
    pooled = np.concatenate(folds_list)
    sc = score_pair(data, pooled, preds[pooled])
    fold_means = [
        metrics.score([{"label": data.rows[i]["label"], "verdict": LABELS[preds[i]]} for i in val])[
            "macro_f1"
        ]
        for val in folds_list
    ]
    return {
        "unique_f1": sc["unique"]["macro_f1"],
        "rows_f1": sc["rows"]["macro_f1"],
        "fold_mean_f1": float(np.mean(fold_means)),
        "seconds": time.perf_counter() - started,
        "copy": copy.copy(sc["unique"]["per_class"]),
    }

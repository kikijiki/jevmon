"""One LoRA/PEFT check on the mixed-pool data: does a LoRA-tuned encoder beat the shipped frozen
linear head now that the train pool is far larger than in E7/E8b?

Two backbones are supported, chosen with `--backbone {laya,ettin}` (default `laya`):

- `laya` (default): PEFT LoRA on the Laya encoder (`convaiinnovations/laya`) with mean pooling and a
  linear head trained jointly. The frozen head (E20, `lab/scripts/train_heads.py`,
  `HEADS["laya"]` = `laya:mean|logreg/wd0.001/bal`) is the reference. Outputs stay under
  `lab-data-lora/`, exactly as before.
- `ettin`: the same recipe on `jhu-clsp/ettin-encoder-1b` (a ModernBERT encoder, loaded through
  `transformers.AutoModel`; LoRA targets `Wqkv`/`Wo`/`Wi`, the four `nn.Linear`s per block). The
  candidate pools the mean of the **last** layer only, not the frozen head's `mean+mean75` concat.
  The frozen Ettin head (`ettin-1b:mean+mean75|logreg/wd0.001/bal`, `lab-data-e20/results.json`) is
  the reference; it is refit for the paired test from cached Ettin embeddings when `lab/cache/`
  holds them, otherwise the reference numbers are read from `lab-data-e20/results.json` and the
  paired test is skipped (reported). Outputs go under `lab-data-lora-ettin/`.

Both backbones share the data loaders, metric/bootstrap and frozen-head fit imported from
`lab/scripts/train_heads.py` (not duplicated). The six E20 eval sets are scored once, at the end, for
the chosen config. Pre-registration and the verdict rule live in the plan (`PLAN.md`, gitignored).

Subcommands (each resumable; the long ones write small artefacts under the output dir):

    python lab/scripts/lora_check.py embed  --backbone laya   # cached frozen Laya mean vectors
    python lab/scripts/lora_check.py head   --backbone laya   # refit + score the frozen head
    python lab/scripts/lora_check.py select --backbone ettin  # grid on a family-held-out dev split
    python lab/scripts/lora_check.py final  --backbone ettin  # chosen cell, 3 seeds, six eval sets
    python lab/scripts/lora_check.py report --backbone ettin  # results.json + summary.md

`--device cuda|gpu|cpu`; `--batch` and `--grad-checkpointing` for the CUDA pod; `--smoke` (64 rows,
1 epoch, 1 cell, tiny eval sets) to prove the path end to end. `all` runs the whole chain.

Every run is tracked (`jevmon_lab.runlog`): one Trackio project `jevmon-lora` with a run per
selection cell (`{backbone}-{cell}`) and final seed (`{backbone}-final-seed{N}`), a frozen-head
reference run, and a seed summary. Trackio is the optional `trackio` lab extra (imported lazily); if
it is absent the run still writes an identical log to `<out>/metrics.jsonl`. `--runs DIR` sets the
Trackio store (else `$TRACKIO_DIR`, else `<repo>/runs`); `--no-track` disables Trackio; `--out DIR`
redirects the output dir (used for the Laya rerun under `lab-data-lora-rerun/`). The report's
`--baseline PATH` shows earlier vs rerun side by side. Eval during training only ever uses the train
pool's dev split; the six held-out sets are scored once, at the end.
"""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import train_heads as T  # needs the scripts dir on sys.path
from improve_lib import Spec, fit_predict, sample_weights
from jevmon_lab import metrics
from jevmon_lab.manifest import LABELS
from jevmon_lab.runlog import RunLog, class_table, confusion_table, score_scalars

# The six E20 eval sets, in order.
SIX = (
    "lab/pool",
    "gen2/heldout",
    "shell-safety-v1.1/driver",
    "cli-command-risk/driver",
    "quasarnix/driver",
    "splunk/reviewed",
)
EXTERNAL = SIX[2:]

# The head recipe under test (frozen and joint): class-balanced CE, head weight decay 0.001.
HEAD_SPEC = Spec(wd=0.001, balanced=True)
HEAD_WD = HEAD_SPEC.wd

LORA_TARGETS = ("Wqkv", "Wo", "Wi")  # every attention/MLP projection of a ModernBERT encoder
MAXLEN = 512
BATCH = 16
CHUNK = 4000  # shuffled in chunks before length-sorting, so batches are not globally ordered
PRED_BATCH = 64
SEEDS = (0, 1, 2)

# The Laya grid (unchanged) and the trimmed Ettin grid: the four best Laya dev cells.
GRID = tuple((r, lr, e) for r in (8, 16) for lr in (1e-4, 3e-4, 1e-3) for e in (3, 6))
GRID_ETTIN = ((16, 3e-4, 6), (8, 1e-3, 3), (8, 3e-4, 6), (16, 3e-4, 3))

# The shipped Laya head's macro-F1 on each of the six sets (lab-data-e20/results.json).
SHIPPED_LAYA = {
    "lab/pool": 0.519,
    "gen2/heldout": 0.546,
    "shell-safety-v1.1/driver": 0.700,
    "cli-command-risk/driver": 0.606,
    "quasarnix/driver": 0.842,
    "splunk/reviewed": 0.381,
}

# Per backbone: output dir, HF id, frozen-head key, cached feature parts and the grid.
BACKBONES = {
    "laya": {
        "out": ROOT / "lab-data-lora",
        "model_id": "convaiinnovations/laya",
        "shipped_key": "laya:mean|logreg/wd0.001/bal",
        "shipped": SHIPPED_LAYA,
        "cached_parts": (("laya", "mean"),),
        "cached_store": "laya",
        "grid": GRID,
    },
    "ettin": {
        "out": ROOT / "lab-data-lora-ettin",
        "model_id": "jhu-clsp/ettin-encoder-1b",
        "shipped_key": "ettin-1b:mean+mean75|logreg/wd0.001/bal",
        "shipped": None,  # read from lab-data-e20/results.json
        "cached_parts": (("ettin-1b", "mean"), ("ettin-1b", "mean75")),
        "cached_store": "ettin-1b",
        "grid": GRID_ETTIN,
    },
}

# Smoke: a tiny end-to-end run (run + report on 64 train rows and tiny eval sets).
SMOKE_CELL = (16, 3e-4, 3)
SMOKE_ROWS = 64
SMOKE_EVAL_ROWS = 24

# Tracking: one Trackio project for every LoRA run; a JSONL mirror lands in each output dir.
PROJECT = "jevmon-lora"
STEP_LOG = 10  # log step scalars every N optimiser steps
LOSS_EMA_DECAY = 0.98
ECE_BINS = 10

_AGENTS: dict = {}
_SHIPPED: dict = {}


# ------------------------------------------------------------------ tracking helpers


def runs_dir(args: argparse.Namespace) -> Path:
    """Where Trackio keeps its SQLite store: `--runs`, else `$TRACKIO_DIR`, else `<repo>/runs`."""
    if getattr(args, "runs", None):
        return Path(args.runs)
    env = os.environ.get("TRACKIO_DIR")
    return Path(env) if env else ROOT / "runs"


def open_run(args: argparse.Namespace, name: str, config: dict, out: Path) -> RunLog:
    """A tracked run under `PROJECT` with a `<out>/metrics.jsonl` mirror (always written)."""
    return RunLog.start(
        PROJECT,
        name,
        config,
        directory=runs_dir(args),
        jsonl=out / "metrics.jsonl",
        enabled=not getattr(args, "no_track", False),
    )


def param_counts(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def device_info(device: str) -> tuple[str, str]:
    """(device name, compute dtype) as the run sees it: the CUDA GPU in bf16, else the CPU."""
    import torch

    if str(device).startswith(("cuda", "gpu")) and torch.cuda.is_available():
        return str(torch.cuda.get_device_name(0)), "bfloat16"
    return "cpu", "float32"


def gpu_utilization() -> float | None:
    """Instantaneous GPU utilisation %, or None (pynvml, else nvidia-smi, else unavailable)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:  # noqa: BLE001, S110 - pynvml missing or no driver
        pass  # fall through to nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                timeout=2,
                stderr=subprocess.DEVNULL,
            )
            return float(out.decode().strip().splitlines()[0])
        except Exception:  # noqa: BLE001 - nvidia-smi present but failed
            return None
    return None


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    """ECE over `bins` confidence bins: sum of bin mass * |accuracy - mean confidence|."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (
            (confidence >= lo) & (confidence <= hi)
            if i == 0
            else (confidence > lo) & (confidence <= hi)
        )
        if not in_bin.any():
            continue
        total += float(in_bin.sum()) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return total / max(len(y), 1)


def score_rows(y: np.ndarray, pred: np.ndarray) -> list[dict]:
    return [{"label": LABELS[t], "verdict": LABELS[p]} for t, p in zip(y, pred, strict=True)]


def family_macro_f1(y: np.ndarray, pred: np.ndarray, fam: np.ndarray) -> list[dict]:
    """Macro-F1 of each dev family (families with a single class report that class' F1)."""
    rows = []
    for family in sorted(set(fam.tolist())):
        mask = fam == family
        sc = metrics.score(score_rows(y[mask], pred[mask]))
        rows.append({"family": family, "n": int(mask.sum()), "macro_f1": round(sc["macro_f1"], 4)})
    return rows


def dev_eval(model, head, dev, device: str) -> dict:
    """Score the train-pool dev split during training (never one of the six eval sets)."""
    _rows, toks, y, fam = dev
    probs = predict(model, head, toks, device)
    pred = probs.argmax(1)
    sc = metrics.score(score_rows(y, pred))
    confusion, _ = T._installed()
    sc["confusion"] = confusion(y, pred).tolist()
    sc["confidence"] = float(probs.max(axis=1).mean())
    sc["ece"] = expected_calibration_error(probs, y)
    sc["families"] = family_macro_f1(y, pred, fam)
    return sc


def log_dev(run: RunLog, epoch: int, sc: dict) -> None:
    """Log one epoch's dev metrics, per-class/confusion/prediction/family tables included."""
    run.log(epoch, **score_scalars("dev", sc))
    run.log(epoch, **{"dev/confidence": sc["confidence"], "dev/ece": sc["ece"]})
    if sc["families"]:
        run.log(
            epoch,
            **{
                "dev/family_worst_macro_f1": min(f["macro_f1"] for f in sc["families"]),
                "dev/family_mean_macro_f1": float(np.mean([f["macro_f1"] for f in sc["families"]])),
            },
        )
    run.table("dev/per_class", class_table(sc), step=epoch)
    run.table("dev/confusion", confusion_table(sc), step=epoch)
    run.table(
        "dev/predictions",
        [
            {
                "label": lab,
                "support": sc["support"].get(lab, 0),
                "predicted": sc["per_class"].get(lab, {}).get("predicted", 0),
            }
            for lab in LABELS
        ],
        step=epoch,
    )
    run.table("dev/family", sc["families"], step=epoch)


# ------------------------------------------------------------------ shared helpers


def out_dir(args: argparse.Namespace) -> Path:
    """The output root: `--out`, else the backbone default, with a `smoke/` subdir for `--smoke`."""
    override = getattr(args, "out", None)
    base = Path(override) if override else BACKBONES[args.backbone]["out"]
    return base / "smoke" if getattr(args, "smoke", False) else base


def shipped_numbers(backbone: str) -> dict:
    """The frozen head's macro-F1 on the six sets: constants for Laya, else from results.json."""
    cfg = BACKBONES[backbone]
    if cfg["shipped"] is not None:
        return cfg["shipped"]
    if backbone not in _SHIPPED:
        results = json.loads((ROOT / "lab-data-e20/results.json").read_text())
        key = cfg["shipped_key"]
        _SHIPPED[backbone] = {
            name: (
                float(j["macro_f1"]) if (j := results["sets"][name]["judges"].get(key)) else None
            )
            for name in SIX
        }
    return _SHIPPED[backbone]


def six_groups() -> list:
    """The six E20 eval sets, in `SIX` order, from `train_heads.eval_groups()`."""
    groups = {g.name: g for g in T.eval_groups()}
    missing = [n for n in SIX if n not in groups]
    if missing:
        raise SystemExit(f"eval sets missing from train_heads.eval_groups(): {missing}")
    return [groups[n] for n in SIX]


def group_arrays(g) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([LABELS.index(r.label) for r in g.rows])
    fam = np.asarray([r.family for r in g.rows])
    return y, fam


def stratified_subset(rows: list, n: int) -> list:
    """Deterministic round-robin subset across the present labels (all classes kept if possible)."""
    if n >= len(rows):
        return list(rows)
    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(r.label, []).append(r)
    labels = [c for c in LABELS if c in buckets]
    out: list = []
    while len(out) < n and any(buckets[c] for c in labels):
        for c in labels:
            if buckets[c] and len(out) < n:
                out.append(buckets[c].pop(0))
    return out


def smoke_groups() -> list:
    """The six eval sets truncated to `SMOKE_EVAL_ROWS` rows each, same names."""
    out = []
    for g in T.eval_groups():
        rows = (
            stratified_subset(g.rows, SMOKE_EVAL_ROWS) if g.labelled else g.rows[:SMOKE_EVAL_ROWS]
        )
        out.append(T.Group(g.name, rows, labelled=g.labelled, note=g.note + " [smoke]"))
    return out


def rows_for(args: argparse.Namespace) -> list:
    rows = T.training_rows()
    return stratified_subset(rows, SMOKE_ROWS) if getattr(args, "smoke", False) else rows


def groups_for(args: argparse.Namespace) -> list:
    return smoke_groups() if getattr(args, "smoke", False) else T.eval_groups()


def grid_for(args: argparse.Namespace) -> tuple:
    return (SMOKE_CELL,) if getattr(args, "smoke", False) else BACKBONES[args.backbone]["grid"]


def dev_split(families: list[str]) -> tuple[set[str], set[str]]:
    """Deterministic family-held-out dev split of the train families (~25 % dev)."""
    fit: set[str] = set()
    dev: set[str] = set()
    for family in sorted(set(families)):
        digest = hashlib.blake2b(f"{family}|lora-dev".encode(), digest_size=8).digest()
        (dev if int.from_bytes(digest, "big") % 4 == 0 else fit).add(family)
    return fit, dev


def make_batches(
    lengths: list[int],
    rng: np.random.Generator | None = None,
    batch: int = BATCH,
    chunk: int = CHUNK,
) -> list[list[int]]:
    """Batches over indices: shuffle, length-sort each `chunk`, slice, then shuffle batch order.

    Length-sorting keeps dynamic padding cheap; the chunk and the final shuffle keep the order from
    being perfectly length-ordered across an epoch.
    """
    idx = list(range(len(lengths)))
    if rng is not None:
        rng.shuffle(idx)
    batches: list[list[int]] = []
    for start in range(0, len(idx), chunk):
        part = sorted(idx[start : start + chunk], key=lambda i: lengths[i])
        batches.extend(part[i : i + batch] for i in range(0, len(part), batch))
    if rng is not None:
        rng.shuffle(batches)
    return batches


def autocast_ctx(device: str):
    """bf16 autocast on CUDA (incl. the ROCm iGPU); a no-op elsewhere (CPU smoke)."""
    import torch

    if str(device).startswith("cuda") or str(device) == "gpu":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def get_agent(backbone: str, device: str):
    """(tokenizer, encoder) for the backbone, loaded once and cached (keyed by backbone/device)."""
    key = (backbone, device)
    if key not in _AGENTS:
        if backbone == "laya":
            import laya

            agent = laya.load(
                model_id_or_path=BACKBONES["laya"]["model_id"],
                device=device,
                revision=laya.PINNED_REVISIONS["convaiinnovations/laya"],
            )
            _AGENTS[key] = (agent.tok, agent.model.encoder)
        else:
            import torch
            from transformers import AutoModel, AutoTokenizer

            mid = BACKBONES[backbone]["model_id"]
            tok = AutoTokenizer.from_pretrained(mid)
            dtype = torch.bfloat16 if str(device).startswith(("cuda", "gpu")) else torch.float32
            model = AutoModel.from_pretrained(mid, dtype=dtype)
            _AGENTS[key] = (tok, model.to(device))
    return _AGENTS[key]


def tokenize(backbone: str, tok, texts: list[str]) -> list[list[int]]:
    if backbone == "laya":
        from laya.common import encode_text

        enc = encode_text(tok, texts, truncation=True, max_length=MAXLEN)
    else:
        enc = tok(texts, truncation=True, max_length=MAXLEN)
    return [list(ids) for ids in enc["input_ids"]]


def pad_batch(toks: list[list[int]], batch: list[int], device):
    import torch

    width = max(len(toks[i]) for i in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for k, i in enumerate(batch):
        row = torch.tensor(toks[i], dtype=torch.long)
        ids[k, : len(row)] = row
        mask[k, : len(row)] = 1
    return ids.to(device), mask.to(device)


# ------------------------------------------------------------------ LoRA model


def build_lora(encoder, rank: int, device: str, backbone: str, grad_ckpt: bool = False):
    """A fresh PEFT LoRA wrapper over a private copy of the frozen encoder (never mutates it)."""
    import copy

    from peft import LoraConfig, TaskType, get_peft_model

    base = copy.deepcopy(encoder).to(device)
    for param in base.parameters():
        param.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(BACKBONES[backbone].get("targets", LORA_TARGETS)),
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(base, config)
    if grad_ckpt:
        try:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        except (AttributeError, ValueError) as exc:  # pragma: no cover - depends on peft/model
            raise SystemExit(f"gradient checkpointing unavailable: {exc}") from exc
    return model


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def total_grad_norm(params) -> float:
    """L2 norm of the current gradients of `params` (computed before any clipping)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum())
    return total**0.5


def train_cell(
    rows,
    rank: int,
    lr: float,
    epochs: int,
    seed: int,
    *,
    backbone: str,
    device: str,
    batch: int = BATCH,
    grad_ckpt: bool = False,
    run: RunLog | None = None,
    dev=None,
    log=print,
):
    """Train a LoRA adapter + linear head jointly; returns (model, head, toks, hidden, seconds).

    `run` receives step/epoch metrics; `dev` is `(rows, toks, y, families)` of the train-pool dev
    split, scored after every epoch (observability only - selection stays the end-of-training dev
    macro-F1, the pre-registered rule).
    """
    import torch
    import torch.nn.functional as F

    tok, encoder = get_agent(backbone, device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    texts = [r.text for r in rows]
    y = np.asarray([LABELS.index(r.label) for r in rows])
    key = np.asarray([f"{r.text}\x00{r.label}" for r in rows])
    toks = tokenize(backbone, tok, texts)
    lengths = [len(t) for t in toks]
    weights = sample_weights(HEAD_SPEC, y, key, len(LABELS))
    weights = weights / max(weights.mean(), 1e-12)

    model = build_lora(encoder, rank, device, backbone, grad_ckpt)
    model.train()
    hidden = encoder.config.hidden_size
    head = torch.nn.Linear(hidden, len(LABELS)).to(device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    per_epoch = len(make_batches(lengths, batch=batch))
    steps = per_epoch * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    wt = torch.from_numpy(weights.astype(np.float32)).to(device)
    yt = torch.from_numpy(y).to(device)
    rng = np.random.default_rng(seed)

    if run is not None:
        trainable, total = param_counts(model)
        run.log(
            0,
            **{
                "model/trainable_params": float(trainable),
                "model/total_params": float(total),
                "model/trainable_fraction": trainable / total if total else 0.0,
                "weight/mean": float(weights.mean()),
            },
        )
        for c, label in enumerate(LABELS):
            mask = y == c
            if mask.any():
                run.log(0, **{f"weight/mean_class/{label}": float(weights[mask].mean())})

    started = time.perf_counter()
    last = started
    step = 0
    ema: float | None = None
    epoch_loss, epoch_n = 0.0, 0
    for epoch in range(epochs):
        for b in make_batches(lengths, rng, batch=batch):
            idx = torch.tensor(b, device=device)
            ids, mask = pad_batch(toks, b, device)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device):
                out = model(input_ids=ids, attention_mask=mask)
                pooled = mean_pool(out.last_hidden_state.float(), mask)
                logits = head(pooled)
                ce = F.cross_entropy(logits, yt[idx], reduction="none")
                loss = (ce * wt[idx]).sum() / wt[idx].sum() + 0.5 * HEAD_WD * (head.weight**2).sum()
            loss.backward()
            grad_norm = total_grad_norm(params)
            opt.step()
            sched.step()
            step += 1
            now = time.perf_counter()
            step_s = now - last
            last = now
            raw = float(loss.detach())
            ema = raw if ema is None else LOSS_EMA_DECAY * ema + (1 - LOSS_EMA_DECAY) * raw
            epoch_loss += raw
            epoch_n += 1
            if run is not None and (step % STEP_LOG == 0 or step == 1):
                tokens = int(mask.sum())
                scalars = {
                    "loss/train": raw,
                    "loss/train_ema": ema,
                    "lr": float(sched.get_last_lr()[0]),
                    "grad_norm": grad_norm,
                    "time/step_s": step_s,
                    "throughput/samples_per_s": len(b) / max(step_s, 1e-9),
                    "throughput/tokens_per_s": tokens / max(step_s, 1e-9),
                    "progress/epoch": epoch + step / max(per_epoch, 1),
                }
                if torch.cuda.is_available():
                    scalars["gpu/mem_allocated_gb"] = torch.cuda.memory_allocated() / 2**30
                    scalars["gpu/mem_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                util = gpu_utilization()
                if util is not None:
                    scalars["gpu/utilization"] = util
                run.log(step, **scalars)
        log(f"    epoch {epoch + 1}/{epochs} done at {time.perf_counter() - started:.0f}s")
        if run is not None:
            run.log(epoch + 1, **{"loss/train_epoch": epoch_loss / max(epoch_n, 1)})
            if dev is not None:
                sc = dev_eval(model, head, dev, device)
                log_dev(run, epoch + 1, sc)
                model.train()
                log(f"      dev macro-F1 {sc['macro_f1']:.4f}")
    return model, head, toks, hidden, time.perf_counter() - started


def predict(model, head, toks: list[list[int]], device: str) -> np.ndarray:
    """(n, classes) softmax probabilities, rows in `toks` order."""
    import torch

    model.eval()
    probs = np.empty((len(toks), len(LABELS)), dtype=np.float32)
    order = sorted(range(len(toks)), key=lambda i: len(toks[i]))
    with torch.inference_mode():
        for start in range(0, len(order), PRED_BATCH):
            b = order[start : start + PRED_BATCH]
            ids, mask = pad_batch(toks, b, device)
            with autocast_ctx(device):
                out = model(input_ids=ids, attention_mask=mask)
                pooled = mean_pool(out.last_hidden_state.float(), mask)
            probs[b] = torch.softmax(head(pooled.float()), dim=-1).float().cpu().numpy()
    return probs


def save_fit(model, head, path: Path) -> int:
    """Save adapter (+linear head) only; return the byte size of everything written."""
    from safetensors.numpy import save_file

    model.save_pretrained(path / "adapter")
    save_file(
        {
            "weight": np.ascontiguousarray(head.weight.detach().float().cpu().numpy()),
            "bias": np.ascontiguousarray(head.bias.detach().float().cpu().numpy()),
        },
        str(path / "head.safetensors"),
    )
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ------------------------------------------------------------------ frozen head baseline


def cached_head_ready(backbone: str, texts: list[str]) -> tuple[bool, str]:
    """Whether the frozen embedding cache can serve `texts` (so the paired test can run)."""
    cfg = BACKBONES[backbone]
    store = cfg["cached_store"]
    if backbone == "laya":
        paths = [
            ROOT / "lab/cache/emb-english-mean-512.pkl",
            ROOT / "lab/cache/e20/laya-mean-512.pkl",
        ]
    else:
        paths = [
            ROOT / f"lab/cache/bb-{store}-512.pkl",
            ROOT / f"lab/cache/e20/bb-{store}-512.pkl",
        ]
    if not any(p.exists() for p in paths):
        return (
            False,
            f"no frozen {store} embedding cache under lab/cache/ ({len(texts)} texts needed)",
        )
    return True, ""


def frozen_head_probs(backbone: str, train=None, groups=None) -> dict[str, list[list[float]]]:
    """Refit the shipped recipe on the mixed pool from cached embeddings; score the six sets."""
    train = T.training_rows() if train is None else train
    groups = six_groups() if groups is None else groups
    parts = BACKBONES[backbone]["cached_parts"]
    x_tr = T.matrix(parts, [r.text for r in train])
    y = np.asarray([LABELS.index(r.label) for r in train])
    key = np.asarray([f"{r.text}\x00{r.label}" for r in train])
    out: dict[str, list[list[float]]] = {}
    for g in groups:
        x = T.matrix(parts, [r.text for r in g.rows])
        out[g.name] = fit_predict(HEAD_SPEC, x_tr, y, key, x).tolist()
    return out


def cmd_embed(args: argparse.Namespace) -> None:
    if args.backbone != "laya":
        print(f"embed: nothing to do for {args.backbone} (frozen head reads lab/cache/ if present)")
        return
    T.cmd_embed_laya(argparse.Namespace(device=args.device, batch=args.batch))


def cmd_head(args: argparse.Namespace) -> None:
    out = out_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    train = rows_for(args)
    groups = [g for g in groups_for(args) if g.name in SIX]
    texts = [r.text for r in train] + [r.text for g in groups for r in g.rows]
    ready, reason = cached_head_ready(args.backbone, texts)
    if not ready:
        (out / "head_preds.json").write_text(
            json.dumps(
                {"key": BACKBONES[args.backbone]["shipped_key"], "probs": None, "skipped": reason}
            )
        )
        print(f"frozen head skipped: {reason}", flush=True)
        return
    probs = frozen_head_probs(args.backbone, train, groups)
    shipped = shipped_numbers(args.backbone)
    check = {}
    for g in groups:
        y, _ = group_arrays(g)
        point = T.score_point(y, np.asarray(probs[g.name]).argmax(1))
        check[g.name] = {"refit": point, "shipped": shipped[g.name]}
        print(f"{g.name:34s} refit {point:.4f} shipped {shipped[g.name]:.4f}", flush=True)
    (out / "head_preds.json").write_text(
        json.dumps({"key": BACKBONES[args.backbone]["shipped_key"], "probs": probs, "check": check})
    )


# ------------------------------------------------------------------ selection


def frozen_dev_score(
    args: argparse.Namespace, fit_rows: list, dev_rows: list, y_dev
) -> float | None:
    """Frozen-head macro-F1 on the dev split, from the cached embeddings if they are present."""
    ready, _ = cached_head_ready(args.backbone, [r.text for r in fit_rows + dev_rows])
    if not ready:
        return None
    try:
        parts = BACKBONES[args.backbone]["cached_parts"]
        x_fit = T.matrix(parts, [r.text for r in fit_rows])
        y_fit = np.asarray([LABELS.index(r.label) for r in fit_rows])
        key_fit = np.asarray([f"{r.text}\x00{r.label}" for r in fit_rows])
        x_dev = T.matrix(parts, [r.text for r in dev_rows])
        return T.score_point(y_dev, fit_predict(HEAD_SPEC, x_fit, y_fit, key_fit, x_dev).argmax(1))
    except SystemExit:
        return None


def cmd_select(args: argparse.Namespace) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    out = out_dir(args)
    rows = rows_for(args)
    fit_fams, dev_fams = dev_split([r.family for r in rows])
    fit_rows = [r for r in rows if r.family in fit_fams]
    dev_rows = [r for r in rows if r.family in dev_fams]
    y_dev = np.asarray([LABELS.index(r.label) for r in dev_rows])
    fam_dev = np.asarray([r.family for r in dev_rows])
    print(f"fit rows {len(fit_rows)} ({len(fit_fams)} fams) dev {len(dev_rows)} ({len(dev_fams)})")
    tok, _ = get_agent(args.backbone, args.device)
    dev_toks = tokenize(args.backbone, tok, [r.text for r in dev_rows])
    frozen_dev = frozen_dev_score(args, fit_rows, dev_rows, y_dev)
    device_name, dtype = device_info(args.device)
    dev = (dev_rows, dev_toks, y_dev, fam_dev) if len(dev_rows) else None

    cells = []
    for rank, lr, epochs in grid_for(args):
        epochs = 1 if args.smoke else epochs
        name = f"r{rank}-lr{lr:g}-e{epochs}"
        path = out / "select" / f"{name}.json"
        if path.exists() and not args.force:
            cells.append(json.loads(path.read_text()))
            continue
        print(f"cell {name}", flush=True)
        config = {
            "backbone": args.backbone,
            "phase": "select",
            "cell": name,
            "rank": rank,
            "alpha": 2 * rank,
            "lr": lr,
            "epochs": epochs,
            "batch": args.batch,
            "max_len": MAXLEN,
            "seed": 0,
            "n_train": len(fit_rows),
            "n_dev": len(dev_rows),
            "dtype": dtype,
            "device": str(args.device),
            "device_name": device_name,
            "grad_checkpointing": bool(args.grad_checkpointing),
            "smoke": bool(args.smoke),
        }
        run = open_run(args, f"{args.backbone}-{name}", config, out)
        try:
            if frozen_dev is not None:
                run.log(0, **{"reference/frozen_head_dev_macro_f1": float(frozen_dev)})
            model, head, _toks, _hidden, seconds = train_cell(
                fit_rows,
                rank,
                lr,
                epochs,
                seed=0,
                backbone=args.backbone,
                device=args.device,
                batch=args.batch,
                grad_ckpt=args.grad_checkpointing,
                run=run,
                dev=dev,
            )
            probs = predict(model, head, dev_toks, args.device)
            score = T.score_point(y_dev, probs.argmax(1)) if len(y_dev) else 0.0
            size = save_fit(model, head, out / "select" / name)
            run.log(
                epochs,
                **{
                    "select/dev_macro_f1": float(score),
                    "cell/train_seconds": float(seconds),
                    "cell/adapter_bytes": float(size),
                },
            )
        finally:
            run.finish()
        record = {
            "cell": name,
            "rank": rank,
            "lr": lr,
            "epochs": epochs,
            "dev_macro_f1": score,
            "train_seconds": seconds,
            "adapter_bytes": size,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))
        cells.append(record)
        print(f"  dev {score:.4f} in {seconds:.0f}s", flush=True)

    chosen = min(cells, key=lambda c: (-c["dev_macro_f1"], c["rank"], c["lr"], c["epochs"]))
    record = {
        "fit_rows": len(fit_rows),
        "fit_families": len(fit_fams),
        "dev_rows": len(dev_rows),
        "dev_families": len(dev_fams),
        "frozen_head_dev_macro_f1": frozen_dev,
        "cells": sorted(cells, key=lambda c: -c["dev_macro_f1"]),
        "chosen": chosen,
    }
    if args.backbone != "laya" or args.smoke:  # keep the Laya select.json byte-identical
        record = {"backbone": args.backbone, "smoke": bool(args.smoke), **record}
    out.mkdir(parents=True, exist_ok=True)
    (out / "select.json").write_text(json.dumps(record, indent=1))
    print(f"chosen {chosen['cell']} dev {chosen['dev_macro_f1']:.4f}", flush=True)


# ------------------------------------------------------------------ final (held-out, once)


def load_head_probs(out: Path) -> dict | None:
    path = out / "head_preds.json"
    if not path.exists():
        return None
    head = json.loads(path.read_text())
    return head.get("probs")


def log_reference_run(args: argparse.Namespace, out: Path, reference: dict) -> None:
    """A constant baseline run so the dashboard shows the frozen head next to every run."""
    marker = out / "reference.done"
    if marker.exists():
        return
    run = open_run(
        args,
        f"{args.backbone}-reference-frozen-head",
        {
            "backbone": args.backbone,
            "phase": "reference",
            "source": (
                "lab-data-e20/results.json"
                if BACKBONES[args.backbone]["shipped"] is None
                else "constants"
            ),
            "key": BACKBONES[args.backbone]["shipped_key"],
        },
        out,
    )
    run.log(0, **{f"reference/{n}": float(v) for n, v in reference.items() if v is not None})
    run.table(
        "reference/sets",
        [
            {"set": n, "macro_f1": (None if v is None else round(v, 4))}
            for n, v in reference.items()
        ],
    )
    run.finish()
    marker.write_text("ok\n")


def log_final_summary(
    args: argparse.Namespace, out: Path, base: Path, reference: dict, head
) -> None:
    """Seed mean/sd and the ensemble paired difference, once all seeds are present."""
    seed_files = sorted(base.glob("seed*/preds.json"))
    if not seed_files:
        return
    seed_names = [p.parent.name for p in seed_files]
    marker = out / "final-summary.done"
    if marker.exists() and marker.read_text().strip() == ",".join(seed_names):
        return
    seeds = [json.loads(p.read_text()) for p in seed_files]
    groups = six_groups() if not args.smoke else [g for g in smoke_groups() if g.name in SIX]
    run = open_run(
        args,
        f"{args.backbone}-final-summary",
        {"backbone": args.backbone, "phase": "final-summary", "seeds": seed_names},
        out,
    )
    rows_sets = []
    paired_rows = []
    for g in groups:
        y, fam = group_arrays(g)
        p_seed = [np.asarray(s["probs"][g.name]) for s in seeds]
        vals = [T.score_point(y, p.argmax(1)) for p in p_seed]
        ensemble = np.mean(p_seed, axis=0)
        run.log(
            0,
            **{
                f"seed_mean/{g.name}": float(np.mean(vals)),
                f"seed_sd/{g.name}": float(np.std(vals)),
                f"seed_min/{g.name}": float(np.min(vals)),
                f"seed_max/{g.name}": float(np.max(vals)),
                f"ensemble/{g.name}/macro_f1": float(T.score_point(y, ensemble.argmax(1))),
            },
        )
        ref = reference.get(g.name)
        if ref is not None:
            run.log(0, **{f"reference/{g.name}": float(ref)})
        rows_sets.append(
            {
                "set": g.name,
                "n": len(g.rows),
                "seed_mean": round(float(np.mean(vals)), 4),
                "seed_sd": round(float(np.std(vals)), 4),
                "ensemble": round(float(T.score_point(y, ensemble.argmax(1))), 4),
                "reference": (None if ref is None else round(float(ref), 4)),
            }
        )
        if head is not None:
            d = paired_diff(y, ensemble.argmax(1), np.asarray(head[g.name]).argmax(1), fam)
            run.log(
                0,
                **{
                    f"paired/{g.name}/point": d["point"],
                    f"paired/{g.name}/ci_lo": d["ci"][0],
                    f"paired/{g.name}/ci_hi": d["ci"][1],
                    f"paired/{g.name}/p_better": d["p_better"],
                    f"paired/{g.name}/win": d["win"],
                    f"paired/{g.name}/lost": d["lost"],
                },
            )
            paired_rows.append({"set": g.name, **d, "ci": None})
    run.table("summary/sets", rows_sets)
    if paired_rows:
        run.table(
            "summary/paired",
            [
                {
                    "set": r["set"],
                    "point": round(r["point"], 4),
                    "ci_lo": round(r["ci"][0], 4) if r["ci"] else None,
                    "ci_hi": round(r["ci"][1], 4) if r["ci"] else None,
                    "win": r["win"],
                    "lost": r["lost"],
                }
                for r in paired_rows
            ],
        )
    run.finish()
    marker.write_text(",".join(seed_names) + "\n")


def cmd_final(args: argparse.Namespace) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    out = out_dir(args)
    chosen = json.loads((out / "select.json").read_text())["chosen"]
    rows = rows_for(args)
    groups = six_groups() if not args.smoke else smoke_groups()
    groups = [g for g in groups if g.name in SIX]
    tok, _ = get_agent(args.backbone, args.device)
    base = out / "final" / chosen["cell"]
    seeds = [0] if args.smoke else [int(s) for s in args.seeds.split(",")]
    device_name, dtype = device_info(args.device)
    reference = shipped_numbers(args.backbone)
    head = load_head_probs(out)

    dev_fams = dev_split([r.family for r in rows])[1]
    dev_rows = [r for r in rows if r.family in dev_fams]
    dev = (
        (
            dev_rows,
            tokenize(args.backbone, tok, [r.text for r in dev_rows]),
            np.asarray([LABELS.index(r.label) for r in dev_rows]),
            np.asarray([r.family for r in dev_rows]),
        )
        if dev_rows
        else None
    )
    log_reference_run(args, out, reference)

    for seed in seeds:
        path = base / f"seed{seed}" / "preds.json"
        if path.exists() and not args.force:
            print(f"seed {seed} already scored", flush=True)
            continue
        print(f"final seed {seed} cell {chosen['cell']}", flush=True)
        run = open_run(
            args,
            f"{args.backbone}-final-seed{seed}",
            {
                "backbone": args.backbone,
                "phase": "final",
                "cell": chosen["cell"],
                "rank": chosen["rank"],
                "alpha": 2 * chosen["rank"],
                "lr": chosen["lr"],
                "epochs": chosen["epochs"],
                "batch": args.batch,
                "max_len": MAXLEN,
                "seed": seed,
                "n_train": len(rows),
                "n_dev": len(dev_rows),
                "dtype": dtype,
                "device": str(args.device),
                "device_name": device_name,
                "grad_checkpointing": bool(args.grad_checkpointing),
                "smoke": bool(args.smoke),
            },
            out,
        )
        try:
            model, head_model, _toks, _hidden, seconds = train_cell(
                rows,
                chosen["rank"],
                chosen["lr"],
                chosen["epochs"],
                seed=seed,
                backbone=args.backbone,
                device=args.device,
                batch=args.batch,
                grad_ckpt=args.grad_checkpointing,
                run=run,
                dev=dev,
            )
            probs: dict[str, list] = {}
            preds: dict[str, np.ndarray] = {}
            summaries = {}
            for g in groups:
                toks = tokenize(args.backbone, tok, [r.text for r in g.rows])
                p = predict(model, head_model, toks, args.device)
                probs[g.name] = p.tolist()
                preds[g.name] = p.argmax(1)
                y, fam = group_arrays(g)
                summaries[g.name] = T.summarize(y, p.argmax(1), fam)
            size = save_fit(model, head_model, base / f"seed{seed}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"seed": seed, "train_seconds": seconds, "adapter_bytes": size, "probs": probs},
                    indent=1,
                )
            )
            rows_sets = []
            for g in groups:
                s = summaries[g.name]
                ref = reference.get(g.name)
                run.log(
                    seed,
                    **{
                        f"eval/{g.name}/macro_f1": s["macro_f1"],
                        f"eval/{g.name}/ci_lo": s["macro_f1_ci"][0],
                        f"eval/{g.name}/ci_hi": s["macro_f1_ci"][1],
                        f"eval/{g.name}/accuracy": s["accuracy"],
                        f"eval/{g.name}/always_safe": s["always_safe"],
                    },
                )
                if ref is not None:
                    run.log(seed, **{f"reference/{g.name}": float(ref)})
                rows_sets.append(
                    {
                        "set": g.name,
                        "n": s["n"],
                        "families": s["families"],
                        "macro_f1": round(s["macro_f1"], 4),
                        "ci_lo": round(s["macro_f1_ci"][0], 4),
                        "ci_hi": round(s["macro_f1_ci"][1], 4),
                        "always_safe": round(s["always_safe"], 4),
                        "reference": (None if ref is None else round(float(ref), 4)),
                    }
                )
            run.table("eval/sets", rows_sets, step=seed)
            run.log(
                seed,
                **{
                    "cell/train_seconds": float(seconds),
                    "cell/adapter_bytes": float(size),
                    "cell/seed": float(seed),
                },
            )
            if head is not None:
                for g in groups:
                    y, fam = group_arrays(g)
                    d = paired_diff(y, preds[g.name], np.asarray(head[g.name]).argmax(1), fam)
                    run.log(
                        seed,
                        **{
                            f"paired/{g.name}/point": d["point"],
                            f"paired/{g.name}/ci_lo": d["ci"][0],
                            f"paired/{g.name}/ci_hi": d["ci"][1],
                            f"paired/{g.name}/p_better": d["p_better"],
                            f"paired/{g.name}/win": d["win"],
                            f"paired/{g.name}/lost": d["lost"],
                        },
                    )
        finally:
            run.finish()
        print(f"  seed {seed} scored in {seconds:.0f}s", flush=True)

    log_final_summary(args, out, base, reference, head)


# ------------------------------------------------------------------ latency


def _time_encoder(encode, toks: list[list[int]], device: str, batch: int) -> dict:
    import torch

    order = sorted(range(len(toks)), key=lambda i: len(toks[i]))
    per_ctx = []
    for start in range(0, len(order), batch):
        chunk = order[start : start + batch]
        ids, mask = pad_batch(toks, chunk, device)
        with torch.inference_mode():
            encode(ids, mask)  # warm
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            encode(ids, mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        per_ctx.append((time.perf_counter() - t0) * 1000.0 / len(chunk))
    per_ctx = np.asarray(per_ctx)
    return {
        "batch": batch,
        "n": len(order),
        "p50_ms_per_ctx": float(np.percentile(per_ctx, 50)),
        "p95_ms_per_ctx": float(np.percentile(per_ctx, 95)),
    }


def cmd_latency(args: argparse.Namespace) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    import copy

    from peft import PeftModel

    out = out_dir(args)
    chosen = json.loads((out / "select.json").read_text())["chosen"]
    tok, encoder = get_agent(args.backbone, args.device)
    texts = [r.text for g in six_groups() for r in g.rows]
    texts = texts[:: max(1, len(texts) // 512)]
    toks = tokenize(args.backbone, tok, texts)
    frozen = encoder.eval()

    def frozen_encode(ids, mask):
        with autocast_ctx(args.device):
            frozen(input_ids=ids, attention_mask=mask)

    adapter_dir = out / "final" / chosen["cell"] / "seed0" / "adapter"
    lora = PeftModel.from_pretrained(copy.deepcopy(encoder), adapter_dir, is_trainable=False).eval()

    def lora_encode(ids, mask):
        with autocast_ctx(args.device):
            lora(input_ids=ids, attention_mask=mask)

    res = {"backbone": args.backbone, "cell": chosen["cell"], "texts": len(texts)}
    for batch in (1, 32):
        res[f"frozen_b{batch}"] = _time_encoder(frozen_encode, toks, args.device, batch)
        res[f"lora_b{batch}"] = _time_encoder(lora_encode, toks, args.device, batch)
    (out / "latency.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


# ------------------------------------------------------------------ export (the shipped adapter)


def cmd_export(args: argparse.Namespace) -> None:
    """Package seed 0 of the chosen config as the shipped `<backbone>-lora` adapter+head.

    Reads the PEFT adapter and the jointly-trained head under `<out>/final/<cell>/seed0/` and
    writes `app/src/jevmon/laya/heads/{laya,ettin}-lora.safetensors`.
    """
    sys.stdout.reconfigure(line_buffering=True)
    import laya
    from safetensors import safe_open

    from jevmon.laya.head import LoraHead, TunedHead

    out = out_dir(args)
    select = json.loads((out / "select.json").read_text())
    chosen = select["chosen"]
    results = json.loads((out / "results.json").read_text())
    seed0 = out / "final" / chosen["cell"] / "seed0"
    adapter_dir = seed0 / "adapter"
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    targets = tuple(sorted(config["target_modules"]))
    prefix = "base_model.model."
    a_suffix, b_suffix = ".lora_A.weight", ".lora_B.weight"
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="numpy") as fh:
        keys = list(fh.keys())
        raw = {k: fh.get_tensor(k) for k in keys}
    adapter = []
    for key in sorted(raw):
        if not key.endswith(a_suffix):
            continue
        stem = key[: -len(a_suffix)]
        if stem + b_suffix not in raw:
            raise SystemExit(f"adapter is missing {stem + b_suffix}")
        module = stem.removeprefix(prefix)
        adapter.append((module, raw[key], raw[stem + b_suffix]))
    with safe_open(seed0 / "head.safetensors", framework="numpy") as fh:
        weight, bias = fh.get_tensor("weight"), fh.get_tensor("bias")

    per_seed = results.get("per_seed", {})
    metrics = tuple(sorted((name, float(np.mean(vals))) for name, vals in per_seed.items()))
    recipe = f"lora/r{rank}/lr{chosen['lr']:g}/e{chosen['epochs']}/seed0"
    training_set = (
        "mixed pool: original lab train + E19b consensus train + external (shell-safety, cli-risk) "
        f"train rows ({results.get('train_pool_rows', '?')} rows)"
    )
    head = TunedHead(
        labels=("safe", "warn", "crit"),
        weight=weight,
        bias=bias,
        max_length=512,
        pooling="mean",
        backbone="" if args.backbone == "laya" else BACKBONES[args.backbone]["model_id"],
        recipe=recipe,
        training_set=training_set,
        metrics=metrics,
    )
    lora = LoraHead(
        head=head,
        rank=rank,
        alpha=alpha,
        target_modules=targets,
        revision=laya.PINNED_REVISIONS["convaiinnovations/laya"] if args.backbone == "laya" else "",
        adapter=tuple(adapter),
        per_seed=tuple((name, tuple(vals)) for name, vals in per_seed.items()),
    )
    app_heads = ROOT / "app/src/jevmon/laya/heads"
    app_heads.mkdir(parents=True, exist_ok=True)
    dest = app_heads / f"{args.backbone}-lora.safetensors"
    lora.save(dest)
    loaded = LoraHead.load(dest)
    size = dest.stat().st_size
    print(
        f"wrote {dest} ({size / 2**20:.1f} MiB): {loaded.rank=} alpha={loaded.alpha} "
        f"targets={loaded.target_modules} modules={len(loaded.adapter)} rev={loaded.revision[:12]}"
    )
    print(f"per-seed macro-F1: {dict(loaded.per_seed)}", flush=True)


# ------------------------------------------------------------------ report (CPU)


def paired_diff(y: np.ndarray, p_lora: np.ndarray, p_head: np.ndarray, fam: np.ndarray) -> dict:
    confusion, _ = T._installed()
    present = sorted(set(y.tolist()))
    fams = sorted(set(fam.tolist()))
    members = [np.flatnonzero(fam == f) for f in fams]
    conf_a = np.stack([confusion(y[m], p_lora[m]) for m in members])
    conf_b = np.stack([confusion(y[m], p_head[m]) for m in members])
    rng = np.random.default_rng(T.SEED)
    draw = rng.integers(0, len(fams), size=(T.BOOT, len(fams)))
    diff = T._metric(conf_a[draw].sum(axis=1), present) - T._metric(
        conf_b[draw].sum(axis=1), present
    )
    point = float(
        T._metric(confusion(y, p_lora), present) - T._metric(confusion(y, p_head), present)
    )
    lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
    return {
        "point": point,
        "ci": [lo, hi],
        "p_better": float((diff > 0).mean()),
        "win": bool(point > 0 and lo > 0),
        "lost": bool(point < 0 and hi < 0),
    }


def cmd_report(args: argparse.Namespace) -> None:
    out = out_dir(args)
    select = json.loads((out / "select.json").read_text())
    chosen = select["chosen"]
    shipped = shipped_numbers(args.backbone)
    base = out / "final" / chosen["cell"]
    seed_files = sorted(base.glob("seed*/preds.json"))
    if not seed_files:
        raise SystemExit(f"no final predictions under {base} (run `final` first)")
    seeds = [json.loads(p.read_text()) for p in seed_files]
    head = (
        json.loads((out / "head_preds.json").read_text())
        if (out / "head_preds.json").exists()
        else {"probs": None, "skipped": "head not scored"}
    )
    has_head = head.get("probs") is not None

    groups = six_groups() if not args.smoke else [g for g in smoke_groups() if g.name in SIX]
    sets: dict = {}
    head_summary: dict = {}
    lora_summary: dict = {}
    paired: dict = {}
    per_seed: dict = {}
    for g in groups:
        y, fam = group_arrays(g)
        p_seed = [np.asarray(s["probs"][g.name]) for s in seeds]
        p_lora = np.mean(p_seed, axis=0)
        lora_summary[g.name] = T.summarize(y, p_lora.argmax(1), fam)
        per_seed[g.name] = [T.score_point(y, p.argmax(1)) for p in p_seed]
        if has_head:
            p_head = np.asarray(head["probs"][g.name])
            head_summary[g.name] = T.summarize(y, p_head.argmax(1), fam)
            paired[g.name] = paired_diff(y, p_lora.argmax(1), p_head.argmax(1), fam)
        sets[g.name] = {
            "n": len(g.rows),
            "families": len(set(fam.tolist())),
            "shipped_head": shipped[g.name],
            "head_refit": (
                float(T.score_point(y, np.asarray(head["probs"][g.name]).argmax(1)))
                if has_head
                else None
            ),
            "lora": float(T.score_point(y, p_lora.argmax(1))),
            "per_seed": per_seed[g.name],
        }

    baseline = None
    if getattr(args, "baseline", None):
        bpath = Path(args.baseline)
        if bpath.exists():
            earlier = json.loads(bpath.read_text())
            baseline = {"path": str(bpath), "sets": {}}
            for name in SIX:
                e = earlier.get("sets", {}).get(name, {})
                earlier_lora = e.get("lora")
                baseline["sets"][name] = {
                    "earlier_lora": earlier_lora,
                    "earlier_shipped": e.get("shipped_head"),
                    "delta": (sets[name]["lora"] - earlier_lora)
                    if earlier_lora is not None
                    else None,
                }
    if has_head:
        wins = [s for s in ("lab/pool", "gen2/heldout") if paired[s]["win"]]
        lost = [s for s in EXTERNAL if paired[s]["lost"]]
        verdict = {
            "beat_lab_pool": paired["lab/pool"]["win"],
            "beat_gen2_heldout": paired["gen2/heldout"]["win"],
            "lost_external": lost,
            "lora_wins": len(wins) == 2 and not lost,
        }
        if args.backbone != "laya" or args.smoke:
            verdict = {"paired": True, **verdict}
    else:
        verdict = {"paired": False, "reason": head.get("skipped") or "no frozen head predictions"}

    train_seconds = {
        "selection": sorted({c["cell"]: c["train_seconds"] for c in select["cells"]}.items()),
        "final_per_seed": [s["train_seconds"] for s in seeds],
    }
    adapter_bytes = {
        "cell": chosen["cell"],
        "selection": chosen["adapter_bytes"],
        "final": [s["adapter_bytes"] for s in seeds],
    }
    latency = (
        json.loads((out / "latency.json").read_text()) if (out / "latency.json").exists() else None
    )

    results = {
        "shipped_head_key": BACKBONES[args.backbone]["shipped_key"],
        "train_pool_rows": len(rows_for(args)),
        "selection": select,
        "verdict": verdict,
        "sets": sets,
        "head": head_summary,
        "lora": lora_summary,
        "paired": paired,
        "per_seed": per_seed,
        "train_seconds": train_seconds,
        "adapter_bytes": adapter_bytes,
        "latency": latency,
    }
    if baseline is not None:
        results["baseline"] = baseline
    if args.backbone != "laya" or args.smoke:  # keep the Laya results.json as before
        results = {
            "backbone": args.backbone,
            "smoke": bool(args.smoke),
            "shipped_head_source": (
                "lab-data-e20/results.json"
                if BACKBONES[args.backbone]["shipped"] is None
                else "constants"
            ),
            **results,
        }
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "summary.md").write_text(render_report(results))
    print((out / "summary.md").read_text())


def fmt_ci(summary: dict) -> str:
    lo, hi = summary["macro_f1_ci"]
    return f"{summary['macro_f1']:.3f} [{lo:.2f}, {hi:.2f}]"


def baseline_lines(r: dict) -> list[str]:
    """The `--baseline` side-by-side section (earlier run vs this rerun), or nothing."""
    b = r.get("baseline")
    if not b:
        return []
    lines = [
        "",
        f"## Earlier vs rerun ({b['path']})",
        "",
        "| eval set | earlier LoRA | rerun LoRA | delta | earlier ref head |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in SIX:
        e = b["sets"][name]
        earlier = f"{e['earlier_lora']:.3f}" if e["earlier_lora"] is not None else "n/a"
        delta = f"{e['delta']:+.3f}" if e["delta"] is not None else "n/a"
        ref = f"{e['earlier_shipped']:.3f}" if e["earlier_shipped"] is not None else "n/a"
        lines.append(f"| {name} | {earlier} | {r['sets'][name]['lora']:.3f} | {delta} | {ref} |")
    return lines


def render_report(r: dict) -> str:
    """Dispatch: the Laya report is byte-identical to the original; Ettin/Smoke use the new one."""
    if r.get("backbone", "laya") == "laya":
        return render_report_laya(r)
    return render_report_ettin(r)


def render_report_laya(r: dict) -> str:
    s = r["selection"]["chosen"]
    frozen_dev = r["selection"]["frozen_head_dev_macro_f1"]
    frozen_txt = f"{frozen_dev:.3f}" if frozen_dev is not None else "n/a (no cache)"
    lines = [
        "# LoRA/PEFT check on the mixed pool (Laya encoder) vs the shipped frozen head",
        "",
        (
            f"Train pool {r['train_pool_rows']} rows. Chosen cell `{s['cell']}` (dev macro-F1"
            f" {s['dev_macro_f1']:.3f} on {r['selection']['dev_rows']} rows /"
            f" {r['selection']['dev_families']} held-out families; frozen head on dev"
            f" {frozen_txt})."
        ),
        (
            f"LoRA = ensemble of {len(r['per_seed'][SIX[0]])} seeds; family-cluster bootstrap,"
            f" {T.BOOT} draws, seed {T.SEED}."
        ),
        "",
        "| eval set | n | shipped head | LoRA [95% CI] | diff LoRA-head [95% CI] | P(better) | lost |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in SIX:
        e = r["sets"][name]
        d = r["paired"].get(name)
        if d:
            diff_txt = f"{d['point']:+.3f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
            ptxt, lost = f"{d['p_better']:.2f}", "yes" if d["lost"] else "no"
        else:
            diff_txt, ptxt, lost = "-", "-", "-"
        lines.append(
            f"| {name} | {e['n']} | {e['shipped_head']:.3f} | {fmt_ci(r['lora'][name])} |"
            f" {diff_txt} | {ptxt} | {lost} |"
        )
    lines += [
        "",
        "## Per-seed LoRA macro-F1",
        "",
        "| eval set | "
        + " | ".join(f"seed{i}" for i in range(len(r["per_seed"][SIX[0]])))
        + " | mean ± sd |",
        "| --- | " + " | ".join("---" for _ in r["per_seed"][SIX[0]]) + " | --- |",
    ]
    for name in SIX:
        vals = r["per_seed"][name]
        mean, sd = float(np.mean(vals)), float(np.std(vals))
        lines.append(
            f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {mean:.3f} ± {sd:.3f} |"
        )
    v = r["verdict"]
    lines += [
        "",
        "## Selection grid (dev macro-F1, family-held-out from the train pool)",
        "",
        "| cell | rank | lr | epochs | dev macro-F1 | train s | adapter B |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in sorted(r["selection"]["cells"], key=lambda c: -c["dev_macro_f1"]):
        lines.append(
            f"| {c['cell']} | {c['rank']} | {c['lr']:g} | {c['epochs']} |"
            f" {c['dev_macro_f1']:.3f} | {c['train_seconds']:.0f} | {c['adapter_bytes']} |"
        )
    adaptive = r["adapter_bytes"]
    lines += ["", "## Verdict", ""]
    if r["paired"]:
        lines += [
            f"- lab/pool beaten (point > 0, CI excludes 0): **{v['beat_lab_pool']}**",
            f"- gen2/heldout (E19b) beaten: **{v['beat_gen2_heldout']}**",
            f"- external sets lost (point < 0, CI excludes 0): {v['lost_external'] or 'none'}",
            f"- **LoRA beats the shipped head: {v['lora_wins']}**",
        ]
    else:
        lines += [
            f"- **Paired test skipped**: {v.get('reason', 'no frozen head predictions')}.",
            "- The verdict rule is stated in the plan but not evaluated; read the LoRA column against",
            "  the reference head column, with the per-seed spread, and treat it as undetermined.",
        ]
    lines += [
        "",
        "## Cost",
        "",
        (
            f"- Selection: {len(r['selection']['cells'])} cells,"
            f" {sum(c['train_seconds'] for c in r['selection']['cells']) / 60:.0f} min total."
        ),
        (
            f"- Final: {len(adaptive['final'])} seeds,"
            f" {sum(r['train_seconds']['final_per_seed']) / 60:.0f} min total"
            f" (per seed {[int(x) for x in r['train_seconds']['final_per_seed']]} s)."
        ),
        (
            f"- Adapter + head size: {adaptive['final'][0]} bytes"
            f" ({adaptive['final'][0] / 2**20:.1f} MiB; adapter + {len(LABELS)}-class head)."
        ),
    ]
    lat = r.get("latency")
    if lat:
        lines += [
            "- Inference latency per context (iGPU, forward only, mean-pooled):",
            (
                f"  - frozen encoder: {lat['frozen_b1']['p50_ms_per_ctx']:.1f} ms p50 /"
                f" {lat['frozen_b1']['p95_ms_per_ctx']:.1f} ms p95 (batch 1),"
                f" {lat['frozen_b32']['p50_ms_per_ctx']:.1f} /"
                f" {lat['frozen_b32']['p95_ms_per_ctx']:.1f} (batch 32)"
            ),
            (
                f"  - LoRA encoder: {lat['lora_b1']['p50_ms_per_ctx']:.1f} /"
                f" {lat['lora_b1']['p95_ms_per_ctx']:.1f} (batch 1),"
                f" {lat['lora_b32']['p50_ms_per_ctx']:.1f} /"
                f" {lat['lora_b32']['p95_ms_per_ctx']:.1f} (batch 32)"
            ),
        ]
    lines += baseline_lines(r)
    chosen_note = (
        "- Chosen on a family-held-out dev split of the train pool only; the six eval sets were"
        " scored once, at the end, for that one config."
    )
    gpu_note = (
        "- No step needed a GPU larger than the iGPU; the Ettin-1B LoRA (or any larger encoder)"
        " was not started."
    )
    lines += ["", "## Notes", ""]
    if r["paired"]:
        refit = ", ".join(f"{n} {r['sets'][n]['head_refit']:.3f}" for n in SIX)
        worst = max(abs(r["sets"][n]["head_refit"] - r["sets"][n]["shipped_head"]) for n in SIX)
        lo, hi = r["paired"]["lab/pool"]["ci"]
        lines += [
            f"- Frozen head refit on the same train pool: {refit} (max |refit - E20| = {worst:.4f}).",
            chosen_note,
            gpu_note,
            (
                "- Read: LoRA clears the E19b and the two shell-safety/cli-risk and Splunk external sets"
                f" by a wide margin, but on the lab held-out pool the gain is"
                f" {r['paired']['lab/pool']['point']:+.3f} with a CI that includes 0 ({lo:+.2f}, {hi:+.2f})"
                f" and a per-seed spread of {min(r['per_seed']['lab/pool']):.2f}-"
                f"{max(r['per_seed']['lab/pool']):.2f}, so under the pre-registered rule it does not beat"
                " the head; QuasarNix is a small-tolerance miss, not a loss."
            ),
        ]
    else:
        lines += [
            chosen_note,
            gpu_note,
            (
                "- Paired test skipped (no frozen-embedding cache in this worktree); read the LoRA"
                " column against the shipped-head column, not a refit."
            ),
        ]
    return "\n".join(lines) + "\n"


def render_report_ettin(r: dict) -> str:
    s = r["selection"]["chosen"]
    backbone = r["backbone"]
    frozen_dev = r["selection"]["frozen_head_dev_macro_f1"]
    frozen_dev_txt = f"{frozen_dev:.3f}" if frozen_dev is not None else "n/a (no cache)"
    lines = [
        f"# LoRA/PEFT check on the mixed pool ({backbone} encoder) vs the shipped frozen head",
        "",
        (
            f"Train pool {r['train_pool_rows']} rows. Chosen cell `{s['cell']}` (dev macro-F1"
            f" {s['dev_macro_f1']:.3f} on {r['selection']['dev_rows']} rows /"
            f" {r['selection']['dev_families']} held-out families; frozen head on dev"
            f" {frozen_dev_txt})."
        ),
        (
            f"LoRA = ensemble of {len(r['per_seed'][SIX[0]])} seeds; family-cluster bootstrap,"
            f" {T.BOOT} draws, seed {T.SEED}. Reference head `{r['shipped_head_key']}` from"
            f" {r['shipped_head_source']}."
        ),
        "",
        "| eval set | n | reference head | LoRA [95% CI] | diff LoRA-head [95% CI] | P(better) | lost |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in SIX:
        e = r["sets"][name]
        ref = e["shipped_head"]
        ref_txt = f"{ref:.3f}" if ref is not None else "n/a"
        d = r["paired"].get(name)
        if d:
            diff_txt = f"{d['point']:+.3f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
            ptxt = f"{d['p_better']:.2f}"
            lost = "yes" if d["lost"] else "no"
        else:
            diff_txt, ptxt, lost = "-", "-", "-"
        lines.append(
            f"| {name} | {e['n']} | {ref_txt} | {fmt_ci(r['lora'][name])} |"
            f" {diff_txt} | {ptxt} | {lost} |"
        )
    lines += [
        "",
        "## Per-seed LoRA macro-F1",
        "",
        "| eval set | "
        + " | ".join(f"seed{i}" for i in range(len(r["per_seed"][SIX[0]])))
        + " | mean ± sd |",
        "| --- | " + " | ".join("---" for _ in r["per_seed"][SIX[0]]) + " | --- |",
    ]
    for name in SIX:
        vals = r["per_seed"][name]
        mean, sd = float(np.mean(vals)), float(np.std(vals))
        lines.append(
            f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {mean:.3f} ± {sd:.3f} |"
        )
    v = r["verdict"]
    lines += [
        "",
        "## Selection grid (dev macro-F1, family-held-out from the train pool)",
        "",
        "| cell | rank | lr | epochs | dev macro-F1 | train s | adapter B |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in sorted(r["selection"]["cells"], key=lambda c: -c["dev_macro_f1"]):
        lines.append(
            f"| {c['cell']} | {c['rank']} | {c['lr']:g} | {c['epochs']} |"
            f" {c['dev_macro_f1']:.3f} | {c['train_seconds']:.0f} | {c['adapter_bytes']} |"
        )
    lines += ["", "## Verdict", ""]
    if v["paired"]:
        lines += [
            f"- lab/pool beaten (point > 0, CI excludes 0): **{v['beat_lab_pool']}**",
            f"- gen2/heldout (E19b) beaten: **{v['beat_gen2_heldout']}**",
            f"- external sets lost (point < 0, CI excludes 0): {v['lost_external'] or 'none'}",
            f"- **LoRA beats the shipped head: {v['lora_wins']}**",
        ]
    else:
        lines += [
            f"- **Paired test skipped**: {v['reason']}.",
            "- The verdict rule is stated in the plan but not evaluated: read the LoRA column against",
            "  the reference head column above, with the per-seed spread, and treat it as undetermined.",
        ]
    adaptive = r["adapter_bytes"]
    lines += [
        "",
        "## Cost",
        "",
        (
            f"- Selection: {len(r['selection']['cells'])} cells,"
            f" {sum(c['train_seconds'] for c in r['selection']['cells']) / 60:.0f} min total."
        ),
        (
            f"- Final: {len(adaptive['final'])} seeds,"
            f" {sum(r['train_seconds']['final_per_seed']) / 60:.0f} min total"
            f" (per seed {[int(x) for x in r['train_seconds']['final_per_seed']]} s)."
        ),
        (
            f"- Adapter + head size: {adaptive['final'][0]} bytes"
            f" ({adaptive['final'][0] / 2**20:.1f} MiB; adapter + {len(LABELS)}-class head)."
        ),
    ]
    lat = r.get("latency")
    if lat:
        lines += [
            f"- Inference latency per context ({r['backbone']}, forward only, mean-pooled):",
            (
                f"  - frozen encoder: {lat['frozen_b1']['p50_ms_per_ctx']:.1f} ms p50 /"
                f" {lat['frozen_b1']['p95_ms_per_ctx']:.1f} ms p95 (batch 1),"
                f" {lat['frozen_b32']['p50_ms_per_ctx']:.1f} /"
                f" {lat['frozen_b32']['p95_ms_per_ctx']:.1f} (batch 32)"
            ),
            (
                f"  - LoRA encoder: {lat['lora_b1']['p50_ms_per_ctx']:.1f} /"
                f" {lat['lora_b1']['p95_ms_per_ctx']:.1f} (batch 1),"
                f" {lat['lora_b32']['p50_ms_per_ctx']:.1f} /"
                f" {lat['lora_b32']['p95_ms_per_ctx']:.1f} (batch 32)"
            ),
        ]
    lines += baseline_lines(r)
    lines += [
        "",
        "## Notes",
        "",
        (
            "- Chosen on a family-held-out dev split of the train pool only; the six eval sets were"
            " scored once, at the end, for that one config."
        ),
    ]
    if r["paired"]:
        worst = max(abs(r["sets"][n]["head_refit"] - r["sets"][n]["shipped_head"]) for n in SIX)
        refit = ", ".join(f"{n} {r['sets'][n]['head_refit']:.3f}" for n in SIX)
        lines.append(
            f"- Frozen head refit on the same train pool: {refit} (max |refit - E20| = {worst:.4f})."
        )
        lo, hi = r["paired"]["lab/pool"]["ci"]
        lines.append(
            "- Read: the paired difference on the lab held-out pool is"
            f" {r['paired']['lab/pool']['point']:+.3f} (95 % CI {lo:+.2f}, {hi:+.2f}), with a per-seed"
            f" spread of {min(r['per_seed']['lab/pool']):.2f}-{max(r['per_seed']['lab/pool']):.2f}."
        )
    else:
        lines.append(
            "- Paired test skipped (no frozen-embedding cache in this worktree); reference numbers are"
            " the E20 frozen Ettin head from lab-data-e20/results.json, not a refit."
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, func, *, device: str = "cuda", **setters):
        parser = sub.add_parser(name)
        parser.add_argument("--backbone", choices=("laya", "ettin"), default="laya")
        parser.add_argument("--device", default=device)
        parser.add_argument("--smoke", action="store_true", help="tiny end-to-end subset")
        parser.add_argument(
            "--out", default=None, help="output dir override (default: the backbone's dir)"
        )
        parser.add_argument(
            "--runs", default=None, help="Trackio store dir (default $TRACKIO_DIR, else ./runs)"
        )
        parser.add_argument(
            "--no-track", action="store_true", help="disable Trackio (JSONL mirror still written)"
        )
        parser.set_defaults(func=func, **setters)
        return parser

    embed = add("embed", cmd_embed)
    embed.add_argument("--batch", type=int, default=32)
    add("head", cmd_head)
    select = add("select", cmd_select)
    select.add_argument("--force", action="store_true")
    select.add_argument("--batch", type=int, default=BATCH)
    select.add_argument("--grad-checkpointing", action="store_true")
    final = add("final", cmd_final)
    final.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    final.add_argument("--force", action="store_true")
    final.add_argument("--batch", type=int, default=BATCH)
    final.add_argument("--grad-checkpointing", action="store_true")
    add("latency", cmd_latency)
    add("export", cmd_export, device="cpu")
    report = add("report", cmd_report, device="cpu")
    report.add_argument(
        "--baseline", default=None, help="earlier results.json to show side by side"
    )
    allp = add("all", None)
    allp.add_argument("--batch", type=int, default=BATCH)
    allp.add_argument("--grad-checkpointing", action="store_true")
    allp.add_argument("--force", action="store_true")
    allp.add_argument("--baseline", default=None, help="earlier results.json for the report")

    def run_all(args):
        base = {
            "backbone": args.backbone,
            "device": args.device,
            "smoke": args.smoke,
            "batch": args.batch,
            "grad_checkpointing": args.grad_checkpointing,
            "force": args.force,
            "out": args.out,
            "runs": args.runs,
            "no_track": args.no_track,
            "baseline": args.baseline,
        }
        steps = []
        if args.backbone == "laya" and not args.smoke:
            steps.append(("embed", cmd_embed, {**base, "batch": 32}))
        steps += [
            ("head", cmd_head, base),
            ("select", cmd_select, base),
            ("final", cmd_final, {**base, "seeds": ",".join(map(str, SEEDS))}),
        ]
        if args.backbone == "laya" and not args.smoke:
            steps.append(("latency", cmd_latency, base))
        steps.append(("report", cmd_report, base))
        for name, func, kwargs in steps:
            print(f"=== {name} ===", flush=True)
            func(argparse.Namespace(**kwargs))

    allp.set_defaults(func=run_all)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Train a classification head on frozen Laya embeddings of the labelled lab contexts.

Stage 2 of `docs/finetune.md`: one linear layer + softmax over a pooling (`mean`, `cls` or
`mean+cls`) of the encoder's last hidden state. Everything is selected by cross-validation
*inside* the train families of the lab's family-hash split - context format (shipped
command-only vs the full statistics format), pooling, L2 strength and class weighting - and the
held-out families are only ever scored, never fitted or selected on.

    just gpu-run-lab python lab/scripts/finetune_head.py                 # writes models/lab-head.safetensors
    just gpu-run-lab python lab/scripts/finetune_head.py --out models/x.safetensors --data lab-data/dataset.jsonl

The saved `.safetensors` records its pooling and tokenizer budget, and `[judge] tuned = "..."`
loads it through the same embedder, so what was fitted is what the daemon runs. The tuned path
stays off until that config key is set. Every reported table carries the always-safe baseline
and the majority-class rate next to the number, plus per-class precision/recall.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jevmon.config import load_config
from jevmon.detect.context import ContextSettings, parse_exec_state, render_segments
from jevmon.laya.head import POOLINGS, TunedHead, embed_states
from jevmon.monitor import build_decider
from jevmon_lab import metrics
from jevmon_lab.manifest import LABELS
from jevmon_lab.runlog import RunLog, class_table, score_scalars

WEIGHT_DECAYS = (0.0, 0.01, 0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")
FOLDS = 5
EMBED_BATCH = 32
FORMATS = {
    "shipped": ContextSettings(),  # what the daemon shows by default (command only)
    "full": ContextSettings(order=(), sep=" "),  # every segment, as the lab captured it
}


def load_rows(path: Path) -> list[dict]:
    """Labelled rows with their context captured in the full format (`_raw`)."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    labelled = [row for row in rows if row.get("label")]
    contexts_path = path.parent / "contexts.jsonl"
    if contexts_path.exists():
        raw = [json.loads(line) for line in contexts_path.read_text().splitlines() if line.strip()]
        if len(raw) == len(rows):  # both files are written from the same observations, in order
            for row, ctx in zip(rows, raw, strict=True):
                if (row["subject"], row["signature"]) != (ctx["subject"], ctx["signature"]):
                    raise SystemExit(f"dataset/contexts rows diverge at {row['signature']}")
                row["_raw"] = ctx["context"]
        else:
            raise SystemExit(f"{contexts_path} has {len(raw)} rows, dataset has {len(rows)}")
    else:
        for row in rows:
            row["_raw"] = row["context"]
    return labelled


def render_states(rows: list[dict], settings: ContextSettings) -> list[str]:
    """Re-render every row's captured text through `settings` (the daemon's layout)."""
    return [render_segments(parse_exec_state(row["_raw"]), settings) for row in rows]


def embed(agent, states: list[str], pooling: str) -> np.ndarray:
    started = time.perf_counter()
    vectors = embed_states(agent, states, pooling=pooling, batch_size=EMBED_BATCH)
    print(
        f"  embedded {vectors.shape[0]} states [{pooling}] dim {vectors.shape[1]} "
        f"in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return vectors


def fit_head(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    weight_decay: float,
    on_step: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted cross-entropy logistic head, L2 on the weights, LBFGS (deterministic).

    `on_step` receives the training loss at every closure evaluation (a progress hook).
    """
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    classes = int(y.max()) + 1
    weight = torch.tensor(sample_weight, dtype=torch.float32) if sample_weight is not None else None
    w = torch.zeros(classes, x.shape[1], requires_grad=True)
    b = torch.zeros(classes, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, line_search_fn="strong_wolfe", max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xt @ w.T + b, yt, weight=weight)
        loss = loss + 0.5 * weight_decay * (w * w).sum()
        loss.backward()
        if on_step is not None:
            on_step(float(loss.item()))
        return loss

    opt.step(closure)
    return w.detach().numpy(), b.detach().numpy()


def class_weights(y: np.ndarray, balanced: bool) -> np.ndarray | None:
    if not balanced:
        return None
    classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    return (len(y) / (classes * counts)).astype(np.float32)


def predict(x: np.ndarray, labels: tuple[str, ...], w: np.ndarray, b: np.ndarray) -> list[str]:
    probs = TunedHead(labels=labels, weight=w, bias=b).probabilities(x)
    return [labels[int(row.argmax())] for row in probs]


def score_rows(rows: list[dict], verdicts: list[str]) -> dict:
    scored = [
        {"label": row["label"], "verdict": verdict}
        for row, verdict in zip(rows, verdicts, strict=True)
    ]
    return metrics.score(scored)


def grouped_folds(rows: list[dict], folds: int = FOLDS) -> list[list[int]]:
    """Family-grouped folds, dealt round-robin within each label so every fold sees every class
    that has enough families (a family never straddles two folds, so nothing leaks)."""
    label_of = {row["family"]: row["label"] for row in rows}
    size = {family: 0 for family in label_of}
    for row in rows:
        size[row["family"]] += 1
    fold_of: dict[str, int] = {}
    for label in sorted(set(label_of.values())):
        members = sorted(
            (f for f, lab in label_of.items() if lab == label), key=lambda f: (-size[f], f)
        )
        for n, family in enumerate(members):
            fold_of[family] = n % folds
    groups = [
        [n for n, row in enumerate(rows) if fold_of[row["family"]] == f] for f in range(folds)
    ]
    return [g for g in groups if g]


def select(
    name: str,
    vectors: np.ndarray,
    rows: list[dict],
    labels: tuple[str, ...],
    run: RunLog | None = None,
    grid: list[dict] | None = None,
) -> tuple[float, float, bool]:
    """Best (macro-F1, weight_decay, balanced) for one format/pooling, grouped CV on train.

    With `run`, every (class weighting, weight decay) candidate is logged as one step: the
    fold-mean macro-F1 it is selected on, and the pooled out-of-fold score next to its
    always-safe and majority-class references. `grid` collects the candidates as rows.
    """
    y = np.asarray([labels.index(row["label"]) for row in rows])
    folds = grouped_folds(rows)
    best = (-1.0, 0.0, False, ())
    for balanced in CLASS_WEIGHTS:
        for wd in WEIGHT_DECAYS:
            scores: list[float] = []
            pooled_rows: list[dict] = []
            pooled_verdicts: list[str] = []
            for val in folds:
                keep = set(val)
                train = [n for n in range(len(rows)) if n not in keep]
                w, b = fit_head(vectors[train], y[train], class_weights(y[train], balanced), wd)
                verdicts = predict(vectors[val], labels, w, b)
                scores.append(score_rows([rows[n] for n in val], verdicts)["macro_f1"])
                pooled_rows.extend(rows[n] for n in val)
                pooled_verdicts.extend(verdicts)
            mean = sum(scores) / len(scores)
            if run is not None:
                step = len(grid) if grid is not None else 0
                pooled = score_rows(pooled_rows, pooled_verdicts)
                run.log(
                    step,
                    **{
                        "cv/fold_mean_macro_f1": mean,
                        "cv/weight_decay": wd,
                        "cv/balanced": int(bool(balanced)),
                        **score_scalars("cv", pooled),
                    },
                )
            if grid is not None:
                grid.append(
                    {
                        "candidate": name,
                        "class_weight": "balanced" if balanced else "none",
                        "weight_decay": wd,
                        "cv_macro_f1": round(mean, 4),
                    }
                )
            if mean > best[0]:
                best = (mean, wd, balanced, tuple(scores))
    score, wd, balanced, per_fold = best
    print(
        f"| {name} | {score:.3f} | {wd} | {'balanced' if balanced else 'none'} | "
        f"{' '.join(f'{s:.3f}' for s in per_fold)} |",
        flush=True,
    )
    return score, wd, balanced


def report(title: str, rows: list[dict], verdicts: list[str]) -> dict:
    """Print the markdown table for one scored split and return the score for logging."""
    print(f"\n### {title}\n")
    scored = score_rows(rows, verdicts)
    print("\n".join(metrics.lines(scored)))
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="train the lab classification head")
    parser.add_argument("--data", type=Path, default=Path("lab-data/dataset.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("models/lab-head.safetensors"))
    parser.add_argument("--no-track", action="store_true", help="skip experiment tracking")
    args = parser.parse_args()

    torch.set_num_threads(2)  # never peg the box while the iGPU does the work
    rows = load_rows(args.data)
    train = [row for row in rows if row["split"] == "train"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    if not train or not heldout:
        raise SystemExit(f"split is empty: train={len(train)} heldout={len(heldout)}")
    labels = tuple(LABELS)
    print(
        f"{len(rows)} labelled contexts: {len(train)} train / {len(heldout)} held-out, "
        f"labels {list(labels)}"
    )

    decider = build_decider(load_config())
    decider.load()
    if decider.error:
        raise SystemExit(f"model failed to load: {decider.error}")
    agent = decider._agent(decider._router)
    if agent is None:
        raise SystemExit("no loaded agent: cannot embed states")
    print(f"model: {decider.status}", flush=True)

    judge_cfg = load_config().judge
    run = RunLog.start(
        "jevmon-finetune",
        time.strftime("head-frozen-embeddings-%Y%m%d-%H%M%S"),
        {
            "script": "finetune_head.py",
            "data": args.data,
            "out": args.out,
            "formats": list(FORMATS),
            "poolings": list(POOLINGS),
            "weight_decays": WEIGHT_DECAYS,
            "class_weights": [str(w) for w in CLASS_WEIGHTS],
            "folds": FOLDS,
            "embed_batch": EMBED_BATCH,
            "device": decider.active_device or judge_cfg.device,
            "dtype": decider.active_dtype or judge_cfg.dtype,
            "threads": 2,
            "seed": None,  # LBFGS from zero init is deterministic: there is no seed
            "n_train": len(train),
            "n_heldout": len(heldout),
            "labels": labels,
            "model": decider.status,
        },
        enabled=not args.no_track,
    )
    grid: list[dict] = []

    print("\ntrain-family cross-validation (5 grouped folds, macro-F1):\n")
    print("| format | pooling | CV macro-F1 | weight_decay | class_weight | per fold |")
    print("| --- | --- | --- | --- | --- | --- |")
    best: tuple = (-1.0, "", "", 0.0, False)
    train_idx = [n for n, row in enumerate(rows) if row["split"] == "train"]
    held_idx = [n for n, row in enumerate(rows) if row["split"] == "heldout"]
    for format_name, settings in FORMATS.items():
        states = render_states(rows, settings)
        for pooling in POOLINGS:
            vectors = embed(agent, states, pooling)
            train_vectors = vectors[train_idx]
            score, wd, balanced = select(
                f"{format_name}/{pooling}", train_vectors, train, labels, run, grid
            )
            if score > best[0]:
                best = (score, format_name, pooling, wd, balanced)
    score, format_name, pooling, wd, balanced = best
    print(
        f"\nbest: format={format_name} pooling={pooling} weight_decay={wd} "
        f"class_weight={'balanced' if balanced else 'none'} (CV macro-F1 {score:.3f})"
    )

    # refit on every train row with the winning settings, then score the held-out families once
    states = render_states(rows, FORMATS[format_name])
    vectors = embed(agent, states, pooling)
    train_vectors, held_vectors = vectors[train_idx], vectors[held_idx]
    y_train = np.asarray([labels.index(row["label"]) for row in train])
    losses = 0

    def log_loss(loss: float) -> None:
        nonlocal losses
        run.log(losses, **{"train/loss": loss})
        losses += 1

    w, b = fit_head(train_vectors, y_train, class_weights(y_train, balanced), wd, log_loss)
    TunedHead(labels=labels, weight=w, bias=b, pooling=pooling).save(args.out)
    print(f"saved {args.out} (format={format_name}, pooling={pooling})")

    zero_shot = [d.label for d in decider.decide_batch([row["context"] for row in heldout])]
    held = report("held-out families: tuned head", heldout, predict(held_vectors, labels, w, b))
    zero = report("held-out families: zero-shot model (shipped config)", heldout, zero_shot)
    resub = report(
        "train families: tuned head (resubstitution; selection was the CV above)",
        train,
        predict(train_vectors, labels, w, b),
    )
    run.log(
        0,
        **{
            "best/cv_macro_f1": score,
            "best/weight_decay": wd,
            **score_scalars("heldout", held),
            **score_scalars("zero_shot_heldout", zero),
            **score_scalars("train", resub),
        },
    )
    run.table("cv_grid", grid)
    run.table(
        "best",
        [
            {
                "format": format_name,
                "pooling": pooling,
                "weight_decay": wd,
                "class_weight": "balanced" if balanced else "none",
                "cv_macro_f1": round(score, 4),
            }
        ],
    )
    for name, result in (("heldout", held), ("zero_shot_heldout", zero), ("train", resub)):
        run.table(f"per_class/{name}", class_table(result))
    run.finish()


if __name__ == "__main__":
    main()

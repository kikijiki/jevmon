"""Full fine-tune of the Laya checkpoint on the lab's train families (stage 2b).

The linear head (stage 2a) was not enough, so this escalates to the prescribed next step: a
full fine-tune of the checkpoint with `laya.train` (encoder + decision head, soft cross-entropy,
option-order shuffling). Training rows are the **train families** of the lab's family-hash
split, rendered as exactly the question and categories the judge asks at inference - same
instructions, same criteria text - so the model learns the task it will be asked. Held-out
families are never trained or calibrated on; they are scored once, at the end.

    just gpu-run-lab python lab/scripts/finetune_laya.py                 # writes models/lab-ft/
    just gpu-run-lab python lab/scripts/finetune_laya.py --epochs 8

The result is loaded by `[judge] checkpoint = "lab-ft"` + `[judge] checkpoint_dir =
"models/lab-ft"` (both shipped off), registered on the router and answered through the same
`router.predict` path as the built-in checkpoint. Hyperparameters are fixed a priori (laya's
defaults, `loss=soft-ce`, `grad_accum=1` so the 124-row set produces dozens of updates rather
than a handful); the only acceptance signal reported from inside the split is train-family
resubstitution.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # shared helpers live next to this script

from finetune_head import load_rows, render_states, report
from jevmon.config import load_config
from jevmon.detect.context import ContextSettings
from jevmon.detect.judge import JudgeSettings
from jevmon.monitor import build_decider
from jevmon_lab.runlog import RunLog, class_table, score_scalars

BASE_CHECKPOINT = "convaiinnovations/laya"  # laya's english checkpoint, in the HF cache
FINE_TUNE_NAME = "lab-ft"


def training_rows(rows: list[dict], judge: JudgeSettings, split: str) -> list[dict]:
    """`{state, questions, expected}` rows for `split`, carrying the judge's own question."""
    questions = {
        "attention": {
            "type": "choice",
            "instructions": judge.question,
            "criteria": dict(judge.categories),
        }
    }
    states = render_states(rows, ContextSettings())  # shipped layout, as the judge reads it
    return [
        {
            "state": state,
            "questions": questions,
            "expected": {"attention": row["label"]},
        }
        for row, state in zip(rows, states, strict=True)
        if row["split"] == split
    ]


def write_dataset(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))
    return len(rows)


def judge_config(base, checkpoint_dir: str):
    return replace(
        base,
        judge=replace(base.judge, checkpoint=FINE_TUNE_NAME, checkpoint_dir=str(checkpoint_dir)),
    )


def track_train_loss(run: RunLog, train_module) -> None:
    """Log laya's per-step training loss: wrap the loss function `train_model` calls.

    laya reports progress only on stdout, so the soft cross-entropy it minimises is wrapped
    once, here, and every micro-batch's value is sent to `run` as `train/loss`.
    """
    original = train_module.soft_ce_loss
    step = 0

    def wrapped(logits, target, mask):
        nonlocal step
        loss = original(logits, target, mask)
        run.log(step, **{"train/loss": float(loss.item())})
        step += 1
        return loss

    train_module.soft_ce_loss = wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description="full fine-tune of the lab judge")
    parser.add_argument("--data", type=Path, default=Path("lab-data/dataset.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("models/lab-ft"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--no-track", action="store_true", help="skip experiment tracking")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # cache only; the checkpoint is local
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
    base = load_config()
    rows = load_rows(args.data)
    train_rows = training_rows(rows, base.judge, "train")
    heldout = [row for row in rows if row["split"] == "heldout"]
    train = [row for row in rows if row["split"] == "train"]
    print(
        f"{len(train_rows)} training rows from {len(train)} train-family contexts; "
        f"{len(heldout)} held-out contexts stay untouched"
    )

    import laya.train as laya_train
    from laya.train import TrainConfig, finetune

    data_path = args.out.parent / "lab-ft-train.jsonl"
    write_dataset(train_rows, data_path)
    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=8,
        grad_accum=args.grad_accum,
        loss="soft-ce",  # hard labels on a small set: plain cross-entropy, no rlcd sampling
        shuffle_options=("choice",),  # no option-position prior to leak into inference
        freeze_encoder=False,  # the full fine-tune the brief asks for
        seed=0,
    )
    config.validate()
    run = RunLog.start(
        "finetune-laya",
        time.strftime("ft-%Y%m%d-%H%M%S"),
        {
            "script": "finetune_laya.py",
            "base": BASE_CHECKPOINT,
            "data": args.data,
            "out": args.out,
            "epochs": config.epochs,
            "micro_batch": config.micro_batch,
            "grad_accum": config.grad_accum,
            "loss": config.loss,
            "freeze_encoder": config.freeze_encoder,
            "shuffle_options": config.shuffle_options,
            "encoder_lr": config.encoder_lr,
            "head_lr": config.head_lr,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "format": "shipped",
            "pooling": "laya decision head (no pooling choice)",
            "device": base.judge.device,
            "dtype": base.judge.dtype,
            "n_train": len(train),
            "n_heldout": len(heldout),
        },
        enabled=not args.no_track,
    )
    track_train_loss(run, laya_train)
    print(
        f"fine-tuning {BASE_CHECKPOINT} -> {args.out} "
        f"(epochs={config.epochs}, micro_batch={config.micro_batch}, "
        f"grad_accum={config.grad_accum}, loss={config.loss}, freeze_encoder="
        f"{config.freeze_encoder})",
        flush=True,
    )
    summary = finetune(
        data=str(data_path),
        model_dir=BASE_CHECKPOINT,
        output_dir=str(args.out),
        config=config,
        device="auto",
    )
    print(
        f"trained: {summary['train_items']} train / {summary['calibration_items']} calibration "
        f"items, epoch losses {[f'{x:.3f}' for x in summary['epoch_loss']]}"
    )
    per_epoch = math.ceil(summary["train_items"] / config.micro_batch)
    for epoch, loss in enumerate(summary["epoch_loss"]):
        run.log((epoch + 1) * per_epoch - 1, **{"train/epoch_loss": float(loss)})

    held_states = [row["context"] for row in heldout]
    if not args.skip_zero_shot:
        zero = build_decider(base)
        zero.load()
        if zero.error:
            raise SystemExit(f"zero-shot model failed to load: {zero.error}")
        zero_verdicts = [d.label for d in zero.decide_batch(held_states)]
        zero_score = report(
            "held-out families: zero-shot model (shipped config)", heldout, zero_verdicts
        )
        run.log(0, **score_scalars("zero_shot_heldout", zero_score))
        run.table("per_class/zero_shot_heldout", class_table(zero_score))
        del zero
        gc.collect()

    tuned = build_decider(judge_config(base, args.out))
    tuned.load()
    if tuned.error:
        raise SystemExit(f"fine-tuned model failed to load: {tuned.error}")
    print(f"fine-tuned model: {tuned.status}", flush=True)
    held_verdicts = [d.label for d in tuned.decide_batch(held_states)]
    held_score = report("held-out families: full fine-tune", heldout, held_verdicts)
    train_states = [row["context"] for row in train]
    train_verdicts = [d.label for d in tuned.decide_batch(train_states)]
    train_score = report(
        "train families: full fine-tune (resubstitution; the only in-split acceptance signal)",
        train,
        train_verdicts,
    )
    epochs_done = len(summary["epoch_loss"])
    final = epochs_done * per_epoch
    run.log(
        final,
        **score_scalars("heldout", held_score),
        **score_scalars("train", train_score),
        **{"train/epochs": epochs_done},
    )
    run.table("per_class/heldout", class_table(held_score), step=final)
    run.table("per_class/train", class_table(train_score), step=final)
    run.finish()
    print(
        f'\nload it with [judge] checkpoint = "{FINE_TUNE_NAME}" and '
        f'[judge] checkpoint_dir = "{args.out}"'
    )


if __name__ == "__main__":
    main()

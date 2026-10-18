"""Latency of the shipped `laya` and `laya-lora` judges through the app's own decider path.

Both judges are built the way the daemon builds them (`jevmon.monitor.build_decider` from a
`[judge] model` setting) and timed on realistic contexts from the real context builder
(`bench_judge.build_states`). `laya` selects the packaged frozen head; `laya-lora` merges the
packaged LoRA adapter into the encoder at load. The merge leaves no runtime `peft` and no extra
forward pass, so the two should cost about the same; cells are interleaved and reported as
per-context percentiles.

    just gpu-run-lab python lab/scripts/bench_lora.py --device gpu --batches 1,8,32
"""

import argparse
import time
from dataclasses import replace

import numpy as np

import bench_judge
from jevmon.config import load_config
from jevmon.monitor import build_decider

MODELS = ("laya", "laya-lora")


def load(model: str, device: str):
    """The app's decider for `model`, fully loaded (includes the warm-up forward)."""
    config = load_config()
    config = replace(config, judge=replace(config.judge, model=model, device=device))
    decider = build_decider(config)
    started = time.perf_counter()
    decider.load()
    if decider.error:
        raise SystemExit(f"{model}: {decider.error}")
    return decider, time.perf_counter() - started


def time_pool(decider, states: list[str], batch: int) -> list[float]:
    """Per-context wall ms for one pass over `states` in chunks of `batch`."""
    decider.decide_batch(states[:batch])  # warm the pool/buffers, not timed
    per_ctx: list[float] = []
    for start in range(0, len(states), batch):
        chunk = states[start : start + batch]
        began = time.perf_counter()
        decider.decide_batch(chunk)
        per_ctx.append((time.perf_counter() - began) * 1000.0 / len(chunk))
    return per_ctx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=256, help="context pool size")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batches", default="1,8,32")
    args = parser.parse_args()

    states = bench_judge.build_states(args.states)
    batches = tuple(int(b) for b in args.batches.split(","))
    print(f"{len(states)} contexts, batches {batches}", flush=True)

    labels: dict[str, list[str]] = {}
    cells: list[tuple[str, object, int]] = []
    for model in MODELS:
        decider, load_s = load(model, args.device)
        print(f"{model}: load {load_s:.1f}s, status {decider.status}", flush=True)
        labels[model] = [d.label for d in decider.decide_batch(states)]
        cells += [(f"{model} batch={batch}", decider, batch) for batch in batches]

    samples: dict[str, list[float]] = {label: [] for label, _, _ in cells}
    for round_index in range(args.rounds):  # interleave, alternating direction, so one load
        order = cells if round_index % 2 == 0 else list(reversed(cells))  # spike cannot rank
        for label, decider, batch in order:
            samples[label].extend(time_pool(decider, states, batch))
        print(f"round {round_index + 1}/{args.rounds} done", flush=True)

    print("\n| cell | ms/context p50 | ms/context p95 |")
    print("| --- | --- | --- |")
    for label, _, _ in cells:
        arr = np.asarray(samples[label])
        print(f"| {label} | {np.percentile(arr, 50):.1f} | {np.percentile(arr, 95):.1f} |")

    base, other = MODELS
    differ = sum(1 for a, b in zip(labels[base], labels[other], strict=True) if a != b)
    print(f"\nlabels: {differ}/{len(states)} differ between {base} and {other}")


if __name__ == "__main__":
    main()

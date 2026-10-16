"""Judge benchmark: ms per context vs device, dtype, batch size and torch threads.

Realistic contexts come from the real `ContextBuilder` (varied 400-1000 char exec
states), and the timed work is the shipped `LayaDecider.decide_batch` over those states. Each
cell reports the median of interleaved rounds, plus CPU seconds per context, peak GTT/VRAM and
(iGPU runs) GPU busy%, so a GPU cell shows both what it costs and what the CPU stopped doing.

    # CPU matrix (project venv, CPU torch wheel): threads x batches
    just run-lab python lab/scripts/bench_judge.py --device cpu --threads 8 --batches 1,8,32

    # iGPU (ROCm torch wheel env): dtypes x batches
    python lab/scripts/bench_judge.py --device gpu --dtype bf16,fp16,fp32 --batches 1,8,32

    # forward-pass width on one device (alias: --cap-sweep)
    python lab/scripts/bench_judge.py --device gpu --pass-sweep --dtype bf16

    # verdict parity: CPU fp32 vs every GPU dtype on the same states (needs one process
    # that can run both devices, i.e. the ROCm torch wheel env)
    python lab/scripts/bench_judge.py --parity
"""

import argparse
import random
import statistics
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Self

from jevmon.core import MemoryCounts
from jevmon.detect.context import ContextBuilder, EnvironmentStats, ExecObservation
from jevmon.detect.system_facts import RealSystemFacts
from jevmon.laya.decider import LayaConfig, LayaDecider

BATCHES = (1, 4, 8, 16, 32)
THREADS = (4, 8)
PASS_SIZES = (0, 1, 2, 4, 8, 16, 32)  # 0 = the shipped per-device default

PROGS = [
    "/nix/store/abc123def456-python3-3.13.15/bin/python3",
    "/usr/bin/bash",
    "/usr/bin/git",
    "/home/user/.local/bin/rg",
    "/usr/libexec/openssh/ssh-session-cleanup",
    "/opt/very/long/path/to/a/vendor/toolchain/bin/aarch64-unknown-linux-gnu-gcc",
    "node",
    "/usr/bin/true",
]
PARENTS = [
    "/usr/bin/bash",
    "/nix/store/abc123def456-python3-3.13.15/bin/python3",
    "/usr/bin/systemd",
    "/home/user/.cargo/bin/cargo",
    "/usr/bin/npm",
]
ARGS = [
    "-c",
    "while true; do sleep 1; done",
    "--model",
    "english",
    "--output-format=json",
    "/home/user/projects/agent/scratchpad/run-2026-10-12.log",
    "-rf",
    "--experimental-fetch",
    "packfile window memory limit configuration override",
    "spawn --name builder-7 -- cp -r /var/tmp/stage /var/tmp/stage-2",
    "GET /api/v1/events?limit=100&cursor=eyJpZCI6MTIzfQ==",
    "git status --porcelain=v2 --branch",
    "cargo build --release --target x86_64-unknown-linux-gnu -j8",
    "nix develop --command uv run pytest -q tests/test_judge.py",
    "--header",
    "Authorization: Bearer abcdef0123456789",
    "--very-long-option-value-with-spaces that keeps the argv line interesting for padding",
]
CWDS = ["/home/user/dev/jevmon", "/tmp", "/home/user/.cache", "/var/lib/containers"]
CGROUPS = ["user.slice/user-1000.slice/session-3.scope", "system.slice/cron.service"]


def build_states(n: int) -> list[str]:
    """Exec contexts through the real builder: varied lengths, learning counts as they go."""
    builder = ContextBuilder(MemoryCounts(), RealSystemFacts())
    env = EnvironmentStats(fork_rate=12.5, fork_baseline=10.0, miss_ratio=0.42)
    rng = random.Random(7)
    now = time.time()
    states = []
    for i in range(n):
        exe = rng.choice(PROGS)
        argv = (exe, *rng.sample(ARGS, rng.randint(0, 5)))
        parent = rng.choice(PARENTS)
        obs = ExecObservation(
            time=now,
            subject=str(1000 + i),
            uid=1000,
            exe=exe,
            path0=exe,
            argv0=argv[0],
            argv=argv,
            cwd=rng.choice(CWDS),
            cgroup=rng.choice(CGROUPS),
            parent_comm=parent.rsplit("/", 1)[-1],
            parent_exe=parent,
            parents=f"{parent} -> {exe}",
        )
        states.append(builder.observe_exec(obs, env).state)
    rng.shuffle(states)
    return states


def load_decider(
    device: str, dtype: str = "auto", threads: int = 8, threads_gpu: int = 2
) -> tuple[LayaDecider, float]:
    """A fully loaded decider through the shipped path (slow; includes the warm-up forward)."""
    started = time.perf_counter()
    decider = LayaDecider(
        LayaConfig(
            offline=True,
            load_timeout=180.0,
            device=device,
            dtype=dtype,
            threads=threads,
            threads_gpu=threads_gpu,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decider.load()
    if decider.error or decider._router is None:
        raise SystemExit(f"model unavailable: {decider.error}")
    return decider, time.perf_counter() - started


def fresh_decider(router, threads: int, pass_size: int = 0, device: str = "cpu") -> LayaDecider:
    """A new wrapper around an already-loaded router (one per thread/pass setting)."""
    decider = LayaDecider(
        LayaConfig(offline=True, threads=threads, pass_size=pass_size, device=device)
    )
    decider._router = router
    decider._active_device = "gpu" if device == "gpu" else "cpu"
    decider._ready.set()
    return decider


class Sensors:
    """Background sampler of GPU busy% and drm memory; busy() reports since the last mark()."""

    def __init__(self) -> None:
        self.baseline: dict[str, int] = {}
        self.samples: list[tuple[int, int, int]] = []  # (busy, gtt_used, vram_used)
        self._mark = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _read(name: str) -> int | None:
        for card in sorted(Path("/sys/class/drm").glob("card*/device")):
            try:
                return int((card / name).read_text())
            except (OSError, ValueError):
                continue
        return None

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            busy = self._read("gpu_busy_percent")
            gtt = self._read("mem_info_gtt_used")
            vram = self._read("mem_info_vram_used")
            if busy is not None and gtt is not None:
                self.samples.append((busy, gtt, vram or 0))

    def __enter__(self) -> Self:
        self._thread.start()
        time.sleep(0.3)
        self.baseline = self.peak()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(1)

    def peak(self) -> dict[str, int]:
        """Run-wide peak of each sampled counter (starting from the pre-run baseline)."""
        out = dict(self.baseline)
        for busy, gtt, vram in self.samples:
            out["gpu_busy_percent"] = max(out.get("gpu_busy_percent", 0), busy)
            out["mem_info_gtt_used"] = max(out.get("mem_info_gtt_used", 0), gtt)
            out["mem_info_vram_used"] = max(out.get("mem_info_vram_used", 0), vram)
        return out

    def mark(self) -> None:
        self._mark = len(self.samples)

    def busy(self) -> str:
        tail = [s[0] for s in self.samples[self._mark :]]
        if not tail:
            return "-"
        return f"busy avg {sum(tail) / len(tail):.0f}% max {max(tail)}%"


def loadavg() -> str:
    try:
        parts = Path("/proc/loadavg").read_text().split()
        return f"{parts[0]} {parts[1]} {parts[2]}"
    except OSError:
        return "?"


def time_cell(decider: LayaDecider, sample: list[str], batch: int, torch_threads: int) -> dict:
    """One interleaved cell: wall ms/ctx and process CPU ms/ctx for a drained batch run."""
    import torch

    torch.set_num_threads(max(1, torch_threads))
    decider.decide_batch(sample[:1])  # pool resize, not timed
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for start in range(0, len(sample), batch):
        decider.decide_batch(sample[start : start + batch])
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    return {"ms": wall * 1000 / len(sample), "cpu_ms": cpu * 1000 / len(sample)}


def run_matrix(
    cells: list[tuple[LayaDecider, str, int]],
    sample: list[str],
    rounds: int,
    sensors: Sensors | None,
) -> dict:
    """Interleave every cell across `rounds`; one load spike cannot rank a whole column."""
    results: dict[str, list[dict]] = {label: [] for _, label, _ in cells}
    for rnd in range(rounds):
        for decider, label, batch in cells:
            torch_threads = (
                decider.config.threads_gpu
                if decider.active_device == "gpu"
                else decider.config.threads
            )
            if sensors is not None:
                sensors.mark()
            m = time_cell(decider, sample, batch, torch_threads)
            results[label].append(m)
            extra = f" cpu {m['cpu_ms']:.0f} ms/ctx"
            if sensors is not None:
                extra += f", {sensors.busy()}"
            print(f"round {rnd + 1} {label}: {m['ms']:.0f} ms/ctx{extra}", flush=True)
    return results


def report(
    results: dict, cells: list[tuple[LayaDecider, str, int]], sensors: Sensors | None
) -> None:
    median = {k: statistics.median(x["ms"] for x in v) for k, v in results.items()}
    cpu_med = {k: statistics.median(x["cpu_ms"] for x in v) for k, v in results.items()}
    print(f"\nload {loadavg()}\n")
    print("| cell | ms/ctx (median) | CPU ms/ctx |")
    print("| --- | --- | --- |")
    for _, label, _ in cells:
        print(f"| {label} | {median[label]:.0f} | {cpu_med[label]:.0f} |")
    if sensors is not None:
        peak = sensors.peak()
        print(
            "\npeak during run: gtt "
            f"{peak.get('mem_info_gtt_used', 0) / 2**30:.2f} GiB (pre-run "
            f"{sensors.baseline.get('mem_info_gtt_used', 0) / 2**30:.2f} GiB), vram "
            f"{peak.get('mem_info_vram_used', 0) / 2**30:.3f} GiB, "
            f"busy max {peak.get('gpu_busy_percent', 0)}%"
        )


def parity(states: list[str]) -> bool:
    """CPU fp32 vs every GPU dtype on the same states: labels, bands and probability deltas."""
    configs = [("cpu", "fp32"), ("gpu", "fp32"), ("gpu", "fp16"), ("gpu", "bf16")]
    deciders = {}
    for device, dtype in configs:
        decider, load_s = load_decider(device, dtype)
        label = f"{device}/{dtype}"
        deciders[label] = decider
        print(f"loaded {label}: {load_s:.1f}s, status {decider.status}", flush=True)

    base = "cpu/fp32"
    answers = {label: [d.decide(s) for s in states] for label, d in deciders.items()}
    ok = True
    for label in deciders:
        if label == base:
            continue
        disagreements = sum(
            1 for a, b in zip(answers[base], answers[label], strict=True) if a.label != b.label
        )
        flips = sum(
            1 for a, b in zip(answers[base], answers[label], strict=True) if a.flagged != b.flagged
        )
        bands = sum(
            1
            for a, b in zip(answers[base], answers[label], strict=True)
            if (a.confidence < 0.6) != (b.confidence < 0.6)
            or (a.confidence < 0.9) != (b.confidence < 0.9)
        )
        max_dp = max_dc = max_da = 0.0
        for a, b in zip(answers[base], answers[label], strict=True):
            for key in set(a.probabilities) | set(b.probabilities):
                max_dp = max(
                    max_dp, abs(a.probabilities.get(key, 0.0) - b.probabilities.get(key, 0.0))
                )
            max_dc = max(max_dc, abs(a.confidence - b.confidence))
            max_da = max(max_da, abs(a.attack - b.attack))
        verdict = "OK" if disagreements == 0 and flips == 0 and bands == 0 else "MISMATCH"
        ok = ok and verdict == "OK"
        print(
            f"{base} vs {label}: labels differ {disagreements}/{len(states)}, "
            f"flagged flips {flips}, confidence-band flips {bands}, "
            f"max d(prob) {max_dp:.4f}, d(conf) {max_dc:.4f}, d(attack) {max_da:.4f} -> {verdict}"
        )
    print("PARITY OK" if ok else "PARITY MISMATCH")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=64, help="context pool size")
    parser.add_argument("--per-cell", type=int, default=24, help="contexts per cell per round")
    parser.add_argument("--rounds", type=int, default=3, help="interleaved rounds (median wins)")
    parser.add_argument("--quick", action="store_true", help="1 round, batch 1/8/32 only")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="model device")
    parser.add_argument(
        "--dtype", default="auto", help="comma list: auto, fp32, fp16, bf16 (gpu cells)"
    )
    parser.add_argument(
        "--threads", default="", help="comma list of CPU torch thread counts (cpu cells)"
    )
    parser.add_argument("--batches", default="", help="comma list of drain batch sizes")
    parser.add_argument(
        "--pass-sweep",
        "--cap-sweep",
        dest="pass_sweep",
        action="store_true",
        help="states per forward pass (drain of 32, median over the pool)",
    )
    parser.add_argument("--parity", action="store_true", help="CPU fp32 vs GPU dtypes, no timing")
    args = parser.parse_args()

    states = build_states(args.states)
    sample = states[: min(args.per_cell, len(states))]
    lengths = sorted(len(s) for s in sample)
    print(
        f"pool {len(states)} contexts, {len(sample)} per cell "
        f"(chars min {lengths[0]} / median {lengths[len(lengths) // 2]} / max {lengths[-1]}); "
        f"load {loadavg()}",
        flush=True,
    )

    if args.parity:
        sys.exit(0 if parity(states) else 1)

    batches = (
        tuple(int(b) for b in args.batches.split(","))
        if args.batches
        else (1, 8, 32)
        if args.quick
        else BATCHES
    )
    rounds = 1 if args.quick else args.rounds
    dtypes = tuple(d.strip() for d in args.dtype.split(",")) if args.dtype else ("auto",)
    threads_list = tuple(int(t) for t in args.threads.split(",")) if args.threads else THREADS

    sensors = Sensors()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cells: list[tuple[LayaDecider, str, int]] = []
        with sensors:
            if args.device == "gpu":
                deciders = {}
                for dtype in dtypes:
                    decider, load_s = load_decider("gpu", dtype, threads=threads_list[0])
                    deciders[dtype] = decider
                    first = time_cell(decider, sample[:2], 2, decider.config.threads_gpu)
                    print(
                        f"load gpu/{dtype}: {load_s:.1f}s, status {decider.status}, "
                        f"first call {first['ms']:.0f} ms/ctx",
                        flush=True,
                    )
                for dtype, decider in deciders.items():
                    for batch in batches:
                        cells.append((decider, f"gpu dtype={dtype} batch={batch}", batch))
            else:
                base, load_s = load_decider(args.device, threads=threads_list[0])
                print(f"load {args.device}: {load_s:.1f}s, status {base.status}", flush=True)
                router = base._router
                for threads in threads_list:
                    decider = fresh_decider(router, threads)
                    first = time_cell(decider, sample[:2], 2, threads)
                    print(
                        f"warmup threads={threads}: first call {first['ms']:.0f} ms/ctx",
                        flush=True,
                    )
                for threads in threads_list:
                    for batch in batches:
                        cells.append(
                            (
                                fresh_decider(router, threads),
                                f"threads={threads} batch={batch}",
                                batch,
                            )
                        )

            if args.pass_sweep:
                sample = states[:64]
                batches = (32,)
                if args.device == "gpu":
                    router = deciders[dtypes[0]]._router
                else:
                    router = base._router
                cells = [
                    (
                        fresh_decider(router, threads_list[0], pass_size=ps, device=args.device),
                        f"pass={ps or 'auto'}",
                        32,
                    )
                    for ps in PASS_SIZES
                ]

            results = run_matrix(cells, sample, rounds, sensors)
            report(results, cells, sensors)


if __name__ == "__main__":
    main()

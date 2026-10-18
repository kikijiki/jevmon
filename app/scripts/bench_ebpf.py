"""Replay a synthetic exec stream through the ebpf source and report the daemon's cost.

The stream is generated here (not recorded): a few hundred synthetic executables with a skewed
frequency, argv lengths and counts shaped like a busy workstation, a small share of launches
with a never-seen argument, and a pool of parent processes. A fake helper in a child process
serves it over a unix socket at a fixed rate, in the helper's wire format, so the daemon side
(socket read, parse, redaction, allow-list check, context build, memoised judge, store) runs
unchanged. The decider is a stand-in that sleeps per context (a GPU wait, no CPU).

    uv run --project app python app/scripts/bench_ebpf.py --rate 500 --seconds 30
"""

import argparse
import json
import multiprocessing
import os
import random
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass

from jevmon.config import load_config
from jevmon.core import SqliteStore
from jevmon.monitor import Monitor

WORDS = [f"opt{i}" for i in range(300)]
SUBCMDS = ["build", "test", "run", "status", "diff", "log", "list", "show", "fetch", "push"]


def make_pool(rng: random.Random, n_exe: int = 480):
    exes = [f"/opt/synth/bin/tool{i:03d}" for i in range(n_exe)]
    weights = [1.0 / (i + 1) ** 1.3 for i in range(n_exe)]
    variants = {
        e: [
            [rng.choice(SUBCMDS), *rng.sample(WORDS, rng.choice([0, 1, 2, 4, 8, 14]))]
            for _ in range(rng.choice([1, 1, 1, 1, 2]))
        ]
        for e in exes
    }
    return exes, weights, variants


def line_for(rng, pool, seq, now, novel):
    exes, weights, variants = pool
    exe = rng.choices(exes, weights)[0]
    args = list(rng.choice(variants[exe]))
    # tokens that the template masks: a path and a number
    args.append(f"/home/user/work/proj{rng.randrange(40)}/file{rng.randrange(9999)}.txt")
    args.append(str(rng.randrange(100000)))
    if rng.random() < novel:
        args.append(f"--never-seen-{rng.randrange(10**9)}")
    if rng.random() < 0.1:
        args.append(f"{rng.getrandbits(160):x}")  # long ids/hashes: masked in templates
    argv = [exe, *args][:64]
    ppid = 1000 + rng.randrange(60)
    obj = {
        "t": "exec",
        "seq": seq,
        "ts": now,
        "mono": int(now * 1e9),
        "start": seq,
        "pid": 100000 + seq % 4_000_000,
        "ppid": ppid,
        "pstart": ppid,
        "uid": 1000,
        "euid": 1000,
        "gid": 1000,
        "cgroup": 1000 + rng.randrange(7),
        "argc": len(argv),
        "flags": 0,
        "comm": os.path.basename(exe)[:15],
        "pcomm": "parent",
        "exe": exe,
        "filename": exe,
        "argv": argv,
        "trunc": 0,
    }
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


def serve(path: str, rate: float, seconds: float, novel: float, seed: int) -> None:
    rng = random.Random(seed)
    pool = make_pool(rng)
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    conn, _ = srv.accept()
    conn.sendall(b'{"t":"hello","v":1,"helper":"fake","version":"0","exe_source":"kernel"}\n')
    start = time.monotonic()
    sent = 0
    last_hb = start
    while time.monotonic() - start < seconds:
        due = int((time.monotonic() - start) * rate) - sent
        batch = b"".join(
            line_for(rng, pool, sent + i, time.time(), novel) for i in range(max(0, min(due, 2000)))
        )
        sent += max(0, min(due, 2000))
        if time.monotonic() - last_hb >= 1.0:
            last_hb = time.monotonic()
            batch += (
                json.dumps(
                    {"t": "hb", "seen": sent, "emitted": sent, "lost": 0, "client_dropped": 0}
                )
                + "\n"
            ).encode()
        if batch:
            conn.sendall(batch)
        time.sleep(0.01)
    conn.close()


class Sleepy:
    """A decider that waits per context like a GPU would: no CPU, ~45 ms each."""

    ready = True
    status = "stub"
    active_device = "gpu"

    def __init__(self, ms: float) -> None:
        self.ms = ms

    def load(self) -> None:
        pass

    def decide(self, state):
        return self.decide_batch([state])[0]

    def decide_batch(self, states):
        from jevmon.laya.decider import Decision

        time.sleep(self.ms * len(states) / 1000)
        return [
            Decision(label="safe", confidence=0.9, attack=0.0, flagged=False, summary="stub")
            for _ in states
        ]


@dataclass
class Sample:
    t: float
    cpu: float


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=500)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--novel", type=float, default=0.01)
    ap.add_argument("--ms", type=float, default=45, help="stub model ms per context")
    ap.add_argument("--db", default="", help="keep counts here; a second run on it is warm")
    ap.add_argument("--allow", default="", help="one allow rule, e.g. '/opt/synth/bin/tool000'")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="jevmon-bench-")
    path = os.path.join(tmp, "exec.sock")
    helper = multiprocessing.Process(
        target=serve, args=(path, args.rate, args.seconds, args.novel, 7), daemon=True
    )
    helper.start()
    while not os.path.exists(path):
        time.sleep(0.01)

    config = load_config("/nonexistent") if False else load_config()
    from dataclasses import replace

    from jevmon.allow import FilterSettings

    exec_s = replace(config.exec, source="ebpf", helper_socket=path)
    cfg = replace(config, exec=exec_s)
    if args.allow:
        cfg = replace(cfg, filter=FilterSettings(allow=(args.allow,)))
    db = args.db or os.path.join(tmp, "e.db")
    store = SqliteStore(db)
    monitor = Monitor(cfg, store, decider=Sleepy(args.ms))
    monitor.start()
    cpu0, wall0 = time.process_time(), time.monotonic()
    ages, depths = [], []
    stop = threading.Event()

    def sampler():
        while not stop.wait(1.0):
            st = monitor.status()
            ages.append(st["queue_age"])
            depths.append(st["queue_depth"])

    threading.Thread(target=sampler, daemon=True).start()
    helper.join(args.seconds + 15)
    time.sleep(1.5)
    cpu, wall = time.process_time() - cpu0, time.monotonic() - wall0
    st = monitor.status()
    stop.set()
    monitor.stop()
    store.close()
    judged, cached = st["judged"], st["cached"]
    out = {
        "rate": args.rate,
        "wall_s": round(wall, 1),
        "daemon_cpu_pct_of_one_core": round(100 * cpu / wall, 1),
        "observed": st["exec_observed"],
        "observed_per_s": round(st["exec_observed"] / wall, 1),
        "allow_skipped": st.get("allow_skipped", 0),
        "judged": judged,
        "cached": cached,
        "memo_hit_rate": round(cached / (judged + cached), 3) if judged + cached else 0,
        "shed_queue_full": st["dropped"],
        "queue_age_max_s": max(ages, default=0),
        "queue_depth_max": max(depths, default=0),
        "exec_lost": st["exec_lost"],
        "events_in_store": store_count(db),
    }
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


def store_count(db: str) -> int:
    s = SqliteStore(db)
    n = s.count_events()
    s.close()
    return n


if __name__ == "__main__":
    raise SystemExit(main())

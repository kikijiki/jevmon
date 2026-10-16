"""Append more variants (existing families, perturbed params) and a few new families.

Deterministic (seeded). Every command stays harmless: sleep/true/echo, cp of system binaries to
new names, bounded loops, python -c loops, curl against the lab's own nginx. Family names are
descriptive; the family hash split decides their side and nothing is steered.
Run once: `uv run python scripts/gen_scenarios2.py` (appends to lab/scenarios.toml).
"""

import json
import random
import tomllib
from collections import defaultdict
from pathlib import Path

PATH = Path("lab/scenarios.toml")
rng = random.Random(20261013)
data = tomllib.loads(PATH.read_text())
scen = data["scenarios"]
by_family: dict[str, list[dict]] = defaultdict(list)
for s in scen:
    by_family[s["family"]].append(s)

NAMES = [
    "agentd",
    "relayd",
    "cachew",
    "tickd",
    "sync2",
    "pushd",
    "warmup",
    "reapd",
    "boxd",
    "feedd",
    "mover",
    "fetchd",
    "linkd",
    "pooler",
    "queuer",
    "spoold",
    "gridd",
    "tracer",
    "rotor",
    "scand",
    "vaultd",
    "panel",
    "fusion",
    "metad",
    "cronx",
    "peerd",
    "unpack",
    "bundled",
    "stagerd",
    "mirrord",
]
DIRS = [
    "/opt/tools",
    "/srv/app",
    "/usr/local/lib",
    "/var/lib/svc",
    "/home/dev/bin",
    "/root/tools",
    "/srv/bin",
    "/opt/vendor/bin",
    "/usr/local/sbin",
    "/var/opt/pkg",
]
CAPS = {"n": 180, "secs": 8, "mb": 380, "count": 150}


def perturb(key: str, value, family_values: list):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        lo = max(1, int(value * 0.6))
        hi = max(lo + 1, int(value * 1.5) + 1)
        return min(CAPS.get(key, 400), rng.randint(lo, hi))
    if isinstance(value, float):
        return round(max(0.1, value * rng.uniform(0.6, 1.5)), 2)
    if isinstance(value, str) and value.startswith("/") and "/" in value[1:]:
        base = value.rsplit("/", 1)[0]
        name = rng.choice(NAMES)
        if rng.random() < 0.5:
            base = rng.choice(DIRS)
        if value.split("/")[-1].startswith("."):
            name = "." + name
        return f"{base}/{name}"
    return rng.choice(family_values)


extra = []
VARIANTS = 4  # new variants per existing family
for family, items in by_family.items():
    if family == "idle_daemon":
        continue
    first = items[0]
    if not first.get("params"):
        continue
    letters = [s["id"].split(".")[-1] for s in items]
    nxt = [chr(ord("a") + i) for i in range(26) if chr(ord("a") + i) not in letters]
    duration = max(s["duration"] for s in items)
    for letter in nxt[:VARIANTS]:
        base = rng.choice(items)
        params = {}
        for k, v in base["params"].items():
            params[k] = perturb(k, v, [s["params"].get(k, v) for s in items])
        extra.append(
            {
                "id": f"{family}.{letter}",
                "family": family,
                "label": first["label"],
                "container": first.get("container", "target"),
                "expected": first.get("expected", 0),
                "duration": duration + 1,
                "params": params,
                "steps": base["steps"],
            }
        )

# new families: (name, label, duration, cmd, [params])
NEW = [
    (
        "tar_pack",
        "safe",
        9,
        "tar czf {out} /etc/hostname /etc/os-release 2>/dev/null; sleep {secs}",
        [
            {"out": "/tmp/pack.tgz", "secs": 2},
            {"out": "/var/tmp/etc.tgz", "secs": 3},
            {"out": "/root/conf.tgz", "secs": 2},
            {"out": "/srv/pack.tgz", "secs": 3},
        ],
    ),
    (
        "sort_count",
        "safe",
        9,
        "printf 'b\\na\\nb\\n' | sort | uniq -c > {out}; sleep {secs}",
        [
            {"out": "/tmp/count.txt", "secs": 2},
            {"out": "/var/tmp/c.txt", "secs": 3},
            {"out": "/root/c.txt", "secs": 2},
            {"out": "/srv/c.txt", "secs": 3},
        ],
    ),
    (
        "proc_listing",
        "safe",
        9,
        "ps aux | head -{n} > {out}; sleep {secs}",
        [
            {"n": 5, "out": "/tmp/ps.txt", "secs": 2},
            {"n": 10, "out": "/var/tmp/ps.txt", "secs": 3},
            {"n": 20, "out": "/root/ps.txt", "secs": 2},
            {"n": 8, "out": "/srv/ps.txt", "secs": 3},
        ],
    ),
    (
        "tree_walk",
        "safe",
        9,
        "find {root} -maxdepth 2 2>/dev/null | wc -l; sleep {secs}",
        [
            {"root": "/usr/share", "secs": 2},
            {"root": "/etc", "secs": 3},
            {"root": "/var/lib", "secs": 2},
            {"root": "/usr/lib", "secs": 3},
        ],
    ),
    (
        "json_check",
        "safe",
        9,
        "python3 -c \"import json; print(json.loads('{payload}'))\"; sleep {secs}",
        [
            {"payload": "[1, 2, 3]", "secs": 2},
            {"payload": '{\\"a\\": 1}', "secs": 3},
            {"payload": "[]", "secs": 2},
            {"payload": "[true, null]", "secs": 3},
        ],
    ),
    (
        "odd_timestamp",
        "warn",
        9,
        "touch -d '2020-01-0{n} 03:00' {path}; sleep {secs}",
        [
            {"n": 1, "path": "/tmp/stamp", "secs": 2},
            {"n": 2, "path": "/var/tmp/old.txt", "secs": 3},
            {"n": 3, "path": "/root/stamp", "secs": 2},
            {"n": 4, "path": "/srv/stamp", "secs": 3},
        ],
    ),
    (
        "deep_path_tool",
        "warn",
        10,
        "mkdir -p {dir} && cp /usr/bin/sleep {dir}/{name} && {dir}/{name} {secs}",
        [
            {"dir": "/opt/a/b/c/d", "name": "probe", "secs": 3},
            {"dir": "/srv/x/y/z", "name": "walker", "secs": 4},
            {"dir": "/usr/local/share/q/r", "name": "mapper", "secs": 3},
            {"dir": "/var/opt/m/n/o", "name": "gaugeit", "secs": 4},
        ],
    ),
    (
        "long_argument",
        "warn",
        9,
        "sleep {secs}; echo {text} > /dev/null",
        [
            {
                "secs": 2,
                "text": "alpha-beta-gamma-delta-epsilon-zeta-eta-theta-iota-kappa-lambda-mu-nu-xi-omicron-pi-rho-sigma-tau",
            },
            {
                "secs": 3,
                "text": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            },
            {
                "secs": 2,
                "text": "the-quick-brown-fox-jumps-over-the-lazy-dog-the-quick-brown-fox-jumps-over-the-lazy-dog",
            },
            {
                "secs": 3,
                "text": "one-two-three-four-five-six-seven-eight-nine-ten-eleven-twelve-thirteen-fourteen-fifteen",
            },
        ],
    ),
    (
        "world_writable_exec",
        "crit",
        10,
        "mkdir -p {dir} && chmod 777 {dir} && cp /usr/bin/sleep {dir}/{name} && {dir}/{name} {secs}",
        [
            {"dir": "/tmp/.w", "name": ".run", "secs": 3},
            {"dir": "/var/tmp/.cache2", "name": ".svc", "secs": 4},
            {"dir": "/tmp/.data", "name": ".job", "secs": 3},
            {"dir": "/var/tmp/.s", "name": ".proc", "secs": 4},
        ],
    ),
    (
        "hosts_edit",
        "crit",
        9,
        "echo '127.0.0.1 {host}' >> /etc/hosts; sleep {secs}",
        [
            {"host": "update.example.test", "secs": 2},
            {"host": "api.example.test", "secs": 3},
            {"host": "mirror.example.test", "secs": 2},
            {"host": "cdn.example.test", "secs": 3},
        ],
    ),
    (
        "chmod_then_run",
        "crit",
        10,
        "cp /usr/bin/sleep {path} && chmod 4755 {path} && {path} {secs}",
        [
            {"path": "/tmp/.pp", "secs": 3},
            {"path": "/var/tmp/.qq", "secs": 4},
            {"path": "/tmp/.rr", "secs": 3},
            {"path": "/var/tmp/.ss", "secs": 4},
        ],
    ),
    (
        "seq_sort_spin",
        "runaway",
        12,
        "seq 1 {n} | sort -R | tail -1 > /dev/null; sleep 1",
        [{"n": 2000000}, {"n": 3000000}, {"n": 1500000}, {"n": 2500000}],
    ),
    (
        "dd_pipe_burn",
        "runaway",
        12,
        "timeout {secs} sh -c 'dd if=/dev/zero bs=1M 2>/dev/null | cat > /dev/null'; true",
        [{"secs": 6}, {"secs": 8}, {"secs": 5}, {"secs": 7}],
    ),
    (
        "wide_spawn_loop",
        "runaway",
        12,
        "i=0; while [ $i -lt {n} ]; do sh -c 'sleep {gap}' & i=$((i+1)); done; wait",
        [{"n": 60, "gap": 3}, {"n": 90, "gap": 3}, {"n": 120, "gap": 4}, {"n": 70, "gap": 5}],
    ),
    (
        "tight_python_spin",
        "runaway",
        12,
        'python3 -c "import time; t = time.time(); x = 0\\nwhile time.time() - t < {secs}: x += 1"',
        [{"secs": 6}, {"secs": 8}, {"secs": 5}, {"secs": 7}],
    ),
]
for name, label, dur, cmd, variants in NEW:
    for letter, params in zip("abcd", variants, strict=True):
        extra.append(
            {
                "id": f"{name}.{letter}",
                "family": name,
                "label": label,
                "container": "target",
                "expected": 3,
                "duration": dur,
                "params": params,
                "steps": [{"cmd": cmd, "timeout": 40}],
            }
        )

rng.shuffle(extra)
end = max(s["at"] + s["duration"] for s in scen if s["id"] != "background.idle") + 3
at = end + 4
out = ["\n# Stage-5 variants and families (generated by scripts/gen_scenarios2.py)"]
for s in extra:
    s["at"] = at
    at += s["duration"] + 3 + 1
    ps = ", ".join(f"{k} = {json.dumps(v)}" for k, v in s["params"].items())
    step = s["steps"][0]
    label_line = (
        'excluded = "runaway: a rate or resource spike cannot be seen in a single launch"'
        if s["label"] == "runaway"
        else f'label = "{s["label"]}"'
    )
    out.append(
        f"""
[[scenarios]]
id = "{s["id"]}"
family = "{s["family"]}"
{label_line}
container = "{s["container"]}"
expected = {s["expected"]}
at = {s["at"]}
duration = {s["duration"]}
params = {{ {ps} }}
steps = [{{ cmd = {json.dumps(step["cmd"])}, timeout = {int(step.get("timeout", 40))} }}]"""
    )
text = PATH.read_text()
PATH.write_text(text + "\n".join(out) + "\n")
total = at + 10
print(f"{len(extra)} scenarios, end at {at}s; set run_seconds >= {total}")

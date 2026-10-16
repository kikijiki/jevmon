"""Write `lab/scenarios_facts.toml`: scenarios whose labelled process differs in its launch-time
facts (E17). Hand-designed by the author; the label is the designer's call from the command shape
and every command is harmless (cp of system binaries to new names, sleep, id, short python).

    uv run --project lab python lab/scripts/gen_scenarios4.py

Why a `payload`: the lab labels only the designated process, which was always the `sh -c` driver
whose own facts (exe, stdio, workdir, privileges) are identical in every scenario. A scenario here
either varies the driver itself through `podman exec` options (`exec_args`: user, workdir) or names
its subject by argv prefix (`payload`), so the labelled row carries the facts that distinguish it.

Two sides, disjoint shapes: `heldout` designs (names picked by the family-hash rule of E14 so they
never enter training) and `train` designs (names picked to land in the train split), so a head
fitted on the train side can learn what the facts mean and the held-out side tests it on new
shapes. The author wrote both sides; none of this is independent ground truth.

Caveat built into the lab: every process runs in the lab container, so `namespaces` and most of
`cgroup_path` are the same for every row; only `unshare` shapes could vary them.
"""

import json
from pathlib import Path

from jevmon_lab.manifest import split_of

OUT = Path("lab/scenarios_facts.toml")
SUFFIXES = [
    "",
    "_job",
    "_run",
    "_task",
    "_step",
    "_cmd",
    "_op",
    "_pass",
    "_probe",
    "_flow",
    "_unit",
    "_x",
    "_alt",
    "_v2",
    "_main",
    "_aux",
]

MEMFD = (
    "python3 -c \"import os; fd=os.memfd_create('{n}'); "
    "os.write(fd, open('/bin/sleep','rb').read()); "
    "os.execv('/proc/self/fd/%d' % fd, ['{n}', '{s}'])\""
)

# side, base, label, duration, exec_args, payload, cmd, 4 variants
D = list[tuple[str, str, str, int, list[str], str, str, list[dict]]]
DESIGNS: D = [
    # ---------------- held-out side
    (
        "heldout",
        "nobody_task",
        "safe",
        9,
        ["-u", "nobody"],
        "",
        "id -u > /tmp/{n}.out; echo done >> /tmp/{n}.out; sleep {s}",
        [{"n": "a1", "s": 2}, {"n": "b2", "s": 3}, {"n": "c3", "s": 2}, {"n": "d4", "s": 3}],
    ),
    (
        "heldout",
        "alias_link",
        "safe",
        10,
        [],
        "/tmp/{n} {s}",
        "ln -sf /bin/sleep /tmp/{n}; /tmp/{n} {s}",
        [
            {"n": "waiter", "s": 3},
            {"n": "napper", "s": 4},
            {"n": "pause_cmd", "s": 3},
            {"n": "idler", "s": 4},
        ],
    ),
    (
        "heldout",
        "hot_swap",
        "safe",
        11,
        [],
        "/usr/local/bin/{n} {s}",
        (
            "cp /bin/sleep /usr/local/bin/{n}; (/usr/local/bin/{n} {s} &) ; sleep 1; "
            "cp /bin/true /usr/local/bin/{n}.new; mv -f /usr/local/bin/{n}.new /usr/local/bin/{n}; "
            "sleep {s}"
        ),
        [
            {"n": "lab-agent", "s": 4},
            {"n": "lab-relay", "s": 5},
            {"n": "lab-poller", "s": 4},
            {"n": "lab-sync", "s": 5},
        ],
    ),
    (
        "heldout",
        "world_tool",
        "warn",
        9,
        [],
        "/tmp/{n}",
        "cp /bin/sleep /tmp/{n}; chmod 777 /tmp/{n}; /tmp/{n} {s}",
        [{"n": "wt1", "s": 2}, {"n": "wt2", "s": 3}, {"n": "wt3", "s": 2}, {"n": "wt4", "s": 3}],
    ),
    (
        "heldout",
        "shm_workdir",
        "warn",
        9,
        ["-w", "/dev/shm"],
        "",
        "ls -la > /tmp/{n}.ls; sleep {s}",
        [{"n": "s1", "s": 2}, {"n": "s2", "s": 3}, {"n": "s3", "s": 2}, {"n": "s4", "s": 3}],
    ),
    (
        "heldout",
        "memfd_run",
        "crit",
        10,
        [],
        "{n} {s}",
        MEMFD,
        [
            {"n": "svc", "s": 4},
            {"n": "worker", "s": 5},
            {"n": "agent", "s": 4},
            {"n": "daemon", "s": 5},
        ],
    ),
    (
        "heldout",
        "kthread_mask",
        "crit",
        10,
        [],
        "[kworker/{k}:{j}] {s}",
        "bash -c 'exec -a \"[kworker/{k}:{j}]\" /bin/sleep {s}'",
        [
            {"k": 0, "j": 1, "s": 4},
            {"k": 1, "j": 2, "s": 5},
            {"k": 2, "j": 0, "s": 4},
            {"k": 3, "j": 1, "s": 5},
        ],
    ),
    (
        "heldout",
        "suid_shell",
        "crit",
        10,
        [],
        "/tmp/{n} -p",
        "cp /bin/sh /tmp/{n}; chmod u+s /tmp/{n}; /tmp/{n} -p -c 'id; sleep {s}'",
        [{"n": "rs1", "s": 3}, {"n": "rs2", "s": 4}, {"n": "rs3", "s": 3}, {"n": "rs4", "s": 4}],
    ),
    # ---------------- train side (different shapes, same facts)
    (
        "train",
        "daemon_user_job",
        "safe",
        9,
        ["-u", "daemon"],
        "",
        "whoami > /tmp/{n}.out; sleep {s}",
        [{"n": "e1", "s": 2}, {"n": "e2", "s": 3}, {"n": "e3", "s": 2}, {"n": "e4", "s": 3}],
    ),
    (
        "train",
        "tool_symlink",
        "safe",
        10,
        [],
        "/usr/local/bin/{n} {s}",
        "ln -sf /bin/sleep /usr/local/bin/{n}; /usr/local/bin/{n} {s}",
        [
            {"n": "hold", "s": 3},
            {"n": "linger", "s": 4},
            {"n": "stall", "s": 3},
            {"n": "rest", "s": 4},
        ],
    ),
    (
        "train",
        "package_upgrade",
        "safe",
        11,
        [],
        "/opt/{n}/bin/run {s}",
        (
            "mkdir -p /opt/{n}/bin; cp /bin/sleep /opt/{n}/bin/run; (/opt/{n}/bin/run {s} &) ; sleep 1; "
            "rm -f /opt/{n}/bin/run; cp /bin/true /opt/{n}/bin/run; sleep {s}"
        ),
        [
            {"n": "pkgone", "s": 4},
            {"n": "pkgtwo", "s": 5},
            {"n": "pkgthree", "s": 4},
            {"n": "pkgfour", "s": 5},
        ],
    ),
    (
        "train",
        "tmp_copy_exec",
        "warn",
        9,
        [],
        "/tmp/{n}",
        "cp /bin/sleep /tmp/{n}; /tmp/{n} {s}",
        [{"n": "lx", "s": 2}, {"n": "ly", "s": 3}, {"n": "lz", "s": 2}, {"n": "lw", "s": 3}],
    ),
    (
        "train",
        "odd_workdir",
        "warn",
        9,
        ["-w", "/var/tmp"],
        "",
        "ls -la > /tmp/{n}.ls; sleep {s}",
        [{"n": "w1", "s": 2}, {"n": "w2", "s": 3}, {"n": "w3", "s": 2}, {"n": "w4", "s": 3}],
    ),
    (
        "train",
        "deleted_runner",
        "crit",
        10,
        [],
        "/tmp/{n} {s}",
        "cp /bin/sleep /tmp/{n}; (/tmp/{n} {s} &) ; rm -f /tmp/{n}; sleep {s}",
        [
            {"n": "gone1", "s": 5},
            {"n": "gone2", "s": 6},
            {"n": "gone3", "s": 5},
            {"n": "gone4", "s": 6},
        ],
    ),
    (
        "train",
        "fake_system_name",
        "crit",
        10,
        [],
        "{n} {s}",
        "bash -c 'exec -a {n} /bin/sleep {s}'",
        [
            {"n": "systemd-logind", "s": 4},
            {"n": "sshd", "s": 5},
            {"n": "dbus-daemon", "s": 4},
            {"n": "cron", "s": 5},
        ],
    ),
    (
        "train",
        "memfd_loader",
        "crit",
        10,
        [],
        "{n} {s}",
        MEMFD,
        [
            {"n": "kernel", "s": 4},
            {"n": "updated", "s": 5},
            {"n": "helper", "s": 4},
            {"n": "netd", "s": 5},
        ],
    ),
    (
        "heldout",
        "nobody_shm_job",
        "warn",
        9,
        ["-u", "nobody", "-w", "/dev/shm"],
        "",
        "sleep {s}; echo {n} > /tmp/{n}.out",
        [{"n": "n1", "s": 3}, {"n": "n2", "s": 4}, {"n": "n3", "s": 3}, {"n": "n4", "s": 4}],
    ),
    (
        "train",
        "nobody_tmp_job",
        "warn",
        9,
        ["-u", "nobody", "-w", "/tmp"],
        "",
        "sleep {s}; echo {n} > /tmp/{n}.out",
        [{"n": "m1", "s": 3}, {"n": "m2", "s": 4}, {"n": "m3", "s": 3}, {"n": "m4", "s": 4}],
    ),
    # ---------------- environment signal (E16), held-out side
    (
        "heldout",
        "toolchain_libpath",
        "safe",
        9,
        ["-e", "LD_LIBRARY_PATH=/opt/{n}/lib"],
        "",
        "ls /opt > /tmp/{n}.out 2>&1; sleep {s}",
        [
            {"n": "gcc12", "s": 2},
            {"n": "llvm17", "s": 3},
            {"n": "cuda12", "s": 2},
            {"n": "rocm6", "s": 3},
        ],
    ),
    (
        "heldout",
        "proxy_fetch",
        "safe",
        9,
        ["-e", "HTTP_PROXY=http://127.0.0.1:{p}", "-e", "NO_PROXY=localhost,127.0.0.1"],
        "",
        "curl -s -m 2 -o /dev/null http://127.0.0.1/ ; sleep {s}",
        [{"p": 3128, "s": 2}, {"p": 8118, "s": 3}, {"p": 3129, "s": 2}, {"p": 8888, "s": 3}],
    ),
    (
        "heldout",
        "preload_tmp",
        "crit",
        9,
        ["-e", "LD_PRELOAD=/tmp/{n}.so"],
        "",
        "id > /tmp/{n}.out 2>&1; sleep {s}",
        [
            {"n": "libx", "s": 2},
            {"n": "libhook", "s": 3},
            {"n": "libaudit", "s": 2},
            {"n": "libnss", "s": 3},
        ],
    ),
    (
        "heldout",
        "path_prefix_tmp",
        "crit",
        9,
        ["-e", "PATH=/tmp/{n}:/usr/bin:/bin"],
        "",
        "ls / > /tmp/{n}.out 2>&1; sleep {s}",
        [
            {"n": "bin1", "s": 2},
            {"n": "bin2", "s": 3},
            {"n": "shims", "s": 2},
            {"n": "wrap", "s": 3},
        ],
    ),
    # ---------------- environment signal (E16), train side
    (
        "train",
        "pythonpath_build",
        "safe",
        9,
        ["-e", "PYTHONPATH=/srv/{n}/lib"],
        "",
        "python3 -c 'import sys; print(len(sys.path))' > /tmp/{n}.out; sleep {s}",
        [
            {"n": "proj1", "s": 2},
            {"n": "proj2", "s": 3},
            {"n": "proj3", "s": 2},
            {"n": "proj4", "s": 3},
        ],
    ),
    (
        "train",
        "ci_variables",
        "safe",
        9,
        ["-e", "CI=true", "-e", "BUILD_ID={n}"],
        "",
        "echo build {n} > /tmp/{n}.out; sleep {s}",
        [
            {"n": "1042", "s": 2},
            {"n": "1043", "s": 3},
            {"n": "1044", "s": 2},
            {"n": "1045", "s": 3},
        ],
    ),
    (
        "train",
        "startup_hook_env",
        "crit",
        9,
        ["-e", "PYTHONSTARTUP=/tmp/{n}.py"],
        "",
        "python3 -c 'print(1)' > /tmp/{n}.out 2>&1; sleep {s}",
        [{"n": "s1", "s": 2}, {"n": "s2", "s": 3}, {"n": "s3", "s": 2}, {"n": "s4", "s": 3}],
    ),
    (
        "train",
        "preload_shm",
        "crit",
        9,
        ["-e", "LD_PRELOAD=/dev/shm/{n}.so"],
        "",
        "id > /tmp/{n}.out 2>&1; sleep {s}",
        [{"n": "p1", "s": 2}, {"n": "p2", "s": 3}, {"n": "p3", "s": 2}, {"n": "p4", "s": 3}],
    ),
]


def pick_name(base: str, side: str) -> str | None:
    for suffix in SUFFIXES:
        name = base + suffix
        if split_of(name) == side:
            return name
    return None


def main() -> None:
    lines = [
        "# Facts scenarios for E17 (generated by scripts/gen_scenarios4.py). Labels are the",
        "# designer's. Held-out-side family names are chosen to hash to held-out, train-side to train.",
        "run_seconds = 3000",
        "grace_seconds = 3.0",
        "",
        "[limits]",
        'memory = "512m"',
        'cpus = "1.0"',
        "pids = 256",
        'scratch = "64m"',
    ]
    at = 5
    kept, dropped = [], []
    for side, base, label, dur, exec_args, payload, cmd, variants in DESIGNS:
        name = pick_name(base, "heldout" if side == "heldout" else "train")
        if name is None:
            dropped.append(base)
            continue
        kept.append((name, label, side))
        for letter, params in zip("abcd", variants, strict=True):
            ps = ", ".join(f"{k} = {json.dumps(v)}" for k, v in params.items())
            lines += [
                "",
                "[[scenarios]]",
                f'id = "{name}.{letter}"',
                f'family = "{name}"',
                f'label = "{label}"',
                'container = "target"',
                "expected = 1",
                f"at = {at}",
                f"duration = {dur}",
                f"params = {{ {ps} }}",
                f"steps = [{{ cmd = {json.dumps(cmd)}, timeout = 40 }}]",
            ]
            if exec_args:
                lines.append(f"exec_args = {json.dumps(exec_args)}")
            if payload:
                lines.append(f"payload = {json.dumps(payload)}")
            at += dur + 4
    OUT.write_text("\n".join(lines) + "\n")
    for side in ("heldout", "train"):
        names = [(n, lb) for n, lb, s in kept if s == side]
        print(side, len(names), names)
    print("scenarios", len(kept) * 4, "ends at", at, "dropped", dropped)


if __name__ == "__main__":
    main()

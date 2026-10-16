"""Write `lab/scenarios_settle.toml`: new held-out-only families for the E14 settle experiment.

    uv run --project lab python lab/scripts/gen_scenarios3.py

The families are hand-designed (the label is the designer's call, by construction, from the command
shape; no model is involved) and every command is harmless (echo/sleep/true, cp of system binaries
to new names, short python -c, curl against the lab's own nginx). They are new shapes, not
perturbed copies of the old families.

Split: the rule is unchanged (`manifest.split_of`, family-name hash, 30% held-out). New families
must never enter training, so each design carries a few synonym names and the first one the hash
assigns to `heldout` is used; a design with no synonym on the held-out side is dropped. The hash is
independent of content, so this chooses names, not examples. The output is a standalone manifest
so the pinned 474-scenario manifest and its tests are untouched.
"""

import base64
import json
from pathlib import Path

from jevmon_lab.manifest import split_of

OUT = Path("lab/scenarios_settle.toml")
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
]


def b64(code: str) -> str:
    return base64.b64encode(code.encode()).decode()


# (base name, label, duration, command template, [params x 4])
DESIGNS: list[tuple[str, str, int, str, list[dict]]] = [
    # ---- safe: everyday shapes
    (
        "disk_usage",
        "safe",
        9,
        "df -h / > {out}; du -sh {dir} 2>/dev/null; sleep {secs}",
        [
            {"out": "/tmp/df.txt", "dir": "/etc", "secs": 2},
            {"out": "/var/tmp/df.txt", "dir": "/usr/share", "secs": 3},
            {"out": "/root/df.txt", "dir": "/var/lib", "secs": 2},
            {"out": "/srv/df.txt", "dir": "/usr/lib", "secs": 3},
        ],
    ),
    (
        "env_dump",
        "safe",
        9,
        "env | sort | head -{n} > {out}; sleep {secs}",
        [
            {"n": 5, "out": "/tmp/env.txt", "secs": 2},
            {"n": 12, "out": "/var/tmp/env.txt", "secs": 3},
            {"n": 8, "out": "/root/env.txt", "secs": 2},
            {"n": 20, "out": "/srv/env.txt", "secs": 3},
        ],
    ),
    (
        "line_count",
        "safe",
        9,
        "wc -l /etc/passwd /etc/hosts > {out}; sleep {secs}",
        [
            {"out": "/tmp/wc.txt", "secs": 2},
            {"out": "/var/tmp/wc.txt", "secs": 3},
            {"out": "/root/wc.txt", "secs": 2},
            {"out": "/srv/wc.txt", "secs": 3},
        ],
    ),
    (
        "grep_release",
        "safe",
        9,
        "grep -c {pat} /etc/os-release /etc/hostname 2>/dev/null; sleep {secs}",
        [
            {"pat": "NAME", "secs": 2},
            {"pat": "VERSION", "secs": 3},
            {"pat": "ID", "secs": 2},
            {"pat": "HOME", "secs": 3},
        ],
    ),
    (
        "symlink_current",
        "safe",
        9,
        "ln -sf {target} {link} && readlink {link}; sleep {secs}",
        [
            {"target": "/etc/hostname", "link": "/tmp/cur", "secs": 2},
            {"target": "/etc/os-release", "link": "/var/tmp/cur", "secs": 3},
            {"target": "/etc/hosts", "link": "/root/cur", "secs": 2},
            {"target": "/etc/passwd", "link": "/srv/cur", "secs": 3},
        ],
    ),
    (
        "checksum_verify",
        "safe",
        9,
        "sha256sum {src} > {out} && sha256sum -c {out}; sleep {secs}",
        [
            {"src": "/etc/os-release", "out": "/tmp/sum", "secs": 2},
            {"src": "/etc/hostname", "out": "/var/tmp/sum", "secs": 3},
            {"src": "/etc/hosts", "out": "/root/sum", "secs": 2},
            {"src": "/etc/passwd", "out": "/srv/sum", "secs": 3},
        ],
    ),
    (
        "dir_listing",
        "safe",
        9,
        "ls -la {dir} | head -{n} > {out}; sleep {secs}",
        [
            {"dir": "/etc", "n": 10, "out": "/tmp/ls.txt", "secs": 2},
            {"dir": "/usr/bin", "n": 15, "out": "/var/tmp/ls.txt", "secs": 3},
            {"dir": "/var", "n": 8, "out": "/root/ls.txt", "secs": 2},
            {"dir": "/usr/lib", "n": 12, "out": "/srv/ls.txt", "secs": 3},
        ],
    ),
    (
        "date_stamp",
        "safe",
        9,
        "date +%s > {out}; hostname >> {out}; sleep {secs}",
        [
            {"out": "/tmp/stamp.txt", "secs": 2},
            {"out": "/var/tmp/stamp.txt", "secs": 3},
            {"out": "/root/stamp.txt", "secs": 2},
            {"out": "/srv/stamp.txt", "secs": 3},
        ],
    ),
    (
        "python_version",
        "safe",
        9,
        'python3 --version; python3 -c "import sys; print({what})"; sleep {secs}',
        [
            {"what": "sys.platform", "secs": 2},
            {"what": "sys.version_info[:2]", "secs": 3},
            {"what": "sys.maxsize", "secs": 2},
            {"what": "len(sys.path)", "secs": 3},
        ],
    ),
    (
        "scratch_workdir",
        "safe",
        9,
        "d=$(mktemp -d) && echo {word} > $d/f && cat $d/f && rm -rf $d; sleep {secs}",
        [
            {"word": "alpha", "secs": 2},
            {"word": "beta", "secs": 3},
            {"word": "gamma", "secs": 2},
            {"word": "delta", "secs": 3},
        ],
    ),
    (
        "http_status",
        "safe",
        9,
        "curl -sf -o /dev/null -w '%{http_code}' http://127.0.0.1/{path}; sleep {secs}",
        [
            {"path": "", "secs": 2},
            {"path": "payload.sh", "secs": 3},
            {"path": "index.html", "secs": 2},
            {"path": "", "secs": 3},
        ],
    ),
    (
        "awk_sum",
        "safe",
        9,
        "awk 'BEGIN{ s = 0; for (i = 0; i < {n}; i++) s += i; print s }' > {out}; sleep {secs}",
        [
            {"n": 100, "out": "/tmp/sum.txt", "secs": 2},
            {"n": 500, "out": "/var/tmp/sum.txt", "secs": 3},
            {"n": 50, "out": "/root/sum.txt", "secs": 2},
            {"n": 900, "out": "/srv/sum.txt", "secs": 3},
        ],
    ),
    (
        "xargs_batch",
        "safe",
        9,
        "printf '%s\\n' {items} | xargs -n1 echo > {out}; sleep {secs}",
        [
            {"items": "a b c d", "out": "/tmp/xa.txt", "secs": 2},
            {"items": "one two three", "out": "/var/tmp/xa.txt", "secs": 3},
            {"items": "x y", "out": "/root/xa.txt", "secs": 2},
            {"items": "p q r s t", "out": "/srv/xa.txt", "secs": 3},
        ],
    ),
    (
        "config_backup",
        "safe",
        9,
        "cp /etc/hostname {out} && cp /etc/os-release {out}.2; sleep {secs}",
        [
            {"out": "/tmp/cfg.bak", "secs": 2},
            {"out": "/var/tmp/cfg.bak", "secs": 3},
            {"out": "/root/cfg.bak", "secs": 2},
            {"out": "/srv/cfg.bak", "secs": 3},
        ],
    ),
    (
        "uptime_report",
        "safe",
        9,
        "uptime > {out}; free -m >> {out}; sleep {secs}",
        [
            {"out": "/tmp/up.txt", "secs": 2},
            {"out": "/var/tmp/up.txt", "secs": 3},
            {"out": "/root/up.txt", "secs": 2},
            {"out": "/srv/up.txt", "secs": 3},
        ],
    ),
    # ---- warn: unusual but not hostile
    (
        "awk_system",
        "warn",
        9,
        "awk 'BEGIN{ system(\"sleep {secs}\") }'",
        [{"secs": 3}, {"secs": 4}, {"secs": 3}, {"secs": 4}],
    ),
    (
        "renamed_shell",
        "warn",
        10,
        "mkdir -p {dir} && cp /bin/sh {dir}/{name} && {dir}/{name} -c 'sleep {secs}'",
        [
            {"dir": "/opt/ven", "name": "shx", "secs": 3},
            {"dir": "/srv/tools", "name": "runner", "secs": 4},
            {"dir": "/usr/local/lib/app", "name": "wrapper", "secs": 3},
            {"dir": "/var/opt/svc", "name": "exec", "secs": 4},
        ],
    ),
    (
        "hidden_helper",
        "warn",
        10,
        "mkdir -p {dir} && cp /usr/bin/sleep {dir}/.{name} && {dir}/.{name} {secs}",
        [
            {"dir": "/home/dev/.local/bin", "name": "helper", "secs": 3},
            {"dir": "/root/.cache/app", "name": "agent", "secs": 4},
            {"dir": "/home/dev/.config/tool", "name": "runner", "secs": 3},
            {"dir": "/root/.local/share/x", "name": "probe", "secs": 4},
        ],
    ),
    (
        "socket_touch",
        "warn",
        9,
        "python3 -c \"import socket; s = socket.create_connection(('127.0.0.1', {port})); s.close()\" 2>/dev/null; sleep {secs}",
        [
            {"port": 80, "secs": 2},
            {"port": 8080, "secs": 3},
            {"port": 80, "secs": 2},
            {"port": 9000, "secs": 3},
        ],
    ),
    (
        "perm_survey",
        "warn",
        9,
        "stat -c '%U:%a %n' {a} {b} 2>/dev/null; sleep {secs}",
        [
            {"a": "/etc/shadow", "b": "/etc/passwd", "secs": 2},
            {"a": "/etc/sudoers", "b": "/etc/hosts", "secs": 3},
            {"a": "/root", "b": "/etc/ssh", "secs": 2},
            {"a": "/etc/group", "b": "/etc/gshadow", "secs": 3},
        ],
    ),
    (
        "user_listing",
        "warn",
        9,
        "getent passwd | cut -d: -f1 > {out}; id; sleep {secs}",
        [
            {"out": "/tmp/users.txt", "secs": 2},
            {"out": "/var/tmp/users.txt", "secs": 3},
            {"out": "/root/users.txt", "secs": 2},
            {"out": "/srv/users.txt", "secs": 3},
        ],
    ),
    (
        "opt_binary",
        "warn",
        10,
        "mkdir -p /usr/local/{name} && cp /usr/bin/sleep /usr/local/{name}/run && /usr/local/{name}/run {secs}",
        [
            {"name": "acmed", "secs": 3},
            {"name": "toolkit", "secs": 4},
            {"name": "svcmgr", "secs": 3},
            {"name": "pulse", "secs": 4},
        ],
    ),
    (
        "proc_survey",
        "warn",
        9,
        "ls /proc | head -{n} > {out}; cat /proc/1/cmdline | tr '\\0' ' '; sleep {secs}",
        [
            {"n": 10, "out": "/tmp/p.txt", "secs": 2},
            {"n": 20, "out": "/var/tmp/p.txt", "secs": 3},
            {"n": 15, "out": "/root/p.txt", "secs": 2},
            {"n": 30, "out": "/srv/p.txt", "secs": 3},
        ],
    ),
    (
        "var_tmp_script",
        "warn",
        10,
        "printf 'sleep {secs}\\n' > {path} && sh {path}",
        [
            {"path": "/var/tmp/job.sh", "secs": 3},
            {"path": "/var/tmp/run.sh", "secs": 4},
            {"path": "/var/tmp/step.sh", "secs": 3},
            {"path": "/var/tmp/do.sh", "secs": 4},
        ],
    ),
    (
        "niced_job",
        "warn",
        9,
        "nice -n {nice} sleep {secs}",
        [
            {"nice": 19, "secs": 3},
            {"nice": 10, "secs": 4},
            {"nice": 15, "secs": 3},
            {"nice": 5, "secs": 4},
        ],
    ),
    (
        "dated_archive",
        "warn",
        9,
        "tar -cf {out} --mtime='2019-01-01' /etc/hostname 2>/dev/null; sleep {secs}",
        [
            {"out": "/tmp/old.tar", "secs": 2},
            {"out": "/var/tmp/old.tar", "secs": 3},
            {"out": "/root/old.tar", "secs": 2},
            {"out": "/srv/old.tar", "secs": 3},
        ],
    ),
    # ---- crit: suspicious shapes
    (
        "shm_exec",
        "crit",
        10,
        "cp /usr/bin/sleep /dev/shm/{name} && chmod +x /dev/shm/{name} && /dev/shm/{name} {secs}",
        [
            {"name": "kworker", "secs": 3},
            {"name": ".sd", "secs": 4},
            {"name": "udevd", "secs": 3},
            {"name": ".x", "secs": 4},
        ],
    ),
    (
        "python_fetch_run",
        "crit",
        11,
        "python3 -c \"import urllib.request as u; open('{path}', 'wb').write(u.urlopen('http://127.0.0.1/payload.sh').read())\" && sh {path}",
        [
            {"path": "/tmp/.f"},
            {"path": "/var/tmp/.g"},
            {"path": "/dev/shm/.h"},
            {"path": "/tmp/.k"},
        ],
    ),
    (
        "sudoers_append",
        "crit",
        9,
        "echo '{user} ALL=(ALL) NOPASSWD:ALL' >> /tmp/sudoers.d.{name}; sleep {secs}",
        [
            {"user": "dev", "name": "x", "secs": 2},
            {"user": "svc", "name": "y", "secs": 3},
            {"user": "deploy", "name": "z", "secs": 2},
            {"user": "backup", "name": "w", "secs": 3},
        ],
    ),
    (
        "passwd_clone",
        "crit",
        9,
        "cp /etc/passwd {path} && echo '{user}:x:0:0::/root:/bin/sh' >> {path}; sleep {secs}",
        [
            {"path": "/tmp/.pw", "user": "toor", "secs": 2},
            {"path": "/var/tmp/.pw", "user": "r00t", "secs": 3},
            {"path": "/dev/shm/.pw", "user": "adm1n", "secs": 2},
            {"path": "/tmp/.p2", "user": "sys", "secs": 3},
        ],
    ),
    (
        "unit_drop",
        "crit",
        10,
        "cp /usr/bin/sleep {bin} && printf '[Service]\\nExecStart={bin} 1\\n' > /tmp/{name}.service && {bin} {secs}",
        [
            {"bin": "/usr/local/sbin/.updated", "name": "update", "secs": 3},
            {"bin": "/opt/.syslogd", "name": "syslogd", "secs": 4},
            {"bin": "/var/lib/.netmgr", "name": "netmgr", "secs": 3},
            {"bin": "/srv/.cupsd", "name": "cupsd", "secs": 4},
        ],
    ),
    (
        "profile_hook",
        "crit",
        10,
        "cp /usr/bin/sleep {bin} && printf '{bin} 1 &\\n' >> /tmp/.profile.{name} && {bin} {secs}",
        [
            {"bin": "/tmp/.pf", "name": "a", "secs": 3},
            {"bin": "/var/tmp/.pg", "name": "b", "secs": 4},
            {"bin": "/dev/shm/.ph", "name": "c", "secs": 3},
            {"bin": "/tmp/.pk", "name": "d", "secs": 4},
        ],
    ),
    (
        "environ_read",
        "crit",
        9,
        "cat /proc/1/environ > {out} 2>/dev/null; sleep {secs}",
        [
            {"out": "/tmp/.env", "secs": 2},
            {"out": "/var/tmp/.env", "secs": 3},
            {"out": "/dev/shm/.env", "secs": 2},
            {"out": "/tmp/.e2", "secs": 3},
        ],
    ),
    (
        "log_truncate",
        "crit",
        9,
        "mkdir -p /var/log/{name} && : > /var/log/{name}/{file}; rm -f /var/log/{name}/{file}; sleep {secs}",
        [
            {"name": "audit", "file": "audit.log", "secs": 2},
            {"name": "auth", "file": "auth.log", "secs": 3},
            {"name": "sys", "file": "syslog", "secs": 2},
            {"name": "login", "file": "wtmp", "secs": 3},
        ],
    ),
    (
        "service_stop",
        "crit",
        9,
        "systemctl stop {svc} 2>/dev/null; systemctl mask {svc} 2>/dev/null; sleep {secs}",
        [
            {"svc": "auditd", "secs": 2},
            {"svc": "rsyslog", "secs": 3},
            {"svc": "apparmor", "secs": 2},
            {"svc": "ufw", "secs": 3},
        ],
    ),
    (
        "encoded_python_pipe",
        "crit",
        9,
        "echo {b64} | base64 -d | python3; sleep 1",
        [
            {"b64": b64("import time; time.sleep(2)")},
            {"b64": b64("import time; time.sleep(3)")},
            {"b64": b64("print('ok'); import time; time.sleep(2)")},
            {"b64": b64("x = 1; import time; time.sleep(3)")},
        ],
    ),
    (
        "hidden_tmp_python",
        "crit",
        10,
        "printf 'import time\\ntime.sleep({secs})\\n' > /tmp/.{name}.py && python3 /tmp/.{name}.py && rm -f /tmp/.{name}.py",
        [
            {"name": "svc", "secs": 3},
            {"name": "upd", "secs": 4},
            {"name": "cfg", "secs": 3},
            {"name": "net", "secs": 4},
        ],
    ),
    (
        "lib_path_hijack",
        "crit",
        10,
        "LD_LIBRARY_PATH={dir} sleep {secs} 2>/dev/null",
        [
            {"dir": "/tmp/.lib", "secs": 3},
            {"dir": "/var/tmp/.l", "secs": 4},
            {"dir": "/dev/shm/.so", "secs": 3},
            {"dir": "/tmp/.x86", "secs": 4},
        ],
    ),
    (
        "copied_python_hidden",
        "crit",
        10,
        'cp /usr/bin/python3 {path} && {path} -c "import time; time.sleep({secs})"',
        [
            {"path": "/tmp/.py", "secs": 3},
            {"path": "/var/tmp/.p3", "secs": 4},
            {"path": "/dev/shm/.pyx", "secs": 3},
            {"path": "/tmp/.run", "secs": 4},
        ],
    ),
    (
        "netcat_like_probe",
        "crit",
        10,
        "python3 -c \"import socket; [socket.socket().connect_ex(('127.0.0.1', p)) for p in range({lo}, {hi})]\"; sleep {secs}",
        [
            {"lo": 20, "hi": 40, "secs": 2},
            {"lo": 75, "hi": 95, "secs": 3},
            {"lo": 400, "hi": 420, "secs": 2},
            {"lo": 3300, "hi": 3320, "secs": 3},
        ],
    ),
]


def pick_name(base: str) -> str | None:
    for suffix in SUFFIXES:
        name = base + suffix
        if split_of(name) == "heldout":
            return name
    return None


def main() -> None:
    lines = [
        "# Held-out-only families for the E14 settle experiment (generated by scripts/gen_scenarios3.py).",
        "# Labels are the designer's, by construction. Family names are chosen so the unchanged family",
        "# hash split puts every one of them on the held-out side.",
        "run_seconds = 2200",
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
    for base, label, dur, cmd, variants in DESIGNS:
        name = pick_name(base)
        if name is None:
            dropped.append(base)
            continue
        kept.append((name, label))
        for letter, params in zip("abcd", variants, strict=True):
            ps = ", ".join(f"{k} = {json.dumps(v)}" for k, v in params.items())
            lines += [
                "",
                "[[scenarios]]",
                f'id = "{name}.{letter}"',
                f'family = "{name}"',
                f'label = "{label}"',
                'container = "target"',
                "expected = 3",
                f"at = {at}",
                f"duration = {dur}",
                f"params = {{ {ps} }}",
                f"steps = [{{ cmd = {json.dumps(cmd)}, timeout = 40 }}]",
            ]
            at += dur + 4
    OUT.write_text("\n".join(lines) + "\n")
    by = {k: sum(1 for _, lb in kept if lb == k) for k in ("safe", "warn", "crit")}
    print(f"{len(kept)} families {by}, {len(kept) * 4} scenarios, ends at {at}s; dropped {dropped}")


if __name__ == "__main__":
    main()

"""Append the stage-4 scenario families to lab/scenarios.toml (generator, run once).

Family names are descriptive; the family hash split decides their side, nothing is steered.
"""

import json
import random
import sys

F = []  # (family, label, secs window, cmd template, [params...], expected)


def fam(name, label, dur, cmd, variants, expected=3):
    F.append((name, label, dur, cmd, variants, expected))


# routine
fam(
    "log_rotate",
    "safe",
    9,
    "mkdir -p {dir} && printf 'line\\n' > {dir}/{name}.log && gzip -f {dir}/{name}.log; sleep {secs}",
    [
        {"dir": "/var/log", "name": "app", "secs": 2},
        {"dir": "/srv/logs", "name": "web", "secs": 3},
        {"dir": "/tmp/logs", "name": "worker", "secs": 2},
    ],
)
fam(
    "source_checkout",
    "safe",
    9,
    "mkdir -p {dir}/src && cd {dir}/src && printf 'int main(){{}}\\n' > main.c && cp main.c main.bak; sleep {secs}",
    [
        {"dir": "/root/proj", "secs": 2},
        {"dir": "/home/dev/work", "secs": 3},
        {"dir": "/srv/build", "secs": 2},
    ],
)
fam(
    "python_tool",
    "safe",
    8,
    'python3 -c "{code}"; sleep {secs}',
    [
        {"code": "print(sum(range(1000)))", "secs": 2},
        {"code": "import platform; print(platform.machine())", "secs": 3},
        {"code": "import os; print(os.getcwd())", "secs": 2},
    ],
)
fam(
    "health_probe",
    "safe",
    9,
    "i=0; while [ $i -lt {n} ]; do curl -sf -o /dev/null http://127.0.0.1/{path}; sleep 1; i=$((i+1)); done",
    [{"n": 3, "path": "payload.sh"}, {"n": 4, "path": ""}, {"n": 2, "path": "payload.sh"}],
)
fam(
    "test_runner",
    "safe",
    8,
    "i=0; while [ $i -lt {n} ]; do python3 -c 'import time; time.sleep({secs})' & i=$((i+1)); done; wait",
    [{"n": 3, "secs": 2}, {"n": 4, "secs": 1.5}, {"n": 2, "secs": 3}],
)
fam(
    "svc_status",
    "safe",
    8,
    "systemctl is-active nginx; systemctl status nginx --no-pager | head -3; sleep {secs}",
    [{"secs": 2}, {"secs": 3}, {"secs": 2}],
)
fam(
    "file_sync",
    "safe",
    9,
    "mkdir -p {dst} && cp -r /etc/default {dst}/ 2>/dev/null; sleep {secs}",
    [
        {"dst": "/srv/mirror", "secs": 2},
        {"dst": "/root/backup-sync", "secs": 3},
        {"dst": "/var/cache/sync", "secs": 2},
    ],
)
fam(
    "report_job",
    "safe",
    9,
    "date > {out}; uname -a >> {out}; df -h >> {out}; sleep {secs}",
    [
        {"out": "/tmp/report.txt", "secs": 2},
        {"out": "/var/tmp/report-daily.txt", "secs": 3},
        {"out": "/root/report.txt", "secs": 2},
    ],
)
# unusual
fam(
    "admin_scan",
    "warn",
    9,
    "find /etc -maxdepth 2 -name '{pat}' 2>/dev/null | head -5; sleep {secs}",
    [{"pat": "*.conf", "secs": 2}, {"pat": "*.d", "secs": 3}, {"pat": "host*", "secs": 2}],
)
fam(
    "disk_audit",
    "warn",
    9,
    "du -sh {a} {b} 2>/dev/null; sleep {secs}",
    [
        {"a": "/usr", "b": "/var", "secs": 2},
        {"a": "/etc", "b": "/opt", "secs": 3},
        {"a": "/root", "b": "/srv", "secs": 2},
    ],
)
fam(
    "local_server",
    "warn",
    10,
    "python3 -m http.server {port} --bind 127.0.0.1 >/dev/null 2>&1 & sleep {secs}; kill $!",
    [{"port": 8099, "secs": 3}, {"port": 8123, "secs": 4}, {"port": 9100, "secs": 3}],
)
fam(
    "custom_daemon",
    "warn",
    9,
    "mkdir -p $(dirname {bin}) && cp /usr/bin/sleep {bin} && {bin} {secs}",
    [
        {"bin": "/usr/local/sbin/syncd", "secs": 4},
        {"bin": "/opt/tools/rotd", "secs": 5},
        {"bin": "/srv/bin/queued", "secs": 4},
    ],
)
fam(
    "env_wrapper",
    "warn",
    8,
    "{wrap} sleep {secs}",
    [
        {"wrap": "env LANG=C TZ=UTC", "secs": 3},
        {"wrap": "nice -n 5", "secs": 4},
        {"wrap": "timeout 20", "secs": 3},
    ],
)
fam(
    "compress_big",
    "warn",
    9,
    "gzip -c /usr/bin/python3* > {out}; sleep {secs}",
    [
        {"out": "/tmp/py.gz", "secs": 2},
        {"out": "/var/tmp/pybin.gz", "secs": 3},
        {"out": "/root/py.gz", "secs": 2},
    ],
)
fam(
    "tool_upgrade",
    "warn",
    9,
    "mkdir -p $(dirname {bin}) && cp /usr/bin/sleep {bin}.new && mv {bin}.new {bin} && {bin} {secs}",
    [
        {"bin": "/usr/local/bin/lumen", "secs": 4},
        {"bin": "/opt/tools/forgeit", "secs": 4},
        {"bin": "/usr/local/bin/pulse", "secs": 5},
    ],
)
# suspicious
fam(
    "cron_persist",
    "crit",
    10,
    "cp /usr/bin/sleep {bin} && printf '* * * * * root {bin}\\n' > /etc/cron.d/{name} && {bin} {secs}",
    [
        {"bin": "/usr/lib/.sysd", "name": "sys-update", "secs": 3},
        {"bin": "/var/lib/.mon", "name": "e2scrub", "secs": 4},
        {"bin": "/tmp/.sd", "name": "anacron-x", "secs": 3},
    ],
)
fam(
    "ssh_key_drop",
    "crit",
    9,
    "mkdir -p /root/.ssh && printf 'ssh-ed25519 AAAA{key} root@x\\n' >> /root/.ssh/authorized_keys; sleep {secs}",
    [{"key": "C3Nza", "secs": 3}, {"key": "Q9wzz", "secs": 2}, {"key": "H7aab", "secs": 3}],
)
fam(
    "base64_pipe",
    "crit",
    9,
    "echo '{b64}' | base64 -d | sh; sleep 1",
    [{"b64": "c2xlZXAgMw=="}, {"b64": "c2xlZXAgNA=="}, {"b64": "c2xlZXAgMg=="}],
)
fam(
    "bind_shell",
    "crit",
    9,
    "python3 -c \"import socket, time; s = socket.socket(); s.bind(('127.0.0.1', {port})); s.listen(1); time.sleep({secs})\"",
    [{"port": 4444, "secs": 4}, {"port": 31337, "secs": 4}, {"port": 5555, "secs": 3}],
)
fam(
    "setuid_copy",
    "crit",
    9,
    "cp /bin/sh {path} && chmod u+s {path} && {path} -c 'sleep {secs}'",
    [
        {"path": "/tmp/.rootsh", "secs": 3},
        {"path": "/var/tmp/.bash2", "secs": 4},
        {"path": "/tmp/.x11", "secs": 3},
    ],
)
fam(
    "masquerade",
    "crit",
    9,
    "cp /usr/bin/sleep {path} && {path} {secs}",
    [
        {"path": "/tmp/sshd", "secs": 4},
        {"path": "/var/tmp/systemd-journald", "secs": 4},
        {"path": "/tmp/kworker", "secs": 3},
    ],
)
fam(
    "cred_read",
    "crit",
    8,
    "cat /etc/passwd /etc/shadow > {out} 2>/dev/null; sleep {secs}",
    [
        {"out": "/tmp/.p", "secs": 3},
        {"out": "/var/tmp/.pw", "secs": 3},
        {"out": "/dev/null", "secs": 2},
    ],
)
fam(
    "remote_script",
    "crit",
    9,
    "curl -s http://127.0.0.1/payload.sh | sh -s; sleep {secs}",
    [{"secs": 1}, {"secs": 2}, {"secs": 1}],
)
fam(
    "history_wipe",
    "crit",
    8,
    "rm -f /root/.bash_history; ln -sf /dev/null /root/.bash_history; sleep {secs}",
    [{"secs": 3}, {"secs": 4}, {"secs": 3}],
)
# runaway
fam(
    "curl_flood",
    "runaway",
    12,
    "i=0; while [ $i -lt {n} ]; do curl -s -o /dev/null http://127.0.0.1/; sleep 0.05; i=$((i+1)); done",
    [{"n": 60}, {"n": 90}, {"n": 120}],
)
fam(
    "gzip_loop",
    "runaway",
    12,
    "timeout {secs} sh -c 'yes | gzip -c > /dev/null'; true",
    [{"secs": 6}, {"secs": 8}, {"secs": 10}],
)
fam(
    "retry_storm",
    "runaway",
    12,
    "python3 -c \"import subprocess; [subprocess.run(['sleep', '{gap}']) for _ in range({n})]\"",
    [{"n": 20, "gap": 0.3}, {"n": 30, "gap": 0.2}, {"n": 15, "gap": 0.5}],
)
fam(
    "big_alloc",
    "runaway",
    11,
    'python3 -c "import time; b = bytearray({mb} * 1024 * 1024); time.sleep({secs})"',
    [{"mb": 200, "secs": 5}, {"mb": 300, "secs": 5}, {"mb": 380, "secs": 5}],
)
fam(
    "pipe_spawn",
    "runaway",
    12,
    "python3 -c \"import subprocess; ps = [subprocess.Popen(['sleep', '{secs}']) for _ in range({n})]; [p.wait() for p in ps]\"",
    [{"n": 60, "secs": 4}, {"n": 100, "secs": 4}, {"n": 150, "secs": 5}],
)
fam(
    "write_loop",
    "runaway",
    12,
    "dd if=/dev/zero of=/scratch/{name} bs=1M count={mb} 2>/dev/null; sleep {secs}; rm -f /scratch/{name}",
    [
        {"name": "blob1", "mb": 40, "secs": 4},
        {"name": "cache.tmp", "mb": 50, "secs": 4},
        {"name": "dump.bin", "mb": 30, "secs": 5},
    ],
)

items = []
for name, label, dur, cmd, variants, exp in F:
    for letter, params in zip("abc", variants, strict=True):
        items.append((name, label, dur, cmd, letter, params, exp))
random.Random(7).shuffle(items)
at = 1332
out = [
    "\n# =====================================================================",
    "# Stage-4 families (generated by scripts/gen_scenarios.py; docs/improve.md)",
    "# =====================================================================",
]
EXCLUDED = 'excluded = "runaway: a rate or resource spike cannot be seen in a single launch"'
for name, label, dur, cmd, letter, params, exp in items:
    label_line = EXCLUDED if label == "runaway" else f'label = "{label}"'
    ps = ", ".join(f"{k} = {json.dumps(v)}" for k, v in params.items())
    out.append(f'''
[[scenarios]]
id = "{name}.{letter}"
family = "{name}"
{label_line}
expected = {exp}
at = {at}
duration = {dur}
params = {{ {ps} }}
steps = [{{ cmd = {json.dumps(cmd)}, timeout = 40 }}]''')
    at += dur + 3 + 1
print(f"end at {at}", file=sys.stderr)
with open("lab/scenarios.toml", "a") as fh:
    fh.write("\n".join(out) + "\n")

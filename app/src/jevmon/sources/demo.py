"""A fictional exec/summary recording: invented processes, users, paths, hosts and units.

Nothing here is read from the machine. Every pid, uid, path, unit, hostname and timestamp comes
from a seeded generator over the table below, on a fixed clock (`DEMO_EPOCH`), so replaying the
recording twice produces byte-identical events. That is what makes `just screenshots` repeatable
and what keeps real local details (a real username, home directory, hostname or process) out of
anything the demo captures.

The stream is a small narrative a reviewer can read: builds, test runners and shells under a
`demo` login, a scheduler and a session manager starting the usual services, one tool from /opt,
a few network clients, and several launches that look out of place. Nothing here is a
judgement - the judge is whatever `[judge] mode` selects.
"""

import random
from dataclasses import dataclass

from jevmon.events import Event, Severity
from jevmon.sources.exec import KIND_EXEC, KIND_SUMMARY

# 2024-10-04 09:00:00 UTC. Timestamps are absolute, so `TZ` cannot move them.
DEMO_EPOCH = 1_728_032_400.0
DEMO_SEED = 20_241_004
DEMO_COUNT = 64


@dataclass(frozen=True, slots=True)
class _Launch:
    exe: str
    comm: str
    parent_exe: str
    parent_comm: str
    parents: str
    ppid: int
    uid: int
    cwd: str
    cgroup: str
    argv: tuple[str, ...]
    # launch-time facts that differ from the derived defaults (see `_facts_for`)
    facts: tuple[tuple[str, str], ...] = ()


_TOOLCHAIN = "/srv/demo/toolchain/bin"
_APP = "/srv/demo/work/demo/app"
_HOME = "/srv/demo/work/demo"
_SESSION = "user-1001.slice/demo-session.service"


def _launch(
    exe: str,
    *args: str,
    parent: str,
    parents: str,
    ppid: int,
    cwd: str = _HOME,
    cgroup: str = _SESSION,
    uid: int = 1001,
    facts: tuple[tuple[str, str], ...] = (),
    argv0: str | None = None,
) -> _Launch:
    """One invented launch: `exe args...` started by `parent` (a basename under /srv/demo)."""
    base = exe.rsplit("/", 1)[-1]
    parent_exe = parent if parent.startswith("/") else f"/srv/demo/bin/{parent}"
    return _Launch(
        exe=exe,
        comm=base[:15],
        parent_exe=parent_exe,
        parent_comm=parent_exe.rsplit("/", 1)[-1][:15],
        parents=parents,
        ppid=ppid,
        uid=uid,
        cwd=cwd,
        cgroup=cgroup,
        argv=(argv0 or exe, *args),
        facts=facts,
    )


_BASH = f"{_TOOLCHAIN}/bash"

# Invented process table: the host is `demo-gw-01`, the user is `demo` (uids 1000/1001), every
# path lives under /srv/demo, /opt/demo or /usr/libexec, and none of it exists on this machine.
_LAUNCHES: tuple[_Launch, ...] = (
    # builds and test runners, started from a shell
    _launch(
        f"{_TOOLCHAIN}/make",
        "-j8",
        "all",
        parent=_BASH,
        parents="bash <- sshd-session <- demo-init",
        ppid=2301,
        cwd=_APP,
    ),
    _launch(
        f"{_TOOLCHAIN}/cc",
        "-O2",
        "-c",
        "src/parser.c",
        "-o",
        "build/parser.o",
        parent=f"{_TOOLCHAIN}/make",
        parents="make <- bash <- sshd-session",
        ppid=2315,
        cwd=_APP,
    ),
    _launch(
        f"{_TOOLCHAIN}/pytest",
        "-q",
        "tests/unit",
        "--maxfail=3",
        parent=_BASH,
        parents="bash <- sshd-session <- demo-init",
        ppid=2301,
        cwd=_APP,
    ),
    _launch(
        f"{_TOOLCHAIN}/cargo",
        "build",
        "--release",
        "--locked",
        parent=_BASH,
        parents="bash <- sshd-session <- demo-init",
        ppid=2301,
        cwd=_APP,
    ),
    _launch(
        f"{_TOOLCHAIN}/python3",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        parent=_BASH,
        parents="bash <- sshd-session <- demo-init",
        ppid=2301,
        cwd=_APP,
    ),
    _launch(
        f"{_TOOLCHAIN}/git",
        "status",
        "--porcelain",
        parent=_BASH,
        parents="bash <- sshd-session <- demo-init",
        ppid=2301,
        cwd=_APP,
    ),
    _launch(
        f"{_TOOLCHAIN}/bash",
        "-lc",
        "./scripts/ci.sh --quick",
        parent="sshd-session",
        parents="sshd-session <- demo-init",
        ppid=905,
        cwd=_APP,
    ),
    # routine housekeeping
    _launch(
        f"{_TOOLCHAIN}/tar",
        "-czf",
        "/srv/demo/archive/demo-nightly.tgz",
        "/srv/demo/work/demo",
        parent="scheduler",
        parents="scheduler <- demo-init",
        ppid=310,
        cgroup="system.slice/demo-backup.service",
    ),
    _launch(
        f"{_TOOLCHAIN}/rsync",
        "-a",
        "/srv/demo/archive/",
        "demo@demo-gw-01:/srv/demo/backup/",
        parent="scheduler",
        parents="scheduler <- demo-init",
        ppid=310,
        cgroup="system.slice/demo-backup.service",
    ),
    _launch(
        f"{_TOOLCHAIN}/journalctl",
        "-u",
        "demo-session.service",
        "--since",
        "-5m",
        parent=_BASH,
        parents="bash <- sshd-session <- demo-init",
        ppid=2301,
    ),
    _launch(
        f"{_TOOLCHAIN}/systemctl",
        "--user",
        "restart",
        "demo-session.service",
        parent=_BASH,
        parents="bash <- sshd-session <- demo-init",
        ppid=2301,
    ),
    # network clients
    _launch(
        f"{_TOOLCHAIN}/curl",
        "-fsS",
        "https://demo-gw-01/healthz",
        "--max-time",
        "3",
        parent="scheduler",
        parents="scheduler <- demo-init",
        ppid=310,
        cgroup="system.slice/demo-telemetry.service",
    ),
    _launch(
        f"{_TOOLCHAIN}/ssh",
        "-o",
        "BatchMode=yes",
        "demo@demo-gw-01",
        "uptime",
        parent=_BASH,
        parents="bash <- sshd-session <- demo-init",
        ppid=2301,
    ),
    _launch(
        f"{_TOOLCHAIN}/wget",
        "-q",
        "http://demo-gw-01:8080/tool.sh",
        "-O",
        "/srv/demo/tmp/tool.sh",
        parent="/srv/demo/tmp/launcher",
        parents="launcher <- sh",
        ppid=905,
        cwd="/srv/demo/tmp",
    ),
    # services and agents
    _launch(
        "/srv/demo/bin/backup-agent",
        "--target",
        "demo-gw-01:/srv/demo/archive",
        "--since",
        "24h",
        "--run-as",
        "demo",
        parent="sessiond",
        parents="sessiond <- demo-init",
        ppid=240,
    ),
    _launch(
        "/srv/demo/bin/reportd",
        "--report",
        "daily",
        "--out",
        "/srv/demo/reports",
        parent="scheduler",
        parents="scheduler <- demo-init",
        ppid=310,
        cgroup="system.slice/demo-report.service",
    ),
    _launch(
        "/srv/demo/bin/metrics-beacon",
        "--interval",
        "15",
        "--endpoint",
        "https://demo-gw-01/metrics",
        parent="/usr/libexec/demo-init",
        parents="demo-init",
        ppid=1,
        cwd="/srv/demo",
        cgroup="system.slice/demo-telemetry.service",
        uid=1000,
    ),
    _launch(
        "/srv/demo/bin/log-shipper",
        "--follow",
        "/srv/demo/logs/app.log",
        parent="sessiond",
        parents="sessiond <- demo-init",
        ppid=240,
    ),
    _launch(
        "/srv/demo/bin/queue-consumer",
        "--queue",
        "reports",
        "--prefetch",
        "8",
        parent="scheduler",
        parents="scheduler <- demo-init",
        ppid=310,
        cgroup="system.slice/demo-worker.service",
    ),
    _launch(
        "/srv/demo/bin/webhook-relay",
        "--listen",
        "localhost:8443",
        "--peer",
        "demo-gw-01",
        parent="sessiond",
        parents="sessiond <- demo-init",
        ppid=240,
    ),
    _launch(
        "/srv/demo/venv/bin/demo-import",
        "--source",
        "/srv/demo/incoming",
        "--as",
        "demo",
        parent="/srv/demo/venv/bin/demo-runner",
        parents="demo-runner <- sessiond <- demo-init",
        ppid=240,
    ),
    _launch(
        "/opt/demo/tools/scan-runner",
        "--root",
        "/srv/demo",
        "--depth",
        "3",
        parent="sessiond",
        parents="sessiond <- demo-init",
        ppid=240,
        cwd="/opt/demo",
        cgroup="system.slice/demo-scan.service",
        uid=1000,
    ),
    # launches that look out of place
    _launch(
        "/srv/demo/tmp/updater",
        "--fetch",
        "http://demo-gw-01:8080/pkg.tar",
        "--out",
        "/srv/demo/tmp/pkg.part",
        parent="/srv/demo/tmp/launcher",
        parents="launcher <- sh",
        ppid=905,
        cwd="/srv/demo/tmp",
    ),
    _launch(
        "/srv/demo/bin/job-runner",
        "--workers",
        "512",
        "--interval",
        "10ms",
        parent="scheduler",
        parents="scheduler <- demo-init",
        ppid=310,
        cgroup="system.slice/demo-worker.service",
    ),
    _launch(
        f"{_TOOLCHAIN}/bash",
        "-c",
        "while true; do /srv/demo/bin/poll-queue --once; done",
        parent="/srv/demo/tmp/launcher",
        parents="launcher <- sh",
        ppid=905,
        cwd="/srv/demo/tmp",
    ),
    _launch(
        f"{_TOOLCHAIN}/find",
        "/srv/demo",
        "-name",
        "*.tmp",
        "-delete",
        parent="/srv/demo/tmp/launcher",
        parents="launcher <- sh",
        ppid=905,
        cwd="/srv/demo/tmp",
    ),
    # launches that exercise the per-process facts (all invented)
    _launch(
        "/opt/demo/libexec/demo-elevate",
        "-u",
        "demo-ops",
        "demo-reload",
        parent=_BASH,
        parents="bash <- sshd-session <- demo-init",
        ppid=2301,
        facts=(
            ("privileges", "uid=1001 euid=0 capeff=000001ffffffffff nnp=0 seccomp=0"),
            ("exe_file", "owner=0 mode=4755 setuid=yes world_writable=no"),
        ),
    ),
    _launch(
        f"{_TOOLCHAIN}/node",
        "build.js",
        parent=_BASH,
        parents="bash <- sshd-session <- demo-init",
        ppid=2301,
        cwd=_APP,
        facts=(("stdio", "0=tty 1=pipe 2=pipe"),),
    ),
    _launch(
        "/srv/demo/tmp/demo-agent (deleted)",
        "--once",
        parent="scheduler",
        parents="scheduler <- demo-init",
        ppid=310,
        cwd="/srv/demo/tmp",
        cgroup="system.slice/demo-worker.service",
        facts=(
            ("exe_state", "deleted"),
            ("exe_file", "owner=1001 mode=0755 setuid=no world_writable=no"),
        ),
    ),
    _launch(
        f"{_TOOLCHAIN}/cat",
        "/srv/demo/work/demo/notes.txt",
        parent="/srv/demo/tmp/launcher",
        parents="launcher <- sh",
        ppid=905,
        cwd="/srv/demo/tmp",
        argv0="demo-reader",
    ),
    _launch(
        "/memfd:/srv/demo/payload (deleted)",
        parent="/srv/demo/tmp/launcher",
        parents="launcher <- sh",
        ppid=905,
        cwd="/srv/demo/tmp",
        facts=(("exe_state", "memfd"), ("stdio", "0=null 1=socket 2=null")),
    ),
    _launch(
        "/opt/demo/bin/demo-web",
        "--port",
        "8080",
        parent="/opt/demo/libexec/demo-container-runtime",
        parents="demo-container-runtime <- demo-init",
        ppid=412,
        cwd="/app",
        cgroup="user.slice/user-1001.slice/user@1001.service/app.slice/libpod-<id>.scope",
        uid=0,
        facts=(
            (
                "namespaces",
                (
                    "differs(mnt,pid,net,user,uts,ipc,cgroup) [mnt=differs pid=differs net=differs "
                    "user=differs uts=differs ipc=differs cgroup=differs time=same] "
                    "uid_map=remapped nspid=3187>1"
                ),
            ),
            ("privileges", "uid=0 euid=0 capeff=00000000a80425fb nnp=0 seccomp=2"),
            ("stdio", "0=null 1=pipe 2=pipe"),
        ),
    ),
    _launch(
        f"{_TOOLCHAIN}/make",
        "-C",
        "/tmp/demo-build",
        parent=_BASH,
        parents="bash <- sshd-session <- demo-init",
        ppid=2301,
        cwd="/tmp/demo-build",
    ),
)


def _facts_for(launch: _Launch) -> dict[str, str]:
    """Believable launch-time facts for an invented launch: defaults derived from where it runs
    (an interactive session, a service, a container), overridden by the launch's own entries."""
    base = launch.argv[0].rsplit("/", 1)[-1]
    interactive = launch.cgroup == _SESSION
    exe_base = launch.exe.rsplit("/", 1)[-1].removesuffix(" (deleted)")
    facts = {
        "exe_state": "normal",
        "argv0_vs_exe": "match" if base.removesuffix(" (deleted)") == exe_base else "differs",
        "stdio": "0=tty 1=tty 2=tty" if interactive else "0=null 1=socket 2=socket",
        "session": (
            "tty=yes sid_leader=no loginuid=set"
            if interactive
            else "tty=no sid_leader=yes loginuid=unset"
        ),
        "workdir": launch.cwd,
        "cgroup_path": "/".join(launch.cgroup.split("/")[-3:]),
        "privileges": (
            f"uid={launch.uid} euid={launch.uid} capeff=0000000000000000 nnp=0 seccomp=0"
        ),
        "namespaces": (
            "host [mnt=same pid=same net=same user=same uts=same ipc=same cgroup=same "
            "time=same] uid_map=identity"
        ),
        "exe_file": "owner=0 mode=0755 setuid=no world_writable=no",
    }
    facts.update(dict(launch.facts))
    return facts


def demo_events(
    *, count: int = DEMO_COUNT, epoch: float = DEMO_EPOCH, seed: int = DEMO_SEED
) -> list[Event]:
    """The recording: `count` exec events with a scan summary every eighth one, all fictional."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    rng = random.Random(seed)
    order = list(range(len(_LAUNCHES)))
    rng.shuffle(order)

    def summary(now: float, observed: int, missed: int, tracked: int) -> Event:
        # `observed`/`missed` are this window's, exactly like the /proc source's own summary.
        total = observed + missed
        return Event(
            time=round(now, 3),
            source="exec",
            subject="exec",
            kind=KIND_SUMMARY,
            severity=Severity.SAFE,
            message="exec scan summary",
            fields={
                "observed": observed,
                "estimated_missed": missed,
                "miss_ratio": round(missed / total, 4) if total else 0.0,
                "effective_interval": 0.05,
                "scan_p50": 0.00042,
                "scan_p95": 0.00117,
                "tracked": tracked,
                "forks_per_s": round(rng.uniform(0.4, 6.0), 2),
            },
        )

    events: list[Event] = [summary(epoch, 0, 0, 0)]  # the poller's own baseline scan
    pid = 2410
    clock = epoch
    observed = 0
    for index in range(count):
        clock += rng.uniform(1.0, 2.2)
        pid += rng.randint(1, 9)
        launch = _LAUNCHES[order[index % len(order)]]
        events.append(
            Event(
                time=round(clock, 3),
                source="exec",
                subject=str(pid),
                kind=KIND_EXEC,
                severity=Severity.SAFE,
                message=f"exec {launch.comm}",
                fields={
                    "pid": pid,
                    "ppid": launch.ppid,
                    "uid": launch.uid,
                    "comm": launch.comm,
                    "starttime": int(clock - epoch) * 100 + index,
                    "state": "R" if index % 5 == 4 else "S",
                    "exe": launch.exe,
                    "argv0": launch.argv[0],
                    "path0": launch.argv[0],
                    "argv": " ".join(launch.argv),
                    "cwd": launch.cwd,
                    "cgroup": launch.cgroup,
                    "parent_comm": launch.parent_comm,
                    "parent_exe": launch.parent_exe,
                    "parents": launch.parents,
                    **{f"fact_{k}": v for k, v in _facts_for(launch).items()},
                },
            )
        )
        observed += 1
        if observed == 8 and index < count - 1:
            window_missed = rng.randint(0, 2)
            now = clock + rng.uniform(0.1, 0.4)
            events.append(summary(now, observed, window_missed, min(index + 1, 320)))
            observed = 0
    return events


__all__ = ["DEMO_COUNT", "DEMO_EPOCH", "DEMO_SEED", "demo_events"]

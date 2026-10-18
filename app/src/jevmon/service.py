"""`jevmon start|stop|status`: bring the daemon (and, for the ebpf source, its root helper) up.

The CLI, the daemon and the TUI stay unprivileged. Only one command is ever elevated: starting
(or stopping) the helper as a transient system unit, `systemd-run --unit=jevmon-exec`, through
`run0`, else `sudo`, else `pkexec`. The elevation tool prompts on the terminal; nothing here
reads or stores a password. Which steps run is decided from the effective config:

- `[exec] source = "ebpf"`: if no helper answers on its socket, elevate the helper start, then
  start the daemon's user service.
- `"proc"` / `"replay"`: no elevation, just the user service.

Plans are plain data (`Step` with an argv list, never a shell string) so they can be checked
without running anything. The helper binary is resolved to an absolute path and must be one that
only root can have written (the nix store, or root-owned files in root-owned directories); a path
the user can write is refused, because it would run as root.
"""

import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from jevmon.config import Config
from jevmon.sources.ebpf import PROTOCOL

HELPER_UNIT = "jevmon-exec"
DAEMON_UNIT = "jevmond.service"
ELEVATORS = ("run0", "sudo", "pkexec")
CAPS = "CAP_BPF CAP_PERFMON CAP_SYS_RESOURCE"
NIX_STORE = "/nix/store/"

# Hardening shared by the transient unit and app/contrib/jevmon-exec.service.
UNIT_PROPERTIES = (
    f"AmbientCapabilities={CAPS}",
    f"CapabilityBoundingSet={CAPS}",
    "NoNewPrivileges=yes",
    "PrivateNetwork=yes",
    "RestrictAddressFamilies=AF_UNIX",
    "ProtectSystem=strict",
    "ProtectHome=yes",
    "PrivateTmp=yes",
    "ProtectKernelModules=yes",
    "ProtectControlGroups=yes",
    "RestrictNamespaces=yes",
    "RestrictRealtime=yes",
    "LockPersonality=yes",
    "MemoryDenyWriteExecute=yes",
    "SystemCallArchitectures=native",
    "DevicePolicy=closed",
    "LimitMEMLOCK=infinity",
    "RuntimeDirectory=jevmon",
    "RuntimeDirectoryMode=0755",
    "Restart=on-failure",
)


class ServiceError(RuntimeError):
    """A start/stop step cannot be planned (no elevation tool, unsafe helper path, ...)."""


@dataclass(frozen=True, slots=True)
class Step:
    argv: tuple[str, ...]
    elevated: bool = False
    note: str = ""


@dataclass(slots=True)
class Plan:
    steps: list[Step] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def choose_elevator(which: Callable[[str], str | None] = shutil.which) -> tuple[str, str]:
    """The first of run0, sudo, pkexec that exists, as (name, absolute path)."""
    for name in ELEVATORS:
        found = which(name)
        if found:
            return name, found
    raise ServiceError(f"need one of {', '.join(ELEVATORS)} to start the root helper; none found")


def _root_only_writable(path: Path) -> str | None:
    """None if only root can have written `path`; otherwise why not."""
    parts = [path, *path.parents]
    for item in parts:
        try:
            st = item.lstat()
        except OSError as exc:
            return f"cannot stat {item}: {exc.strerror}"
        if st.st_uid != 0:
            return f"{item} is not owned by root"
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH) and not st.st_mode & stat.S_ISVTX:
            return f"{item} is writable by group or others"
    return None


def validate_helper_path(candidate: str) -> str:
    """Resolve `candidate` to an absolute real path that is safe to run as root, or raise."""
    real = Path(os.path.realpath(candidate))
    if not real.is_file() or not os.access(real, os.X_OK):
        raise ServiceError(f"helper {real} is not an executable file")
    if str(real).startswith(NIX_STORE):
        return str(real)  # the store is root-owned and read-only
    problem = _root_only_writable(real)
    if problem:
        raise ServiceError(
            f"refusing to run {real} as root: {problem}. Build the helper with "
            "`just exec-build` (nix store) or install it as root (`just exec-install`)."
        )
    return str(real)


def find_helper(config: Config, repo_root: Path | None = None) -> str:
    """The helper binary for `config`: `[exec] helper_path`, else `jevmon-exec` on PATH, else the
    repo's `app/helper/result` build. Validated for running as root."""
    configured = config.exec.helper_path
    if configured:
        return validate_helper_path(configured)
    found = shutil.which("jevmon-exec")
    if found:
        return validate_helper_path(found)
    if repo_root is not None:
        built = repo_root / "app" / "helper" / "result" / "bin" / "jevmon-exec"
        if built.exists():
            return validate_helper_path(str(built))
    raise ServiceError("helper binary not found; build it with `just exec-build`")


def helper_start_argv(elevator: str, systemd_run: str, helper: str, uid: int, socket_path: str):
    """The one elevated command: a transient system unit running the helper. An argv list."""
    return (
        elevator,
        systemd_run,
        f"--unit={HELPER_UNIT}",
        "--collect",
        "--quiet",
        "--description=jevmon exec helper (eBPF exec feed)",
        *(f"--property={p}" for p in UNIT_PROPERTIES),
        "--",
        helper,
        "--uid",
        str(uid),
        "--socket",
        socket_path,
    )


def helper_stop_argv(elevator: str, systemctl: str) -> tuple[str, ...]:
    return (elevator, systemctl, "stop", f"{HELPER_UNIT}.service")


@dataclass(frozen=True, slots=True)
class HelperState:
    answering: bool  # the socket accepted us and said hello
    unit: str  # "transient", "installed" or "" (no such unit / unknown)
    active: bool


def probe_socket(path: str, timeout: float = 1.0) -> bool:
    """True if a helper accepts this user on `path` and greets with the expected protocol."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(path)
        data = b""
        while b"\n" not in data and len(data) < 4096:
            chunk = sock.recv(1024)
            if not chunk:
                return False
            data += chunk
        hello = json.loads(data.split(b"\n")[0])
        return hello.get("t") == "hello" and int(hello.get("v", 0)) == PROTOCOL
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    finally:
        sock.close()


def unit_state(
    systemctl: str = "systemctl", run: Callable[..., subprocess.CompletedProcess] = subprocess.run
) -> tuple[str, bool]:
    """(`transient` | `installed` | ``, active) for the helper unit, read without root."""
    try:
        out = run(
            [
                systemctl,
                "show",
                HELPER_UNIT + ".service",
                "-p",
                "LoadState",
                "-p",
                "ActiveState",
                "-p",
                "Transient",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "", False
    props = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    if props.get("LoadState") in (None, "not-found"):
        return "", False
    kind = "transient" if props.get("Transient") == "yes" else "installed"
    return kind, props.get("ActiveState") == "active"


def plan_start(
    config: Config,
    *,
    helper_answering: bool,
    uid: int,
    which: Callable[[str], str | None] = shutil.which,
    helper_finder: Callable[[Config], str] = find_helper,
) -> Plan:
    """What `jevmon start` does for this config, as data."""
    plan = Plan()
    source = config.exec.source
    if source == "ebpf":
        if helper_answering:
            plan.messages.append("exec helper already running")
        else:
            name, elevator = choose_elevator(which)
            systemd_run = which("systemd-run")
            if not systemd_run:
                raise ServiceError("systemd-run not found; cannot start the helper")
            helper = helper_finder(config)
            plan.messages.append(
                f"source is ebpf: starting the exec helper as root through {name} "
                f"({helper}); expect one password prompt"
            )
            plan.steps.append(
                Step(
                    helper_start_argv(
                        elevator, systemd_run, helper, uid, config.exec.helper_socket
                    ),
                    elevated=True,
                    note="start helper",
                )
            )
    else:
        plan.messages.append(f"source is {source}: no root helper needed")
    systemctl = which("systemctl") or "systemctl"
    plan.steps.append(Step((systemctl, "--user", "start", DAEMON_UNIT), note="start daemon"))
    return plan


def plan_stop(
    *,
    unit_kind: str,
    unit_active: bool,
    which: Callable[[str], str | None] = shutil.which,
) -> Plan:
    """What `jevmon stop` does: stop the daemon; stop the helper only if it is a transient unit."""
    plan = Plan()
    systemctl = which("systemctl") or "systemctl"
    plan.steps.append(Step((systemctl, "--user", "stop", DAEMON_UNIT), note="stop daemon"))
    if unit_active and unit_kind == "transient":
        name, elevator = choose_elevator(which)
        plan.messages.append(f"stopping the exec helper through {name}; expect one password prompt")
        plan.steps.append(
            Step(helper_stop_argv(elevator, systemctl), elevated=True, note="stop helper")
        )
    elif unit_active and unit_kind == "installed":
        plan.messages.append(
            "the exec helper was installed with `just exec-install`; leaving it running "
            "(`just exec-uninstall` removes it)"
        )
    return plan


def execute(
    plan: Plan,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    after: dict[str, Callable[[], bool]] | None = None,
) -> int:
    """Run the steps in order, stopping at the first failure. Elevated steps keep the terminal so
    the elevation tool can prompt."""
    for message in plan.messages:
        sys.stderr.write(f"jevmon: {message}\n")
    for step in plan.steps:
        result = run(list(step.argv), check=False)
        if result.returncode != 0:
            sys.stderr.write(
                f"jevmon: {step.note or step.argv[0]} failed (exit {result.returncode})\n"
            )
            return result.returncode or 1
        check = (after or {}).get(step.note)
        if check is not None and not check():
            sys.stderr.write(f"jevmon: {step.note}: the helper did not answer in time\n")
            return 1
    return 0


def wait_for_helper(path: str, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if probe_socket(path):
            return True
        time.sleep(0.25)
    return False


def repo_root() -> Path | None:
    """The checkout this package runs from (a directory with flake.nix above it), if any."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "flake.nix").exists():
            return parent
    return None


def daemon_active(systemctl: str = "systemctl") -> bool:
    try:
        return (
            subprocess.run(
                [systemctl, "--user", "is-active", "--quiet", DAEMON_UNIT], check=False, timeout=5
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def status_lines(config: Config, daemon_status: dict | None) -> list[str]:
    """Plain status for `jevmon status`: source, helper, daemon. Unprivileged."""
    source = config.exec.source
    lines = [f"source  {source}"]
    if source == "ebpf":
        answering = probe_socket(config.exec.helper_socket)
        kind, active = unit_state()
        how = f", {kind} unit" if kind else ""
        lines.append(
            f"helper  {'running' if answering else 'not running'}{how if answering or active else ''}"
        )
    active = daemon_active()
    lines.append(f"daemon  {'active' if active else 'not running'}")
    if daemon_status:
        err = daemon_status.get("exec_error")
        if err:
            lines.append(f"        {err}")
        if source == "ebpf" and daemon_status.get("exec_connected"):
            lines.append(
                f"        {daemon_status.get('exec_observed', 0)} execs seen, "
                f"{daemon_status.get('exec_lost', 0)} lost, "
                f"{daemon_status.get('exec_per_s', 0)}/s"
            )
        lines.append(
            f"        judged {daemon_status.get('judged', 0)}, queue {daemon_status.get('queue_depth', 0)}"
        )
    return lines


def argv_text(argv: Sequence[str]) -> str:
    import shlex

    return " ".join(shlex.quote(a) for a in argv)

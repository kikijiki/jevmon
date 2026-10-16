"""Rootless podman pod for the lab: an internal-only network, a systemd target and an actor.

Every container gets hard cgroup limits and only the internal network (no LAN, no internet).
Teardown removes containers and the network even on failure (`try/finally` in the runner). The
podman calls go through an injectable `Executor` so lifecycle/teardown logic is unit-testable
without podman running.
"""

import subprocess
import time
from dataclasses import dataclass
from typing import Protocol

from jevmon_lab.manifest import Limits

TARGET_IMAGE = "jevlab-target:latest"
ACTOR_IMAGE = "jevlab-actor:latest"
SCRATCH = "/scratch"


class Executor(Protocol):
    def run(self, argv: list[str], timeout: float) -> subprocess.CompletedProcess: ...


class RealExecutor:
    def run(self, argv: list[str], timeout: float) -> subprocess.CompletedProcess:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def network_create_argv(name: str) -> list[str]:
    return ["podman", "network", "create", "--internal", name]


def network_remove_argv(name: str) -> list[str]:
    return ["podman", "network", "rm", "--force", name]


def target_run_argv(name: str, image: str, net: str, limits: Limits) -> list[str]:
    return [
        "podman",
        "run",
        "-d",
        "--replace",
        "--name",
        name,
        "--systemd=always",
        "--network",
        net,
        "--network-alias",
        "target",
        "--memory",
        limits.memory,
        "--memory-swap",
        limits.memory,
        "--cpus",
        limits.cpus,
        "--pids-limit",
        str(limits.pids),
        "--tmpfs",
        f"{SCRATCH}:size={limits.scratch},mode=1777",
        image,
    ]


def actor_run_argv(name: str, image: str, net: str) -> list[str]:
    return [
        "podman",
        "run",
        "-d",
        "--replace",
        "--name",
        name,
        "--network",
        net,
        "--network-alias",
        "actor",
        "--memory",
        "256m",
        "--memory-swap",
        "256m",
        "--cpus",
        "1.0",
        "--pids-limit",
        "256",
        image,
    ]


def teardown_argv(names: list[str], net: str) -> list[list[str]]:
    return [["podman", "rm", "-f", *names], network_remove_argv(net)]


@dataclass(slots=True)
class CommandResult:
    argv: list[str]
    returncode: int
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Pod:
    """Lifecycle wrapper. `exec_target` / `exec_actor` run shell commands in the containers."""

    def __init__(
        self,
        run_id: str,
        limits: Limits,
        executor: Executor | None = None,
        target_image: str = TARGET_IMAGE,
        actor_image: str = ACTOR_IMAGE,
    ) -> None:
        self.run_id = run_id
        self.limits = limits
        self.executor = executor or RealExecutor()
        self.target_image = target_image
        self.actor_image = actor_image
        self.net = f"jevlab-{run_id}"
        self.target = f"jevlab-{run_id}-target"
        self.actor = f"jevlab-{run_id}-actor"
        self._up = False

    def run_podman(self, argv: list[str], timeout: float = 300.0) -> CommandResult:
        proc = self.executor.run(argv, timeout)
        return CommandResult(argv, proc.returncode, proc.stdout or "", proc.stderr or "")

    def up(self, ready_timeout: float = 90.0) -> None:
        self.run_podman(network_create_argv(self.net))
        res = self.run_podman(
            target_run_argv(self.target, self.target_image, self.net, self.limits)
        )
        if not res.ok and "already exists" not in res.err:
            raise RuntimeError(f"target failed to start: {res.err.strip()}")
        self.run_podman(actor_run_argv(self.actor, self.actor_image, self.net))
        self._up = True
        self.wait_ready(ready_timeout)

    def wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            res = self.exec_target("systemctl is-system-running", timeout=15)
            last = (res.out + res.err).strip()
            # `degraded` is fine: a lab unit intentionally failing must not block startup.
            if res.ok and last in ("running", "degraded"):
                return
            time.sleep(2)
        raise RuntimeError(f"target systemd not ready in {timeout}s (last: {last!r})")

    def exec_target(
        self, command: str, timeout: float = 60.0, exec_args: tuple[str, ...] = ()
    ) -> CommandResult:
        argv = ["podman", "exec", *exec_args, self.target, "sh", "-c", command]
        return self.run_podman(argv, timeout)

    def exec_actor(
        self, command: str, timeout: float = 60.0, exec_args: tuple[str, ...] = ()
    ) -> CommandResult:
        argv = ["podman", "exec", *exec_args, self.actor, "sh", "-c", command]
        return self.run_podman(argv, timeout)

    def container_ids(self, timeout: float = 60.0) -> dict[str, str]:
        """Container id (64-hex) by role, read back from `podman inspect` for mapping."""
        res = self.run_podman(
            ["podman", "inspect", "--format", "{{.Id}}", self.target, self.actor], timeout
        )
        lines = [ln.strip() for ln in res.out.splitlines() if ln.strip()]
        ids: dict[str, str] = {}
        for role, line in zip(("target", "actor"), lines):
            if len(line) == 64 and all(c in "0123456789abcdef" for c in line):
                ids[line] = role
        return ids

    def down(self) -> None:
        for argv in teardown_argv([self.target, self.actor], self.net):
            self.run_podman(argv, timeout=120.0)
        self._up = False

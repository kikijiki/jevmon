"""Tag observed execs with the scenario they belong to: container cgroup plus tag window.

Mapping rules:

1. **Container**: an exec event carries the `cgroup` field produced by `unit_from_cgroup`. A
   rootless podman container process resolves to `libpod-<64-hex>.scope` (possibly with a
   sub-cgroup behind it); the hex id is matched against the ids the runner read back from
   `podman inspect` for the lab's own target/actor containers. Events with no lab container id
   (host processes, other containers such as an unrelated workload on the box) are not lab
   observations and are never recorded.
2. **Tag window**: inside a lab container, an exec whose event time falls in a scenario's window
   `[start, start+duration+grace)` is tagged with that scenario id. Windows are validated
   non-overlapping per container, so the tag is unambiguous. Lab-container execs outside every
   window are kept as untagged container background (no ground-truth label).

Limits by design: attribution is by container, not by process tree, so an unrelated process that
the container's own init starts inside a window (the lab keeps containers otherwise idle to make
this rare) would be tagged to the scenario; a scenario process scanned after its window closed
(dragged past a step timeout) is not tagged. Host-side churn never enters the dataset.
"""

import re
from dataclasses import dataclass

from jevmon_lab.manifest import ResolvedScenario

_LIBPOD = re.compile(r"libpod-([0-9a-f]{64})")


def container_id_from_cgroup(cgroup: str) -> str | None:
    """The 64-hex podman container id in an event's cgroup text, if it names one."""
    match = _LIBPOD.search(cgroup)
    return match.group(1) if match else None


def build_container_ids(target_id: str, actor_id: str) -> dict[str, str]:
    """Map container id -> role for the lab's own containers."""
    return {target_id: "target", actor_id: "actor"}


@dataclass(frozen=True, slots=True)
class Window:
    scenario_id: str
    container: str  # "target" | "actor"
    start: float
    end: float  # exclusive; includes the grace tail
    drivers: tuple[str, ...] = ()  # normalized `sh -c <cmd>` argv the scenario itself launches
    payload: str = ""  # normalized argv prefix of the labelled process when not the driver


def build_windows(resolved: tuple[ResolvedScenario, ...], grace: float) -> tuple[Window, ...]:
    """Absolute tag windows for every scenario, grace tail included."""
    return tuple(
        Window(
            s.id,
            s.container,
            s.start,
            s.end + grace,
            driver_argvs(s.commands),
            normalize_argv(s.payload),
        )
        for s in resolved
        if not s.skip
    )


def normalize_argv(text: str) -> str:
    """Whitespace-normalised argv text, the form `designated` compares."""
    return " ".join(text.split())


def driver_argvs(commands: tuple[str, ...]) -> tuple[str, ...]:
    """The redacted `sh -c <cmd>` argv strings the lab itself starts for a scenario's steps."""
    from jevmon.redact import redact_argv_text

    return tuple(normalize_argv(redact_argv_text(["sh", "-c", cmd])) for cmd in commands)


def designated(window: "Window | None", argv: str) -> bool | None:
    """True when `argv` is one of the scenario's own driver commands (the process the scenario
    designates), False for any other exec inside its window (helpers, children), None when there
    is no window or the capture recorded no argv (old runs)."""
    if window is None or not argv:
        return None
    text = normalize_argv(argv)
    if window.payload:  # the scenario names its subject: only that process carries the label
        return text.startswith(window.payload)
    return text in window.drivers


def tag(
    cgroup: str,
    time: float,
    windows: tuple[Window, ...],
    container_ids: dict[str, str],
) -> tuple[str | None, str | None]:
    """(scenario_id, container_role) for an exec event; (None, None) when it is not a lab exec."""
    cid = container_id_from_cgroup(cgroup)
    if cid is None:
        return None, None
    role = container_ids.get(cid)
    if role is None:
        return None, None
    for window in windows:
        if window.container == role and window.start <= time < window.end:
            return window.scenario_id, role
    return None, role

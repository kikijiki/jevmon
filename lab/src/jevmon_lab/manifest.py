"""Lab scenario manifest: typed schema plus TOML parsing and validation.

A scenario is a scripted process-launch sequence inside one lab container: it starts `at` seconds
after the run, lasts `duration` seconds and drives its shell `steps` in that container. Ground
truth is the scenario's `label` - one of the judge's three classes `safe`, `warn`, `crit` (see
`[judge.categories]` in `src/jevmon/default.toml`) - known by construction. A scenario without a
label carries an `excluded` reason instead (rate and resource scenarios have no single-launch
label): it is captured and kept in the manifest but never labelled, exported for training or
scored. `family` groups parameterised variants; a whole
family goes to one side of the train/held-out split (see `split_of`).

Validation enforces what the mapping needs: unique ids, one label per family, and non-overlapping
tag windows per container (a window covers `at .. at+duration+grace`; overlapping windows in one
container would make observation tagging ambiguous).
"""

import hashlib
import re
import tomllib
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path

# The judge's classes, in prompt order (see [judge.categories] in default.toml).
LABELS = ("safe", "warn", "crit")
CONTAINERS = ("target", "actor")
HELDOUT_PERCENT = 30
SPLIT_SALT = "jevmon-lab-v1"  # frozen: changing it changes which families are held out
DEFAULT_GRACE = 3.0  # seconds of tagging tail after a scenario's window


def split_of(family: str, percent: int = HELDOUT_PERCENT) -> str:
    """Deterministic train / heldout split by scenario family, so a whole family is unseen."""
    digest = hashlib.sha256(f"{SPLIT_SALT}:family:{family}".encode()).digest()
    return "heldout" if int.from_bytes(digest[:4], "big") % 100 < percent else "train"


class ManifestError(ValueError):
    """Invalid manifest; the message names the offending scenario or key."""


@dataclass(frozen=True, slots=True)
class Step:
    cmd: str
    timeout: float = 60.0


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    family: str
    label: str
    container: str = "target"
    expected: int = 0  # payload launches by construction; 0 = not counted (e.g. background)
    at: float = 0.0  # seconds after run start
    duration: float = 20.0
    params: dict = field(default_factory=dict)
    steps: tuple[Step, ...] = ()
    skip: str = ""
    excluded: str = ""  # why this scenario has no label (set exactly when `label` is empty)
    # Extra `podman exec` options for the driver (user, workdir, tty, env), so the driver's own
    # launch-time facts (privileges, workdir, session) can differ between scenarios.
    exec_args: tuple[str, ...] = ()
    # When set, the labelled process is the one whose argv starts with this text (after
    # `format_cmd`), not the `sh -c` driver; the driver then carries no label. Used by scenarios
    # whose subject is a process with a particular exe state or argv0.
    payload: str = ""

    def payload_text(self) -> str:
        return format_cmd(self.payload, self.params) if self.payload else ""

    def commands(self) -> list[str]:
        return [format_cmd(s.cmd, self.params) for s in self.steps]


@dataclass(frozen=True, slots=True)
class Limits:
    memory: str = "512m"
    cpus: str = "1.0"
    pids: int = 256
    scratch: str = "64m"  # size-limited tmpfs at /scratch


@dataclass(frozen=True, slots=True)
class Manifest:
    run_seconds: float
    limits: Limits
    scenarios: tuple[Scenario, ...]
    grace_seconds: float = DEFAULT_GRACE

    def window_len(self, scenario: Scenario) -> float:
        """Tag-window length of one scenario: its window plus the grace tail."""
        return scenario.duration + self.grace_seconds

    def selected(self, only: tuple[str, ...] = (), smoke: bool = False) -> "Manifest":
        """Return a manifest restricted to a selection, re-scheduled to start near t0.

        The full manifest spreads scenarios over several minutes. A selection (smoke, or `--only`)
        is compacted so it is quick to run: chosen scenarios are re-timed from +5s with the spacing
        their own window needs, which keeps `just lab-smoke` well under three minutes.
        """
        if smoke:
            chosen = [s for s in self.scenarios if s.id in SMOKE_IDS]
        elif only:
            chosen = [s for s in self.scenarios if s.family in only or s.id in only]
        else:
            chosen = list(self.scenarios)
        if not chosen:
            raise ManifestError(f"no scenarios match only={only} smoke={smoke}")
        if only or smoke:
            chosen.sort(key=lambda s: (s.at, s.id))
            out: list[Scenario] = []
            cursor = 5.0
            for s in chosen:
                out.append(replace(s, at=cursor))
                cursor += self.window_len(s) + 2.0
            chosen = out
            run_seconds = cursor + 5.0
        else:
            run_seconds = self.run_seconds
        return replace(self, scenarios=tuple(chosen), run_seconds=run_seconds)


# Three fast, complementary scenarios that exercise a safe fan-out, a crit copied binary and an
# excluded (unlabelled) CPU spinner. Kept under `just lab-smoke` (well under 3 minutes with pod up).
SMOKE_IDS = ("build_fanout.a", "tmpfs_copy.a", "cpu_spin.a")


@dataclass(frozen=True, slots=True)
class ResolvedScenario:
    """A scenario pinned to absolute wall-clock times; what the capture and dataset consume."""

    id: str
    family: str
    label: str
    container: str
    expected: int
    start: float
    end: float
    params: dict
    skip: str = ""
    excluded: str = ""
    commands: tuple[str, ...] = ()  # rendered step commands (what the driver shells run)
    payload: str = ""  # argv prefix of the labelled process when not the driver


_TEMPLATE = re.compile(r"\{(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\}")


def format_cmd(cmd: str, params: dict) -> str:
    """Substitute `{key}` placeholders from params.

    Unknown `{...}` is left untouched, so shell format strings pass through unchanged. Use this
    only for lab-authored commands.
    """

    def sub(match: re.Match[str]) -> str:
        key = match.group("key")
        return str(params[key]) if key in params else match.group(0)

    return _TEMPLATE.sub(sub, cmd)


def _scenario(raw: dict, where: str) -> Scenario:
    if not isinstance(raw, dict):
        raise ManifestError(f"{where}: must be a table")
    for key in ("id", "family"):
        if key not in raw:
            raise ManifestError(f"{where}: missing `{key}`")
    label = str(raw.get("label", ""))
    excluded = str(raw.get("excluded", ""))
    if label and excluded:
        raise ManifestError(f"{where}: a labelled scenario cannot also be `excluded`")
    if not label and not excluded:
        raise ManifestError(f"{where}: needs a `label` ({LABELS}) or an `excluded` reason")
    if label and label not in LABELS:
        raise ManifestError(f"{where}.label: use one of {LABELS}")
    container = str(raw.get("container", "target"))
    if container not in CONTAINERS:
        raise ManifestError(f"{where}.container: use one of {CONTAINERS}")
    steps = raw.get("steps", [])
    if not isinstance(steps, list):
        raise ManifestError(f"{where}.steps: must be an array of tables")
    parsed = [_step(s, f"{where}.steps[{i}]") for i, s in enumerate(steps)]
    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise ManifestError(f"{where}.params: must be a table")
    expected = int(raw.get("expected", 0))
    if expected < 0:
        raise ManifestError(f"{where}.expected: must be >= 0")
    return Scenario(
        id=str(raw["id"]),
        family=str(raw["family"]),
        label=label,
        container=container,
        expected=expected,
        at=float(raw.get("at", 0.0)),
        duration=float(raw.get("duration", 20.0)),
        params=params,
        steps=tuple(parsed),
        skip=str(raw.get("skip", "")),
        excluded=excluded,
        exec_args=tuple(str(a) for a in raw.get("exec_args", [])),
        payload=str(raw.get("payload", "")),
    )


def _step(raw: dict, where: str) -> Step:
    if not isinstance(raw, dict):
        raise ManifestError(f"{where}: must be a table")
    if "cmd" not in raw:
        raise ManifestError(f"{where}: missing `cmd`")
    timeout = float(raw.get("timeout", 60.0))
    if timeout <= 0:
        raise ManifestError(f"{where}.timeout: must be > 0")
    return Step(str(raw["cmd"]), timeout)


def _limits(raw: dict) -> Limits:
    if not isinstance(raw, dict):
        raise ManifestError("limits: must be a table")
    d = Limits()
    return Limits(
        memory=str(raw.get("memory", d.memory)),
        cpus=str(raw.get("cpus", d.cpus)),
        pids=int(raw.get("pids", d.pids)),
        scratch=str(raw.get("scratch", d.scratch)),
    )


def _validate(scenarios: tuple[Scenario, ...], grace: float) -> None:
    seen: set[str] = set()
    family_labels: dict[str, str] = {}
    for s in scenarios:
        if s.id in seen:
            raise ManifestError(f"duplicate scenario id {s.id!r}")
        seen.add(s.id)
        prior = family_labels.setdefault(s.family, s.label)
        if prior != s.label:
            raise ManifestError(
                f"family {s.family!r} has labels {prior!r} and {s.label!r}; "
                "one family must have one ground-truth label"
            )
    # Tag windows must not overlap inside one container: `mapping.tag` attributes by container
    # plus time, so two live windows in the same container would be ambiguous.
    by_container: dict[str, list[Scenario]] = {}
    for s in scenarios:
        by_container.setdefault(s.container, []).append(s)
    for container, group in by_container.items():
        ordered = sorted(group, key=lambda s: s.at)
        for prev, cur in pairwise(ordered):
            if cur.at < prev.at + prev.duration + grace:
                raise ManifestError(
                    f"scenarios {prev.id!r} and {cur.id!r} overlap in container {container!r} "
                    f"(window {prev.at:g}..{prev.at + prev.duration + grace:g} vs at={cur.at:g})"
                )


def load(path: Path) -> Manifest:
    data = tomllib.loads(Path(path).read_text())
    raw_scenarios = data.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ManifestError("manifest needs at least one [[scenarios]] entry")
    scenarios = tuple(_scenario(s, f"scenarios[{i}]") for i, s in enumerate(raw_scenarios))
    grace = float(data.get("grace_seconds", DEFAULT_GRACE))
    if grace < 0:
        raise ManifestError("grace_seconds: must be >= 0")
    _validate(scenarios, grace)
    run_seconds = float(data.get("run_seconds", 240.0))
    if any(s.at + s.duration + grace > run_seconds for s in scenarios):
        raise ManifestError("a scenario's window exceeds run_seconds; extend run_seconds")
    return Manifest(
        run_seconds=run_seconds,
        limits=_limits(data.get("limits", {})),
        scenarios=scenarios,
        grace_seconds=grace,
    )


def resolve(scenarios: tuple[Scenario, ...], t0: float) -> tuple[ResolvedScenario, ...]:
    """Pin every scenario to absolute `[start, end)` times given a run start `t0`."""
    return tuple(
        ResolvedScenario(
            id=s.id,
            family=s.family,
            label=s.label,
            container=s.container,
            expected=s.expected,
            start=t0 + s.at,
            end=t0 + s.at + s.duration,
            params=s.params,
            skip=s.skip,
            excluded=s.excluded,
            commands=tuple(s.commands()),
            payload=s.payload_text(),
        )
        for s in scenarios
    )

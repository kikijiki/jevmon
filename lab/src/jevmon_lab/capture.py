"""Run the real exec pipeline (host /proc poller + context builder) against a lab run.

The capture is the system under test: `ExecSource` polls the host `/proc` through `RealProcReader`
exactly as the daemon does (rootless podman processes are visible there with their container
cgroup), and `ContextBuilder` turns each exec into model-facing context text. Every observed exec
in a lab container is tagged with its scenario via `mapping.tag`; contexts are recorded, not
judged (see `dataset.py` for judge modes).

Structural facts (`RealSystemFacts`) would stat container paths against the host filesystem, where
they do not exist. `RootedSystemFacts` reuses the process's own root and mount table
(`/proc/<pid>/root`, `/proc/<pid>/mountinfo`) so `writable_dir`, ownership, mtime and mount type
describe the file the process actually executed - as long as the pid is still alive, else the
facts degrade to defaults.
"""

import dataclasses
import os
import stat
import threading
import time
from collections import OrderedDict

from jevmon.core import EVENT_KIND_EXEC, MemoryCounts
from jevmon.detect.context import Context, ContextBuilder, EnvironmentStats, ExecObservation
from jevmon.detect.stats import ForkRateBaseline
from jevmon.detect.system_facts import Mount, StructuralFacts, mount_for_path, parse_mounts
from jevmon.events import Event
from jevmon.redact import redact_argv_text
from jevmon.sources.exec import KIND_SUMMARY, ExecSettings, ExecSource
from jevmon.sources.proc import RealProcReader
from jevmon_lab.manifest import ResolvedScenario
from jevmon_lab.mapping import Window, build_windows, designated, tag


class RootedSystemFacts:
    """Inspect executables through the owning process's root (lab-only adapter).

    The builder passes an exe path only, so the capture sets `current_pid` from the event being
    built just before calling `observe_exec` (single writer thread). `proc_root` follows the
    source, so tests can point it at a fake /proc tree.
    """

    MOUNT_CACHE_MAX = 128

    def __init__(self, proc_root: str = "/proc") -> None:
        self.proc_root = proc_root.rstrip("/") or "/proc"
        self.current_pid: int | None = None
        self._mounts: OrderedDict[int, list[Mount]] = OrderedDict()

    def inspect(self, exe: str, now: float, recent_seconds: float) -> StructuralFacts:
        pid = self.current_pid
        if not exe or pid is None:
            return StructuralFacts()
        root = f"{self.proc_root}/{pid}/root{exe}"
        try:
            st = os.stat(root)
            parent = os.stat(os.path.dirname(root) or f"{self.proc_root}/{pid}/root/")
        except OSError:
            return StructuralFacts()
        mount = mount_for_path(exe, self._mounts_for(pid))
        written = max(st.st_mtime, st.st_ctime)
        return StructuralFacts(
            readable=True,
            writable_dir=bool(parent.st_mode & (stat.S_IWOTH | stat.S_IWGRP)),
            owned_by_nonroot=st.st_uid != 0,
            recently_written=(now - written) <= recent_seconds,
            mount_type=mount.fstype if mount else "unknown",
            on_tmpfs_or_noexec=bool(
                mount and (mount.fstype == "tmpfs" or "noexec" in mount.options)
            ),
        )

    def _mounts_for(self, pid: int) -> list[Mount]:
        cached = self._mounts.get(pid)
        if cached is not None:
            self._mounts.move_to_end(pid)
            return cached
        try:
            with open(
                f"{self.proc_root}/{pid}/mountinfo", encoding="utf-8", errors="replace"
            ) as fh:
                mounts = parse_mounts(fh.read())
        except OSError:
            return []
        self._mounts[pid] = mounts
        while len(self._mounts) > self.MOUNT_CACHE_MAX:
            self._mounts.popitem(last=False)
        return mounts


def _environ(data: bytes | None) -> dict[str, str]:
    """`/proc/<pid>/environ` as an ordered dict (empty when unreadable)."""
    out: dict[str, str] = {}
    for item in (data or b"").split(b"\0"):
        key, sep, value = os.fsdecode(item).partition("=")
        if sep and key:
            out[key] = value
    return out


class RawExecSource(ExecSource):
    """Lab-only source: also puts the unmasked (sanitized, length-bounded) argv, exe and cwd on
    the event as `*_raw` fields, so the lab can build the model-facing text from the raw command
    line. The daemon's source never does this; lab data is synthetic."""

    def _make_event(self, now, stat, argv, exe):  # type: ignore[no-untyped-def]
        event = super()._make_event(now, stat, argv, exe)
        decoded = [os.fsdecode(a) for a in argv]
        env = _environ(self.reader.read_file(stat.pid, "environ"))
        parent_env = _environ(self.reader.read_file(stat.ppid, "environ")) if stat.ppid > 0 else {}
        own = dict(os.environ)
        extra = {
            "env_raw": "\n".join(f"{k}={v}" for k, v in env.items())[:16000],
            # variables new or changed against the parent / against the capture process: the raw
            # parent and daemon environments are never stored (they hold the host's own values)
            "env_parent_diff": "\n".join(
                f"{k}={v}" for k, v in env.items() if parent_env.get(k) != v
            )[:8000],
            "env_base_diff": "\n".join(f"{k}={v}" for k, v in env.items() if own.get(k) != v)[
                :8000
            ],
            "argv_raw": redact_argv_text(decoded, mask=False),
            "exe_raw": redact_argv_text([exe], mask=False) if exe else "",
            "cwd_raw": event.fields.get("cwd", ""),
        }
        return dataclasses.replace(event, fields={**event.fields, **extra})


class Observation:
    """One recorded exec: the built context plus its ground-truth mapping."""

    __slots__ = (
        "argv",
        "container",
        "context",
        "context_raw",
        "designated",
        "env",
        "scenario_id",
    )

    def __init__(
        self,
        context: Context,
        scenario_id: str | None,
        container: str,
        designated: bool | None = None,
    ) -> None:
        self.context = context
        self.scenario_id = scenario_id
        self.container = container
        # True: the process the scenario itself launched (its label is the ground truth);
        # False: a descendant/helper in the window (no label of its own); None: unknown (old run).
        self.designated = designated
        self.argv = ""  # the redacted argv the poller saw (what `designated` compared)
        self.env: dict[str, str] = {}  # raw env, parent-diff and base-diff (lab data only)
        self.context_raw = ""  # model-facing state built from the unmasked argv, if captured

    def as_dict(self) -> dict:
        """The raw row written to `contexts.jsonl`."""
        return {
            "time": round(self.context.time, 3),
            "subject": self.context.subject,
            "kind": self.context.kind,
            "signature": self.context.signature,
            "context": self.context.state,
            "scenario_id": self.scenario_id,
            "container": self.container,
            "fields": self.context.fields,
            "designated": self.designated,
            "argv": self.argv,
            "context_raw": self.context_raw,
            **self.env,
        }


class Capture:
    """Poll host /proc in a background thread and record tagged lab contexts."""

    def __init__(
        self,
        exec_settings: ExecSettings,
        context_settings,
        windows: tuple[Window, ...],
        container_ids: dict[str, str],
        reader=None,
    ) -> None:
        self.windows = windows
        self.container_ids = container_ids
        self.facts = RootedSystemFacts(exec_settings.proc_root)
        self.builder = ContextBuilder(MemoryCounts(), self.facts, context_settings)
        self.env = EnvironmentStats()
        self.baseline = ForkRateBaseline()
        # Wall clock, not monotonic: tag windows and ground-truth records are wall times.
        self.raw_builder = ContextBuilder(MemoryCounts(), self.facts, context_settings)
        self.source = RawExecSource(
            reader if reader is not None else RealProcReader(exec_settings.proc_root),
            exec_settings,
            emit=self._on_event,
            clock=time.time,
        )
        self.observations: list[Observation] = []
        self.host_events = 0
        self.background_events = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def for_run(
        cls,
        exec_settings: ExecSettings,
        context_settings,
        resolved: tuple[ResolvedScenario, ...],
        container_ids: dict[str, str],
        grace: float,
        reader=None,
    ) -> "Capture":
        return cls(
            exec_settings,
            context_settings,
            build_windows(resolved, grace),
            container_ids,
            reader=reader,
        )

    # -- lifecycle ------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="lab-capture", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        self.source.run(self._stop)

    # -- event flow -----------------------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        """ExecSource emit hook. Single writer thread; readers run after `stop()`."""
        if event.kind == KIND_SUMMARY:
            self.env.fork_rate = float(event.fields.get("forks_per_s", 0.0))
            self.baseline.update(self.env.fork_rate)
            self.env.fork_baseline = self.baseline.ewma
            self.env.miss_ratio = float(event.fields.get("miss_ratio", 0.0))
            return
        if event.kind != EVENT_KIND_EXEC:
            return
        cgroup = str(event.fields.get("cgroup", ""))
        scenario_id, role = tag(cgroup, event.time, self.windows, self.container_ids)
        if role is None:
            self.host_events += 1
            return
        pid = int(str(event.fields.get("pid", 0)) or 0)
        self.facts.current_pid = pid if pid > 0 else None
        context = self.builder.observe_exec(ExecObservation.from_event(event), self.env)
        raw = event.fields.get("argv_raw")
        raw_state = ""
        if raw is not None:
            raw_event = dataclasses.replace(
                event,
                fields={
                    **event.fields,
                    "argv": raw,
                    "exe": event.fields.get("exe_raw", ""),
                },
            )
            raw_state = self.raw_builder.observe_exec(
                ExecObservation.from_event(raw_event), self.env
            ).state
        window = next((w for w in self.windows if w.scenario_id == scenario_id), None)
        argv = str(event.fields.get("argv", ""))
        observation = Observation(context, scenario_id, role, designated(window, argv))
        observation.argv = argv
        observation.context_raw = raw_state
        observation.env = {
            k: str(event.fields.get(k, "")) for k in ("env_raw", "env_parent_diff", "env_base_diff")
        }
        self.observations.append(observation)
        if scenario_id is None:
            self.background_events += 1

    # -- results --------------------------------------------------------------------------

    @property
    def tagged(self) -> int:
        return sum(1 for o in self.observations if o.scenario_id is not None)

    def poller_stats(self) -> dict:
        """Source-level counters for the report (host-global, measured by the real poller)."""
        source = self.source
        return {
            "observed_total": source.observed_total,
            "estimated_missed": source.missed_total,
            "miss_ratio": round(source.miss_ratio, 4),
            "effective_interval": round(source.effective_interval, 4),
            "tracked": source.tracked,
            "lab_contexts": len(self.observations),
            "tagged_contexts": self.tagged,
            "untagged_container_contexts": self.background_events,
            "host_events_not_recorded": self.host_events,
        }

"""Run the lab: bring up the pod, drive scenario timelines, capture the exec pipeline's contexts.

The capture (`jevmon_lab.capture.Capture`) is the system under test: it polls the host `/proc`
with the real exec source and context builder while scenarios run, tagging every observed exec
with its scenario (container cgroup + tag window, see `mapping.py`). This module writes raw
`contexts.jsonl` (every recorded lab context with its tag), `scenarios.jsonl` (ground truth by
construction plus observed/missed launch counts) and `meta.json` under the output directory;
`dataset.py` turns that into a judged dataset and report. Teardown runs in a `finally` so a failed
scenario, timeout or Ctrl-C still removes every container and the network.
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from jevmon.config import Config, load_config
from jevmon.sources.facts import FACT_NAMES
from jevmon_lab.capture import Capture, Observation
from jevmon_lab.manifest import Manifest, Scenario, format_cmd, resolve, split_of
from jevmon_lab.pod import Pod


def _sleep_until(target: float) -> None:
    delta = target - time.time()
    if delta > 0:
        time.sleep(delta)


def _run_scenario(pod: Pod, scenario: Scenario, t0: float, log) -> dict:
    """Run one scenario's steps in its container and return its ground-truth record."""
    scheduled_start = t0 + scenario.at
    scheduled_end = scheduled_start + scenario.duration
    if scenario.skip:
        log(f"  skip  {scenario.id}: {scenario.skip}")
        return _record(
            scenario, scheduled_start, scheduled_end, start=None, end=None, skip=scenario.skip
        )
    _sleep_until(scheduled_start)
    start = time.time()
    errors: list[str] = []
    runner = pod.exec_actor if scenario.container == "actor" else pod.exec_target
    for step in scenario.steps:
        try:
            cmd = format_cmd(step.cmd, scenario.params)
            res = (
                runner(cmd, timeout=step.timeout, exec_args=scenario.exec_args)
                if scenario.exec_args
                else runner(cmd, timeout=step.timeout)
            )
            if not res.ok:
                errors.append(f"rc={res.returncode} {res.err.strip()[:200]}")
        except Exception as exc:  # noqa: BLE001 - one bad step must not abort the whole lab
            errors.append(f"{type(exc).__name__}: {exc}")
    end = time.time()
    status = "ok" if not errors else "error"
    log(
        f"  run   {scenario.id} [{status}] at +{scenario.at:g}s"
        + (f" ({'; '.join(errors)})" if errors else "")
    )
    return _record(scenario, scheduled_start, scheduled_end, start=start, end=end, errors=errors)


def _record(
    scenario: Scenario,
    scheduled_start: float,
    scheduled_end: float,
    *,
    start: float | None,
    end: float | None,
    errors: list[str] | None = None,
    skip: str = "",
) -> dict:
    return {
        "id": scenario.id,
        "family": scenario.family,
        "label": scenario.label,
        "excluded": scenario.excluded,
        "container": scenario.container,
        "expected": scenario.expected,
        "split": split_of(scenario.family),
        "window_start": round(scheduled_start, 3),
        "window_end": round(scheduled_end, 3),
        "start": round(start, 3) if start is not None else None,
        "end": round(end, 3) if end is not None else None,
        "skip": skip,
        "errors": errors or [],
        "observed": 0,
        "missed": 0,
        "observed_any": False,
    }


@dataclass(frozen=True, slots=True)
class RunResult:
    meta: dict
    records: tuple[dict, ...]
    observations: tuple[Observation, ...]
    poller: dict


def run(
    manifest: Manifest,
    out_dir: Path,
    *,
    pod: Pod | None = None,
    capture: Capture | None = None,
    config: Config | None = None,
    only: tuple[str, ...] = (),
    smoke: bool = False,
    log=print,
) -> RunResult:
    """Run the selected scenarios and write the raw captures into `out_dir`."""
    selected = manifest.selected(only, smoke) if (only or smoke) else manifest
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{int(time.time())}-{os.getpid()}"
    pod = pod or Pod(run_id, manifest.limits)
    records: list[dict] = []
    capture_started = False
    t0 = time.time()
    try:
        log(
            f"lab run {run_id}: {len(selected.scenarios)} scenarios, up to {selected.run_seconds:g}s"
        )
        pod.up()
        t0 = time.time()
        resolved = resolve(selected.scenarios, t0)
        if capture is None:
            cfg = config if config is not None else load_config()
            capture = Capture.for_run(
                replace(cfg.exec, parent_argv=True, facts=FACT_NAMES),
                # the capture keeps every segment; datasets re-render through the daemon's order
                replace(cfg.context, order=()),
                resolved,
                pod.container_ids(),
                selected.grace_seconds,
            )
        capture.start()
        capture_started = True
        for scenario in sorted(selected.scenarios, key=lambda s: (s.at, s.id)):
            records.append(_run_scenario(pod, scenario, t0, log))
        _sleep_until(t0 + selected.run_seconds)
    finally:
        if capture_started and capture is not None:
            capture.stop()
        pod.down()

    observations = tuple(capture.observations) if capture is not None else ()
    poller = capture.poller_stats() if capture is not None else {}
    _join_observations(records, observations)
    with (out_dir / "contexts.jsonl").open("w") as fh:
        for obs in observations:
            fh.write(json.dumps(obs.as_dict()) + "\n")
    with (out_dir / "scenarios.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    meta = {
        "run_id": run_id,
        "started_at": t0,
        "run_seconds": selected.run_seconds,
        "grace_seconds": selected.grace_seconds,
        "limits": asdict(manifest.limits),
        "scenarios": len(records),
        "skipped": [r["id"] for r in records if r["skip"]],
        "with_errors": [r["id"] for r in records if r["errors"]],
        "contexts": len(observations),
        "poller": poller,
        "judge": {"mode": None},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(
        f"wrote {out_dir}: {len(observations)} contexts, "
        f"{poller.get('tagged_contexts', 0)} tagged, miss_ratio={poller.get('miss_ratio', 0)}"
    )
    return RunResult(meta=meta, records=tuple(records), observations=observations, poller=poller)


def _join_observations(records: list[dict], observations: tuple[Observation, ...]) -> None:
    """Attach observed/missed launch counts per scenario (ground truth vs the poller)."""
    per_scenario = Counter(o.scenario_id for o in observations if o.scenario_id is not None)
    for rec in records:
        observed = per_scenario.get(rec["id"], 0)
        rec["observed"] = observed
        rec["missed"] = max(0, rec["expected"] - observed) if rec["expected"] else 0
        rec["observed_any"] = observed > 0

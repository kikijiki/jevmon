"""Judge the captured contexts, write the JSONL dataset and build the markdown report.

Judge modes (`--judge`):

- `none`: record contexts only. Every row keeps `verdict = null`; this is the training-data
  export path (context text + signature + ground truth, no model involved).
- `fake`: a deterministic stand-in for tests and smoke runs. The label is a stable hash of the
  context text over the four judge categories - reproducible, but it is not a judgement.
- `laya`: the real Laya judge, loaded from the config (`[judge]`: device "auto", categories,
  bands). Contexts are judged through `decide_batch` in fixed chunks; latency is the chunk wall
  time divided over its rows.

The report is honest about what it could not observe: scenarios whose launches the tier-0 poller
missed, untagged in-container background, host-global miss estimates and unavailable verdicts.
"""

import hashlib
import json
import time
from statistics import mean

from jevmon.config import Config, load_config
from jevmon.detect.context import ContextSettings, parse_exec_state, render_segments
from jevmon.detect.stats import percentile
from jevmon.monitor import build_decider
from jevmon_lab.capture import Observation
from jevmon_lab.manifest import LABELS

JUDGE_MODES = ("none", "fake", "laya")
_CHUNK = 16  # contexts per decide_batch call for --judge laya


def build_rows(
    observations: tuple[Observation, ...] | list[Observation],
    records: tuple[dict, ...] | list[dict],
    context_settings: ContextSettings | None = None,
) -> list[dict]:
    """Dataset rows: every recorded lab context joined to its scenario's ground truth.

    With `context_settings` (the daemon's `[context]`), each context is re-rendered through the
    current segment order first, so a run captured under an older format is exported and judged
    exactly as the daemon would show it today. The raw capture in `contexts.jsonl` is untouched.
    """
    by_id = {rec["id"]: rec for rec in records}
    rows: list[dict] = []
    for obs in observations:
        rec = by_id.get(obs.scenario_id) if obs.scenario_id else None
        state = obs.context.state
        if context_settings is not None:
            state = render_segments(parse_exec_state(state), context_settings)
        rows.append(
            {
                "time": obs.context.time,
                "subject": obs.context.subject,
                "signature": obs.context.signature,
                "context": state,
                "scenario_id": obs.scenario_id,
                "container": obs.container,
                # a helper/child in a scenario window carries no label of its own: only the
                # process the scenario designates (its driver command) is ground truth
                "label": (rec["label"] or None) if rec and obs.designated is not False else None,
                "scenario_label": (rec["label"] or None) if rec else None,
                "excluded": (rec.get("excluded") or None) if rec else None,
                "designated": obs.designated,
                "family": rec["family"] if rec else None,
                "split": rec["split"] if rec else None,
                "observed": True,  # every row exists because the poller saw the launch
                "verdict": None,
                "confidence": None,
                "judged": False,
                "latency_ms": None,
            }
        )
    return rows


def fake_verdict(state: str, labels: tuple[str, ...] = LABELS) -> tuple[str, float]:
    """Deterministic stand-in verdict: a stable hash of the context text."""
    digest = hashlib.sha256(state.encode("utf-8", "replace")).digest()
    label = labels[digest[0] % len(labels)]
    confidence = 0.60 + (digest[1] % 40) / 100
    return label, round(confidence, 2)


def judge_rows(rows: list[dict], mode: str, *, config: Config | None = None) -> dict:
    """Fill verdict/confidence/judged/latency_ms on every row; return judge stats for the report."""
    if mode not in JUDGE_MODES:
        raise ValueError(f"judge mode must be one of {JUDGE_MODES}, got {mode!r}")
    if mode == "none" or not rows:
        return {"mode": mode, "judged": 0, "unavailable": 0, "latencies_ms": []}
    if mode == "fake":
        latencies: list[float] = []
        for row in rows:
            started = time.perf_counter()
            label, confidence = fake_verdict(row["context"])
            latency = (time.perf_counter() - started) * 1000
            row.update(
                verdict=label, confidence=confidence, judged=True, latency_ms=round(latency, 3)
            )
            latencies.append(latency)
        return {
            "mode": mode,
            "judged": len(rows),
            "unavailable": 0,
            "device": "fake",
            "status": "deterministic stand-in",
            "latencies_ms": latencies,
        }
    return _judge_laya(rows, config)


def _judge_laya(rows: list[dict], config: Config | None) -> dict:
    cfg = config if config is not None else load_config()
    decider = build_decider(cfg)
    decider.load()
    latencies: list[float] = []
    judged = 0
    unavailable = 0
    for start in range(0, len(rows), _CHUNK):
        chunk = rows[start : start + _CHUNK]
        states = [row["context"] for row in chunk]
        batched = getattr(decider, "decide_batch", None)
        started = time.perf_counter()
        try:
            decisions = (
                list(batched(states)) if callable(batched) else [decider.decide(s) for s in states]
            )
        except Exception:  # noqa: BLE001 - a broken model degrades the batch, never the run
            decisions = None
        elapsed = (time.perf_counter() - started) * 1000
        per_row = elapsed / len(chunk)
        latencies.extend([per_row] * len(chunk))
        if decisions is None:
            unavailable += len(chunk)
            continue
        for row, decision in zip(chunk, decisions, strict=True):
            if decision is None or not decision.available:
                unavailable += 1
                continue
            row.update(
                verdict=decision.label,
                confidence=round(decision.confidence, 3),
                judged=True,
                latency_ms=round(per_row, 3),
            )
            judged += 1
    return {
        "mode": "laya",
        "judged": judged,
        "unavailable": unavailable,
        "device": getattr(decider, "active_device", "") or "cpu",
        "status": decider.status,
        "latencies_ms": latencies,
    }


def write_dataset(rows: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)


# -- report -------------------------------------------------------------------------------


def percentile_ms(values: list[float], q: float) -> float:
    """q-quantile of the per-context latencies in ms (0.0 for an empty list)."""
    return round(percentile(values, q), 2) if values else 0.0


def confusion(rows: list[dict]) -> dict[tuple[str, str], int]:
    """(truth, verdict) counts over judged, labelled rows."""
    matrix: dict[tuple[str, str], int] = {}
    for row in rows:
        if row["label"] and row["judged"] and row["verdict"]:
            key = (row["label"], row["verdict"])
            matrix[key] = matrix.get(key, 0) + 1
    return matrix


def _family_rows(records: list[dict] | tuple[dict, ...]) -> list[dict]:
    families: dict[str, dict] = {}
    for rec in records:
        fam = families.setdefault(
            rec["family"],
            {
                "family": rec["family"],
                "label": rec["label"] or "excluded",
                "split": rec["split"],
                "scenarios": 0,
                "observed": 0,
                "not_observed": 0,
                "expected": 0,
                "observed_launches": 0,
                "missed": 0,
            },
        )
        fam["scenarios"] += 1
        fam["observed" if rec["observed_any"] else "not_observed"] += 1
        fam["expected"] += rec["expected"]
        fam["observed_launches"] += rec["observed"]
        fam["missed"] += rec["missed"]
    return [families[k] for k in sorted(families)]


def _label_rows(records: list[dict] | tuple[dict, ...]) -> list[dict]:
    labels: dict[str, dict] = {}
    for fam in _family_rows(records):
        lab = labels.setdefault(
            fam["label"],
            {
                "label": fam["label"],
                "scenarios": 0,
                "observed": 0,
                "not_observed": 0,
                "expected": 0,
                "observed_launches": 0,
                "missed": 0,
            },
        )
        for key in (
            "scenarios",
            "observed",
            "not_observed",
            "expected",
            "observed_launches",
            "missed",
        ):
            lab[key] += fam[key]
    return [labels[k] for k in (*LABELS, "excluded") if k in labels]


def _table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def _unobservable(
    records: tuple[dict, ...] | list[dict], rows: list[dict], judge_stats: dict
) -> list[str]:
    """The honest-limits bullet list: what the poller or the run did not see."""
    notes: list[str] = []
    for rec in records:
        if rec["skip"]:
            notes.append(f"scenario `{rec['id']}` was skipped: {rec['skip']}")
        elif rec["expected"] and rec["observed"] == 0:
            notes.append(
                f"scenario `{rec['id']}` (expected {rec['expected']} launches): "
                "**no launch was observed** - every payload was shorter than the poll interval"
            )
        elif rec["missed"]:
            notes.append(
                f"scenario `{rec['id']}`: observed {rec['observed']} of {rec['expected']} "
                f"expected launches ({rec['missed']} missed by tier 0)"
            )
        if rec["errors"]:
            notes.append(f"scenario `{rec['id']}` had step errors: {'; '.join(rec['errors'])}")
    untagged = sum(1 for row in rows if row["label"] is None)
    if untagged:
        notes.append(
            f"{untagged} context(s) from lab containers outside every scenario window "
            "(background daemons); they carry no ground-truth label and are excluded from the matrix"
        )
    if judge_stats.get("mode") == "fake":
        notes.append(
            "the judge was the **fake** stand-in (stable hash of the context text): "
            "the confusion matrix and false-positive rate measure the pipeline, not model quality"
        )
    if judge_stats.get("mode") == "none":
        notes.append("no judge ran (`--judge none`): contexts only, no verdicts")
    if judge_stats.get("unavailable"):
        notes.append(
            f"{judge_stats['unavailable']} context(s) could not be judged "
            f"(model status: {judge_stats.get('status', 'unknown')})"
        )
    notes.append(
        "the poller's miss estimate reads the **host-global** `/proc/stat` fork counter: "
        "it counts every fork on the box, not just the lab's"
    )
    notes.append(
        "scenario attribution is container + time window: an unrelated process "
        "started inside the container during a window would be tagged to that scenario"
    )
    return notes


def build_report(
    records: tuple[dict, ...] | list[dict],
    rows: list[dict],
    poller: dict,
    judge_stats: dict,
    meta: dict,
) -> str:
    """The markdown report: per family / per category, confusion, FPR, latency, honest limits."""
    lines: list[str] = ["# jevmon lab report", ""]
    lines.append(
        f"Run `{meta.get('run_id', '?')}` - {len(records)} scenarios, {len(rows)} recorded "
        f"contexts, judge mode `{judge_stats.get('mode', 'none')}`."
    )
    lines += ["", "## Poller (tier 0/0+, host /proc)", ""]
    lines += _table(
        ["metric", "value"],
        [
            ["observed (all pids, incl. first-scan census)", poller.get("observed_total", 0)],
            ["estimated missed (fork counter)", poller.get("estimated_missed", 0)],
            ["miss ratio (source, cumulative)", poller.get("miss_ratio", 0)],
            ["effective poll interval (s)", poller.get("effective_interval", 0)],
            ["tracked pids (end)", poller.get("tracked", 0)],
            ["lab contexts recorded", poller.get("lab_contexts", 0)],
            ["tagged to a scenario", poller.get("tagged_contexts", 0)],
            ["in-container, untagged (background)", poller.get("untagged_container_contexts", 0)],
            ["host events not recorded", poller.get("host_events_not_recorded", 0)],
        ],
    )

    lines += ["", "## Scenarios by family", ""]
    lines += _table(
        [
            "family",
            "label",
            "split",
            "scenarios",
            "observed",
            "no launches seen",
            "expected",
            "observed launches",
            "missed",
        ],
        [
            [
                f"`{f['family']}`",
                f["label"],
                f["split"],
                f["scenarios"],
                f["observed"],
                f["not_observed"],
                f["expected"],
                f["observed_launches"],
                f["missed"],
            ]
            for f in _family_rows(records)
        ],
    )

    lines += ["", "## Scenarios by category (judge labels)", ""]
    lines += _table(
        [
            "label",
            "scenarios",
            "observed",
            "no launches seen",
            "expected",
            "observed launches",
            "missed",
        ],
        [
            [
                lab["label"],
                lab["scenarios"],
                lab["observed"],
                lab["not_observed"],
                lab["expected"],
                lab["observed_launches"],
                lab["missed"],
            ]
            for lab in _label_rows(records)
        ],
    )

    lines += ["", "## Verdict vs truth (confusion matrix)", ""]
    judged_rows = [r for r in rows if r["label"] and r["judged"] and r["verdict"]]
    labelled = [r for r in rows if r["label"]]
    if judged_rows:
        matrix = confusion(rows)
        verdicts = sorted({v for _, v in matrix})
        lines += _table(
            ["truth \\ verdict", *verdicts],
            [
                [truth, *[matrix.get((truth, v), 0) for v in verdicts]]
                for truth in LABELS
                if any(truth == t for t, _ in matrix)
            ],
        )
        lines.append("")
        lines.append(
            f"{len(judged_rows)} of {len(labelled)} labelled context(s) judged; "
            f"{len(labelled) - len(judged_rows)} unjudged."
        )
    else:
        lines.append(
            f"no judged labelled contexts ({len(labelled)} labelled, "
            f"{judge_stats.get('mode')} mode) - no matrix."
        )

    lines += ["", "## Over-flagging of safe launches", ""]
    safe = [r for r in labelled if r["label"] == "safe" and r["judged"] and r["verdict"]]
    if safe:
        flagged = [r for r in safe if r["verdict"] != "safe"]
        crit = [r for r in safe if r["verdict"] == "crit"]
        lines += _table(
            ["metric", "value"],
            [
                ["safe contexts judged", len(safe)],
                ["verdict warn or crit", f"{len(flagged) / len(safe):.1%}"],
                ["verdict crit", f"{len(crit) / len(safe):.1%}"],
            ],
        )
    else:
        lines.append("no judged safe contexts - no over-flagging rate.")

    lines += ["", "## Judge latency", ""]
    latencies = judge_stats.get("latencies_ms", [])
    if latencies:
        lines += _table(
            ["metric", "value"],
            [
                ["mode", judge_stats.get("mode", "none")],
                ["device", judge_stats.get("device", "-")],
                ["status", judge_stats.get("status", "-")],
                ["judged", judge_stats.get("judged", 0)],
                ["unavailable", judge_stats.get("unavailable", 0)],
                ["mean ms/context", round(mean(latencies), 2)],
                ["p50 ms/context", percentile_ms(latencies, 0.50)],
                ["p95 ms/context", percentile_ms(latencies, 0.95)],
                ["total judge wall (s)", round(sum(latencies) / 1000, 2)],
            ],
        )
    else:
        lines += _table(["metric", "value"], [["mode", judge_stats.get("mode", "none")]])

    lines += ["", "## What could not be observed (honest limits)", ""]
    for note in _unobservable(records, rows, judge_stats):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)

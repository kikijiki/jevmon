"""E19: validation, dedupe, label QA and context rendering for the synthetic corpus.

Pipeline stages (3)-(5) of the E19 plan: schema + single-launch-shape
checks, near-duplicate filtering, a second-pass blind QA judgment from the same model, and
rendering through the real `ContextBuilder` (no process is ever run).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

from gen_corpus_lib import (
    CLASS_DESCRIPTIONS,
    LABELS,
    LLMClient,
    extract_json,
    row_signature,
    synthesize_facts,
    unsafe_text_reasons,
)
from jevmon.sources.facts import FACT_NAMES

# A single execve-level launch must not itself encode a narrative of multiple steps joined by
# shell control operators INSIDE an argv element that is not itself a shell's own `-c` string
# (i.e. a bare argv[0] like a compiled binary should not contain ';'/'&&'/'|' - that is the
# generator describing a script, not a launch). `sh -c "...; ..."` is fine: that whole string is
# the one argv element belonging to the shell's own execve.
_SHELL_EXES = re.compile(r"/(ba|da|z|fi|k|tc)?sh$")
_STEP_OPS = re.compile(r"(;|&&|\|\||\n)")


def single_launch_reasons(row: dict) -> list[str]:
    reasons = []
    exe_is_shell = bool(_SHELL_EXES.search(row["exe"]))
    has_c_flag = exe_is_shell and ("-c" in row["argv"] or "-ic" in row["argv"])
    for i, arg in enumerate(row["argv"]):
        if _STEP_OPS.search(arg) and not (has_c_flag and i == len(row["argv"]) - 1):
            reasons.append(f"argv[{i}] looks like a multi-step script body, not one launch")
            break
    if len(row["argv"]) > 1 and not has_c_flag:
        # heuristic: a long literal script string in a non-shell argv is a script body, not argv
        joined = " ".join(row["argv"])
        if len(joined) > 600 and ("\n" in joined or joined.count(";") > 3):
            reasons.append("argv reads as an embedded script body")
    return reasons


@dataclass(slots=True)
class ValidationResult:
    kept: list[dict]
    rejected: list[dict]  # each carries "_reject_reasons"


def validate_rows(rows: list[dict]) -> ValidationResult:
    kept, rejected = [], []
    for row in rows:
        reasons = []
        if not row.get("exe", "").startswith("/"):
            reasons.append("exe not an absolute path")
        if not row.get("argv"):
            reasons.append("empty argv")
        if row.get("label") not in LABELS:
            reasons.append(f"bad label {row.get('label')!r}")
        reasons += single_launch_reasons(row)
        text = json.dumps(row)
        reasons += unsafe_text_reasons(text)
        if reasons:
            row = {**row, "_reject_reasons": reasons}
            rejected.append(row)
        else:
            kept.append(row)
    return ValidationResult(kept=kept, rejected=rejected)


def dedupe_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Exact + near-duplicate filter (prefix signature). Keeps first occurrence, in order."""
    seen: set[str] = set()
    out = []
    dropped = 0
    for row in rows:
        sig = row_signature(row)
        if sig in seen:
            dropped += 1
            continue
        seen.add(sig)
        out.append(row)
    return out, dropped


def attach_facts(rows: list[dict], seed: int = 0) -> None:
    """Fill in the 9 launch-time facts in place, deterministically from `seed` + row index."""
    import random

    for i, row in enumerate(rows):
        key = f"{seed}\x00{i}\x00{row['exe']}\x00{' '.join(row['argv'])}"
        rng = random.Random(key)
        facts = synthesize_facts(row.get("fact_hint", "normal"), rng)
        facts["workdir"] = row["cwd"]
        row["facts"] = facts


# --------------------------------------------------------------------------------- QA (second pass)

QA_SYSTEM = (
    "You are a security triage analyst labelling a single process launch on a Linux workstation. "
    "You see only the launch itself, not its intended training label, and you know nothing about "
    "what the process does afterwards. Answer from the launch text alone."
)
# The canonical class descriptions: the blind pass, the generator and the hosted judge must
# read the classes the same way (gen_corpus_lib.CLASS_DESCRIPTIONS).
QA_CATEGORIES = CLASS_DESCRIPTIONS


def render_qa_text(row: dict) -> str:
    facts = row.get("facts", {})
    parts = [
        f"exe: {row['exe']}",
        f"argv: {' '.join(row['argv'])}",
        f"parent_exe: {row.get('parent_exe', '')}",
        f"parent_argv: {' '.join(row.get('parent_argv', []))}",
        f"cwd: {row.get('cwd', '')}",
        f"uid: {row.get('uid', '')}",
    ]
    parts += [f"{name}: {facts[name]}" for name in FACT_NAMES if name in facts]
    return "  ".join(parts)


def qa_prompt(row: dict) -> str:
    cats = "\n".join(f"- {k}: {v}" for k, v in QA_CATEGORIES.items())
    return (
        "Classify this single process launch into exactly one category:\n"
        f"{cats}\n\nLaunch:\n{render_qa_text(row)}\n\n"
        'Answer as JSON: {"label": "safe"|"warn"|"crit", "confidence": 0.0-1.0}'
    )


def qa_label(client: LLMClient, model: str, row: dict) -> tuple[str, float]:
    text = client.chat(model, qa_prompt(row), system=QA_SYSTEM, temperature=0.2, max_tokens=60)
    try:
        obj = extract_json(text)
        label = str(obj["label"])
        conf = float(obj.get("confidence", 0.5))
    except (ValueError, KeyError, TypeError):
        label, conf = "safe", 0.0
    if label not in LABELS:
        label = "safe"
    return label, max(0.0, min(1.0, conf))


def apply_qa(rows: list[dict], client: LLMClient, model: str, workers: int = 1) -> dict:
    """Mutates rows in place: adds `qa_label`, `qa_confidence`, `agrees`, `disputed`. Returns a
    summary dict (agreement rate per class, counts)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max(1, workers)) as ex:
        results = list(ex.map(lambda r: qa_label(client, model, r), rows))
    agree: Counter = Counter()
    total: Counter = Counter()
    for row, (label, conf) in zip(rows, results, strict=True):
        row["qa_label"] = label
        row["qa_confidence"] = conf
        row["agrees"] = label == row["label"]
        row["disputed"] = not row["agrees"]
        total[row["label"]] += 1
        if row["agrees"]:
            agree[row["label"]] += 1
    return {
        "agreement_overall": sum(agree.values()) / max(1, sum(total.values())),
        "agreement_per_class": {k: agree[k] / total[k] for k in total if total[k]},
        "disputed_count": sum(1 for r in rows if r["disputed"]),
        "rows": len(rows),
    }


# --------------------------------------------------------------------------------- rendering


def render_context(row: dict) -> tuple[str, str]:
    """(state text, signature) through the real `ContextBuilder`, from raw fields only."""
    import sys
    from pathlib import Path

    app_src = Path(__file__).resolve().parents[2] / "app" / "src"
    if str(app_src) not in sys.path:
        sys.path.insert(0, str(app_src))
    from jevmon.core import MemoryCounts
    from jevmon.detect.context import (
        ContextBuilder,
        ContextSettings,
        EnvironmentStats,
        ExecObservation,
    )
    from jevmon.detect.system_facts import StructuralFacts
    from jevmon.events import Event
    from jevmon.redact import redact_argv_text

    facts = row.get("facts", {})
    exe_file = facts.get("exe_file", "")
    struct = StructuralFacts(
        readable=True,
        writable_dir="world_writable=yes" in exe_file or "tmpfs" in str(row["cwd"]),
        owned_by_nonroot=row["uid"] != 0,
        recently_written=facts.get("exe_state") == "deleted",
        mount_type="tmpfs" if "world_writable=yes" in exe_file else "overlay",
        on_tmpfs_or_noexec="world_writable=yes" in exe_file,
    )

    class OneShotFacts:
        def inspect(self, exe: str, now: float, recent_seconds: float) -> StructuralFacts:
            return struct

    builder = ContextBuilder(
        MemoryCounts(),
        OneShotFacts(),
        ContextSettings(order=("header", "exe", "template", "argv") + FACT_NAMES),
    )
    argv = row["argv"]
    ev = Event(
        time=1_700_000_000.0,
        source="exec",
        subject="1",
        kind="exec",
        severity="safe",
        message="exec",
        fields={
            "pid": 1,
            "uid": row["uid"],
            "exe": redact_argv_text([row["exe"]]),
            "path0": redact_argv_text([argv[0]]) if argv else "",
            "argv0": redact_argv_text([argv[0]]).split()[0] if argv else "",
            "argv": redact_argv_text(argv),
            "cwd": redact_argv_text([row["cwd"]]).strip(),
            "cgroup": row.get("cgroup", "user.slice"),
            "parent_comm": "",
            "parent_exe": row["parent_exe"],
            "parents": row["parent_exe"],
            "ppid": 2,
            "parent_argv": redact_argv_text(row.get("parent_argv", [])),
            **{f"fact_{name}": facts[name] for name in FACT_NAMES if name in facts},
        },
    )
    obs = ExecObservation.from_event(ev)
    ctx = builder.observe_exec(obs, EnvironmentStats())
    return ctx.state, ctx.signature


def render_rows(rows: list[dict]) -> None:
    """Mutates rows in place: adds `context` and `signature`."""
    for row in rows:
        state, sig = render_context(row)
        row["context"] = state
        row["signature"] = sig

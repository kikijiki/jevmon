"""Convert tomngdev/shell-safety-v1.1 (allow / ask / deny shell commands) to the launch schema.

    python lab/scripts/ingest_shell_safety.py --src data/external/hf-shell-safety-v1.1 \
        --out lab-data-ext-shell-safety

Posix commands only. The source `test` split is the external test (role `test`); `train` and
`validation` give an augmentation-shaped file (role `aug_train`) that is never used by any
evaluation here, with templates seen in the test or in the lab sets removed. `session_context`
(git state, last prompt) is dropped: a launch does not carry it. Labels come from
`lab/ingest/shell_safety.toml`. Nothing is executed.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from jevmon_lab import ingest


def read_split(src: Path, name: str, table: dict, role: str, shell: dict) -> tuple[list, dict]:
    stats = {"rows": 0, "unique_commands": 0, "dropped_shell": 0, "label_conflict": 0}
    by_cmd: dict[str, list[dict]] = defaultdict(list)
    for line in (src / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        stats["rows"] += 1
        by_cmd[rec["command"]].append(rec)
    rows: list[ingest.Row] = []
    for command, recs in by_cmd.items():
        targets = {ingest.apply_rules(table["rule"], rec)[0] for rec in recs}
        if "drop" in targets and len(targets) == 1:
            stats["dropped_shell"] += 1
            continue
        if len(targets) > 1:  # the same command text under two labels: context decided it
            stats["label_conflict"] += 1
            continue
        stats["unique_commands"] += 1
        rec = recs[0]
        target, rule, why = ingest.apply_rules(table["rule"], rec)
        rows += ingest.command_rows(
            command,
            source=table["source"],
            role=role,
            label=None if target == "null" else target,
            rule=rule,
            uncertain=why,
            family=str(rec[table["family_field"]]),
            source_label={k: rec[k] for k in ("label", "category", "shell", "reason")},
            driver=DRIVER,
            shell=shell,
            extra={"split": name},
        )
    return rows, stats


DRIVER = {"exe": "/usr/bin/dash", "argv0": "sh"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lab", type=Path, nargs="*", default=None, help="lab sources for dedupe")
    args = ap.parse_args()
    table = ingest.load_toml("shell_safety")
    shell = ingest.load_toml("shell")
    lab = ingest.load_lab_keys(args.lab if args.lab is not None else ingest.DEFAULT_LAB_SOURCES)

    test_rows, st_test = read_split(args.src, "test", table, "test", shell)
    test_rows, dd_test = ingest.dedupe(test_rows, forbidden=lab)
    test_keys = {r.key for r in test_rows}

    aug: list[ingest.Row] = []
    st_aug = {}
    for name in ("train", "validation"):
        rows, st = read_split(args.src, name, table, "aug_train", shell)
        aug += rows
        st_aug[name] = st
    aug, dd_aug = ingest.dedupe(aug, forbidden=lab | test_keys)

    out = ingest.render_all(test_rows + aug)
    meta = {
        "source": table["source"],
        "mapping": "lab/ingest/shell_safety.toml",
        "lab_keys": len(lab),
        "test": {"read": st_test, "dedupe": dd_test},
        "aug_train": {"read": st_aug, "dedupe": dd_aug},
        "summary": ingest.summarize(out),
        "note": "text-only external test; aug_train rows are shaped for augmentation and never used",
    }
    ingest.write_dataset(args.out, out, meta)
    print(json.dumps(meta["summary"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

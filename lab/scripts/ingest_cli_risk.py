"""Convert yannelli/cli-command-risk (risk_level + risk_categories on shell commands) to the
launch schema.

    python lab/scripts/ingest_cli_risk.py --src data/external/hf-cli-command-risk \
        --out lab-data-ext-cli-risk

The source `test` split is the external test; `train` and `validation` give an augmentation-shaped
file never used here. The source repeats each command once per shell flavour: rows are collapsed to
unique command text first. Categories that describe self-inflicted damage are dropped by the
mapping table (`lab/ingest/cli_risk.toml`), not labelled. Nothing is executed.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from jevmon_lab import ingest

DRIVER = {"exe": "/usr/bin/dash", "argv0": "sh"}


def read_split(src: Path, name: str, table: dict, role: str, shell: dict) -> tuple[list, dict]:
    stats = {"rows": 0, "unique_commands": 0, "dropped": 0, "label_conflict": 0}
    by_cmd: dict[str, list[dict]] = defaultdict(list)
    for rec in pq.read_table(src / "data" / f"{name}.parquet").to_pylist():
        stats["rows"] += 1
        by_cmd[rec["command"]].append(rec)
    rows: list[ingest.Row] = []
    for command, recs in by_cmd.items():
        results = {ingest.apply_rules(table["rule"], rec)[0] for rec in recs}
        if results == {"drop"}:
            stats["dropped"] += 1
            continue
        if len(results) > 1:
            stats["label_conflict"] += 1
            continue
        rec = recs[0]
        target, rule, why = ingest.apply_rules(table["rule"], rec)
        stats["unique_commands"] += 1
        family = next((rec[f] for f in table["family_fields"] if rec[f]), "unknown")
        rows += ingest.command_rows(
            command,
            source=table["source"],
            role=role,
            label=None if target == "null" else target,
            rule=rule,
            uncertain=why,
            family=str(family),
            source_label={
                "risk_level": rec["risk_level"],
                "risk_categories": list(rec["risk_categories"] or []),
            },
            driver=DRIVER,
            shell=shell,
            extra={"split": name},
        )
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lab", type=Path, nargs="*", default=None, help="lab sources for dedupe")
    args = ap.parse_args()
    table = ingest.load_toml("cli_risk")
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
        "mapping": "lab/ingest/cli_risk.toml",
        "lab_keys": len(lab),
        "test": {"read": st_test, "dedupe": dd_test},
        "aug_train": {"read": st_aug, "dedupe": dd_aug},
        "summary": ingest.summarize(out),
        "note": "machine-generated labels on the harm axis; aug_train rows are never used",
    }
    ingest.write_dataset(args.out, out, meta)
    print(json.dumps(meta["summary"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

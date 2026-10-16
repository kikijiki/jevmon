"""Re-map a captured four-class lab run onto the three single-launch classes.

    python lab/scripts/relabel_three_class.py ../jevmon-improve/lab-data-r6 lab-data-3c-r6

Captured data keeps the old ground truth (routine/unusual/suspicious/runaway). This writes a
sibling directory with `dataset.jsonl`, `contexts.jsonl` and `scenarios.jsonl` relabelled:
routine -> safe, unusual -> warn, suspicious -> crit. `runaway` is not relabelled: those rows and
scenarios get `label = null` and an explicit `excluded` reason, so they drop out of the labelled
export and every score (a rate or resource spike cannot be seen in a single launch). Rows keep
their order (contexts.jsonl stays aligned) and the family -> split rule is untouched: `split` is
recomputed from the family hash exactly as before.
"""

import argparse
import json
import shutil
from pathlib import Path

from jevmon_lab.manifest import split_of

MAP = {
    "routine": "safe",
    "unusual": "warn",
    "unusual-but-benign": "warn",
    "suspicious": "crit",
}
EXCLUDED = {"runaway": "runaway: a rate or resource spike cannot be seen in a single launch"}


def relabel(value: str | None) -> tuple[str | None, str | None]:
    """(new label, excluded reason) for an old label."""
    if value in MAP:
        return MAP[value], None
    if value in EXCLUDED:
        return None, EXCLUDED[value]
    return None, None


def convert(src: Path, dst: Path) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name in ("dataset.jsonl", "scenarios.jsonl"):
        path = src / name
        if not path.exists():
            continue
        out = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("label", "scenario_label"):
                if key in row:
                    new, why = relabel(row[key])
                    row[key] = new
                    if why:
                        row["excluded"] = why
            if row.get("family"):
                row["split"] = split_of(row["family"])
            if name == "dataset.jsonl" and row.get("label"):
                counts[row["label"]] = counts.get(row["label"], 0) + 1
            out.append(json.dumps(row, ensure_ascii=False))
        (dst / name).write_text("\n".join(out) + "\n")
    for name in ("contexts.jsonl", "meta.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    counts = convert(args.src, args.dst)
    print(f"{args.dst}: labelled rows {counts}")


if __name__ == "__main__":
    main()

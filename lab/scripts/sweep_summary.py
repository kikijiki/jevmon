"""Re-plot the train-CV sweep ranked per method, as Trackio project `jevmon-sweep3`.

    python lab/scripts/sweep_summary.py <runs dir> [--db other/jevmon-3class.db ...] [--prefix s12-]

Reads `cv/unique_f1` (and its sd) of the `<prefix>*` runs that `improve_sweep.py` wrote to the
`jevmon-3class` store and logs one run per method, step = rank by cross-validated macro-F1.
"""

import argparse
import json
import sqlite3
from pathlib import Path

from jevmon_lab.runlog import RunLog

ap = argparse.ArgumentParser()
ap.add_argument("runs", type=Path)
ap.add_argument("--prefix", default="s12-")
ap.add_argument("--source", default="jevmon-3class")
ap.add_argument("--db", type=Path, action="append", default=[], help="extra Trackio .db files")
ap.add_argument("--project", default="jevmon-sweep3")
args = ap.parse_args()

paths = [args.runs / f"{args.source}.db", *args.db]
groups: dict[str, list[tuple[float, float]]] = {}
entries = []
for path in paths:
    entries += sqlite3.connect(path).execute("select run_name, metrics from metrics").fetchall()
for name, raw in entries:
    if not name.startswith(args.prefix):
        continue
    m = json.loads(raw)
    if "cv/unique_f1" not in m:
        continue
    if "-ft-lora" in name:
        g = "lora"
    elif "-ft-full" in name:
        g = "full fine-tune"
    elif "logreg" in name:
        g = "head: logistic"
    elif "mlp" in name:
        g = "head: mlp"
    else:
        g = "head: knn / prototype"
    groups.setdefault(g, []).append((m["cv/unique_f1"], m.get("cv/unique_f1_sd", 0.0)))

for g, vals in groups.items():
    vals.sort(reverse=True)
    run = RunLog.start(
        args.project, f"cv-sweep {g}", {"group": g, "n": len(vals)}, directory=args.runs
    )
    for i, (f1, sd) in enumerate(vals):
        run.log(i, **{"sweep/cv_macro_f1": f1, "sweep/cv_macro_f1_sd": sd})
    run.finish()
print({g: len(v) for g, v in groups.items()})

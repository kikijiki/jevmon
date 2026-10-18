"""Learning-curve figure: macro-F1 against the number of training families.

The training pool (original lab train + E19b consensus train + shell-safety and cli-command-risk
own train rows) is grouped by family. For each fraction of families (5% to 100%) a stratified
subset of families is drawn, the fixed recipe of each shipped head is fitted on the rows of those
families only, and the head is scored on the held-out sets. Nothing held-out is ever fitted or
selected on; the 100% point is the shipped `lab-data-e20/results.json` number, which the script
checks and prints.

    uv run --project lab --extra plots python lab/scripts/learning_curve.py

Writes `docs/screenshots/learning.png` (2x3 small multiples, x = training families on a log scale,
y = macro-F1, one line per head with a band over seeds) and `lab-data-e20/learning_curve.json`.
CPU only, a few minutes: the heads are linear classifiers over cached frozen embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import train_heads as th
from jevmon_lab.manifest import LABELS

FRACTIONS = (0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
SEEDS = 8
SEED_BASE = 20261020
OUT_PNG = ROOT / "docs/screenshots/learning.png"
OUT_JSON = ROOT / "lab-data-e20/learning_curve.json"

# One panel per held-out set. `kind` names how the metric reads (QuasarNix is crit only).
PANELS = (
    ("lab/pool", "lab held-out pool", "macro-F1"),
    ("gen2/heldout", "E19b held-out", "macro-F1"),
    ("shell-safety-v1.1/driver", "shell-safety driver", "macro-F1"),
    ("cli-command-risk/driver", "cli-risk driver", "macro-F1"),
    ("quasarnix/driver", "QuasarNix (recall)", "recall"),
    ("splunk/reviewed", "Splunk reviewed", "macro-F1"),
)

HEADS_ORDER = ("laya", "ettin", "qwen3e")
HEADS_NAME = {"laya": "Laya", "ettin": "Ettin-1B", "qwen3e": "Qwen3-Embedding-8B"}
COLORS = {"laya": "#0072B2", "ettin": "#009E73", "qwen3e": "#D55E00"}


# ------------------------------------------------------------------ subsampling


def families_by_source(rows: list[th.Row]) -> dict[str, list[str]]:
    """Source name -> sorted family names, one entry per family."""
    seen: dict[str, set[str]] = {}
    for r in rows:
        seen.setdefault(r.source, set()).add(r.family)
    return {src: sorted(fams) for src, fams in sorted(seen.items())}


def sample_families(by_source: dict[str, list[str]], frac: float, rng) -> list[str]:
    """A `frac` of each source's families (at least one, sorted unique).

    Stratifying by source keeps the pool's four sources represented at every fraction; drawing
    within a source keeps whole families together.
    """
    chosen: list[str] = []
    for src in sorted(by_source):
        fams = by_source[src]
        k = min(len(fams), max(1, round(frac * len(fams))))
        idx = rng.choice(len(fams), size=k, replace=False)
        chosen.extend(fams[int(i)] for i in idx)
    return sorted(set(chosen))


def draw(by_source, frac, seed, labels_of, fams):
    """A subset of family names whose rows contain all three classes (resamples if needed)."""
    rng = np.random.default_rng((SEED_BASE + seed, round(frac * 1000)))
    chosen = sample_families(by_source, frac, rng)
    for _ in range(100):
        if len({int(v) for v in labels_of[np.isin(fams, chosen)]}) == len(LABELS):
            return chosen
        chosen = sample_families(by_source, frac, rng)
    raise SystemExit(f"could not draw all three classes at fraction {frac}")


# ------------------------------------------------------------------ fit + score


def build_plan(train: list[th.Row]) -> list[dict]:
    """Sampling plan shared by the three heads: (fraction, families, row indices) per seed."""
    fams = np.asarray([r.family for r in train])
    labels_of = np.asarray([LABELS.index(r.label) for r in train])
    by_source = families_by_source(train)
    plan = []
    for frac in FRACTIONS:
        seeds = [0] if frac == 1.0 else range(SEEDS)
        entries = []
        for seed in seeds:
            chosen = draw(by_source, frac, seed, labels_of, fams)
            idx = np.flatnonzero(np.isin(fams, chosen))
            entries.append((len(chosen), idx))
        plan.append({"fraction": frac, "entries": entries})
    return plan


def run(seeds: int = SEEDS) -> dict:
    train = th.training_rows()
    groups = {g.name: g for g in th.eval_groups()}
    panels = [(name, title, groups[name]) for name, title, _ in PANELS]

    train_texts = [r.text for r in train]
    y_train = np.asarray([LABELS.index(r.label) for r in train])
    key_train = np.asarray([f"{r.text}\x00{r.label}" for r in train])

    eval_texts, panel_slices = [], []
    offset = 0
    for name, title, g in panels:
        y = np.asarray([LABELS.index(r.label) for r in g.rows])
        fam = np.asarray([r.family for r in g.rows])
        panel_slices.append((name, slice(offset, offset + len(g.rows)), y, fam))
        eval_texts += [r.text for r in g.rows]
        offset += len(g.rows)

    plan = build_plan(train)
    points: dict[str, list[dict]] = {h: [] for h in HEADS_ORDER}
    for name in HEADS_ORDER:
        parts, spec, _config, _backbone = th.HEADS[name]
        x_train = th.matrix(parts, train_texts)
        x_eval = th.matrix(parts, eval_texts)
        for step in plan:
            values = {p: [] for p, _, _, _ in panel_slices}
            for _nfam, idx in step["entries"]:
                probs = th.fit_predict(spec, x_train[idx], y_train[idx], key_train[idx], x_eval)
                pred = probs.argmax(1)
                for pname, sl, y, fam in panel_slices:
                    values[pname].append(th.score_point(y, pred[sl]))
            families = float(np.mean([n for n, _ in step["entries"]]))
            points[name].append(
                {"fraction": step["fraction"], "families": families, "values": values}
            )
            print(
                f"  {name:7s} frac {step['fraction']:.2f} "
                f"families {families:.0f} "
                + " ".join(f"{p.split('/')[0]}:{np.mean(v):.3f}" for p, v in values.items()),
                flush=True,
            )
    return {"panels": [list(p) for p in PANELS], "seeds": seeds, "points": points}


# ------------------------------------------------------------------ verification


def reference_values() -> dict[str, dict[str, float]]:
    """The shipped numbers from lab-data-e20/results.json, keyed by panel and head."""
    results = json.loads((ROOT / "lab-data-e20/results.json").read_text())
    out = {}
    for name, _title, _kind in PANELS:
        entry = results["sets"][name]["judges"]
        out[name] = {h: float(entry[th.head_key(h)]["macro_f1"]) for h in HEADS_ORDER}
    return out


def check_full(curve: dict, ref: dict) -> bool:
    """Print and assert the 100% point equals the shipped results.json number."""
    ok = True
    print("\n100% point vs lab-data-e20/results.json:")
    for name, _title, _kind in PANELS:
        for h in HEADS_ORDER:
            got = next(p["values"][name][0] for p in curve["points"][h] if p["fraction"] == 1.0)
            want = ref[name][h]
            good = abs(got - want) < 1e-9
            ok &= good
            mark = "ok" if good else "MISMATCH"
            print(f"  {name:28s} {h:7s} {got:.9f} vs {want:.9f}  {mark}")
    return ok


# ------------------------------------------------------------------ figure


def plot(curve: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), dpi=100, facecolor="white")
    xs = [p["families"] for p in curve["points"][HEADS_ORDER[0]]]
    ticks = [round(x) for x in xs]
    for ax, (name, title, kind) in zip(axes.ravel(), PANELS):
        lo, hi = 1.0, 0.0
        for h in HEADS_ORDER:
            steps = curve["points"][h]
            mean = [float(np.mean(s["values"][name])) for s in steps]
            low = [float(np.min(s["values"][name])) for s in steps]
            high = [float(np.max(s["values"][name])) for s in steps]
            ax.fill_between(xs, low, high, color=COLORS[h], alpha=0.16, linewidth=0)
            ax.plot(
                xs, mean, "-o", color=COLORS[h], markersize=3.4, linewidth=1.5, label=HEADS_NAME[h]
            )
            lo, hi = min(lo, *low), max(hi, *high)
        ax.set_title(f"{title}\n(y = {kind})" if kind == "recall" else title, fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=8.5)
        ax.minorticks_off()
        pad = max(0.04, 0.08 * (hi - lo))
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
        ax.grid(True, which="major", color="#000000", alpha=0.07, linewidth=0.6)
        ax.tick_params(labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("macro-F1", fontsize=9.5)
    for ax in axes[1, :]:
        ax.set_xlabel("training families", fontsize=9.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, facecolor="white")
    print(f"\nwrote {OUT_PNG}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--plot-only", action="store_true", help="reuse lab-data-e20/learning_curve.json"
    )
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if args.plot_only:
        curve = json.loads(OUT_JSON.read_text())
    else:
        curve = run()
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(curve, indent=1))
        if not check_full(curve, reference_values()):
            raise SystemExit("the 100% point does not match lab-data-e20/results.json")
    plot(curve)


if __name__ == "__main__":
    main()

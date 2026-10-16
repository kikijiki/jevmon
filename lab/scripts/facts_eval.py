"""E17 (launch-time facts) and E16 (environment): stage 1, stage 2 and the held-out score.

    python scripts/facts_eval.py show
    python scripts/facts_eval.py stage1 [--which e17|e16]
    python scripts/facts_eval.py cv     [--which e17|e16]      # stage 2, train-CV only
    python scripts/facts_eval.py score  --cells N,ALL,...      # stage 2, held-out once

All cells are rendered from the same captured rows (`lab-data-fx-*`), so they are paired. A cell is
the base format (E15 winner A: header, exe, template, argv) plus some facts and/or an environment
segment. Sets: train (old train families + the facts train side), old, re, fresh, facts (facts
manifest, held-out side), P1 = old + re + fresh, pool = P1 + facts. Wording, recipe and the
decision rule are pre-registered in the research log (E16/E17).
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from backbone_eval import macro_f1
from ctx_eval import RED_LABEL  # noqa: F401  (label wording lives in the app)
from improve_lib import EmbeddingCache, Spec, cv_scores, fit_predict, laya_agent, make_data, to_rows
from jevmon.detect.context import (
    ContextSettings,
    order_env,
    parse_exec_state,
    render_segments,
)
from jevmon.sanitize import clean
from jevmon.sources.facts import FACT_NAMES
from jevmon_lab import metrics
from jevmon_lab.manifest import LABELS, split_of
from jevmon_lab.runlog import RunLog

BASE = ("header", "exe", "template", "argv")
SPEC = Spec("logreg", wd=0.01, balanced=True)
BOOT = 10000
SEED = 14
CACHE = Path("cache")
CAPS = (1000, 3000)
CONTAINER = ("namespaces", "cgroup_path", "privileges")
LAUNCH = ("stdio", "session", "workdir", "exe_state", "argv0_vs_exe")
PRIORITY = ContextSettings().env_priority


# -------------------------------------------------------------------------------- data


def load_dir(path: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in (path / "dataset.jsonl").read_text().splitlines() if ln]
    raw = [json.loads(ln) for ln in (path / "contexts.jsonl").read_text().splitlines() if ln]
    assert len(rows) == len(raw), path
    out = []
    for row, c in zip(rows, raw, strict=True):
        if not row.get("label"):
            continue
        row["_state"] = c["context"]
        row["_fields"] = c.get("fields") or {}
        row["_env"] = c.get("env_raw", "")
        row["_env_parent"] = c.get("env_parent_diff", "")
        row["_env_base"] = c.get("env_base_diff", "")
        row["split"] = split_of(row["family"])
        out.append(row)
    return out


def load(root: Path):
    rows = []
    train_fams: set[str] = set()
    for r in load_dir(root / "lab-data-fx-train"):
        if r["split"] == "train":
            r["cap"] = "train"
            rows.append(r)
    for name in ("lab-data-fx-facts", "lab-data-fx-facts2"):
        if not (root / name).exists():
            continue
        for r in load_dir(root / name):
            r["cap"] = "train" if r["split"] == "train" else "facts"
            rows.append(r)
    train_fams = {r["family"] for r in rows if r["cap"] == "train"}
    for cap in ("old", "re", "fresh"):
        for r in load_dir(root / f"lab-data-fx-{cap}"):
            if r["split"] != "heldout":
                continue
            assert r["family"] not in train_fams, r["family"]
            r["cap"] = cap
            rows.append(r)
    data, _ = make_data(rows, lambda r: r["_state"])
    data.extra["cap"] = np.asarray([r["cap"] for r in rows])
    # a fixed permutation inside each set drives the SHUF controls
    rng = np.random.default_rng(SEED)
    src = np.arange(len(rows))
    for cap in set(data.extra["cap"]):
        idx = np.flatnonzero(data.extra["cap"] == cap)
        src[idx] = rng.permutation(idx)
    data.extra["shuf"] = src
    return data


def sets_of(data):
    cap = data.extra["cap"]
    p1 = np.isin(cap, ("old", "re", "fresh"))
    return {
        "train": np.flatnonzero(cap == "train"),
        "old": np.flatnonzero(cap == "old"),
        "re": np.flatnonzero(cap == "re"),
        "fresh": np.flatnonzero(cap == "fresh"),
        "facts": np.flatnonzero(cap == "facts"),
        "P1": np.flatnonzero(p1),
        "pool": np.flatnonzero(cap != "train"),
    }


# -------------------------------------------------------------------------------- cells


def parse_cell(name: str) -> dict:
    """`N`, `ALL`, `f:a+b`, `SHUF`, `ENV:FS:1000`, `ALL+ENV:FS:3000`, `SHUF+ENV:FS:1000`."""
    cell = {"facts": (), "env": None, "cap": 1000, "shuf": False}
    for part in name.split("+"):
        if part == "N":
            continue
        if part == "ALL":
            cell["facts"] = FACT_NAMES
        elif part == "SHUF":
            cell["facts"] = FACT_NAMES
            cell["shuf"] = True
        elif part == "CONTAINER":
            cell["facts"] = CONTAINER
        elif part == "LAUNCH":
            cell["facts"] = LAUNCH
        elif part.startswith("f:"):
            cell["facts"] = tuple(n for n in FACT_NAMES if n in part[2:].split(","))
        elif part.startswith("ENV:"):
            _, kind, cap = part.split(":")
            cell["env"], cell["cap"] = kind, int(cap)
        elif part == "ENVSHUF":
            cell["shuf_env"] = True
        else:
            raise ValueError(part)
    cell["facts"] = tuple(n for n in FACT_NAMES if n in cell["facts"])
    return cell


def env_items(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.split("\n"):
        key, sep, value = line.partition("=")
        if sep and key:
            out.append((key, value))
    return out


def render(data, i: int, cell: dict) -> str:
    row = data.rows[i]
    src = data.rows[int(data.extra["shuf"][i])] if cell["shuf"] else row
    segs = [kv for kv in parse_exec_state(row["_state"]) if kv[0] in BASE]
    for name in cell["facts"]:
        value = src["_fields"].get(f"fact_{name}")
        if value is not None:
            segs.append((name, f"{name}: {clean(str(value), 300)}"))
    order = BASE + tuple(n for n in cell["facts"])
    text = render_segments(segs, ContextSettings(order=order, model_max=cell["cap"]))
    if cell["env"]:
        esrc = data.rows[int(data.extra["shuf"][i])] if cell.get("shuf_env") else row
        raw = {"F": "_env", "FS": "_env", "D": "_env_parent", "B": "_env_base"}[cell["env"]]
        items = env_items(esrc[raw])
        if cell["env"] != "F":
            items = order_env(items, PRIORITY)
        body = " ".join(f"{k}={clean(v, 200)}" for k, v in items) or "-"
        text = f"{text} environ: {body}"
        text = text[: cell["cap"]]  # saturating: what comes first survives
    return text


def cell_texts(data, name: str) -> list[str]:
    cell = parse_cell(name)
    return [render(data, i, cell) for i in range(len(data.rows))]


def keyed(data, texts):
    return type(data)(
        rows=data.rows,
        split=data.split,
        family=data.family,
        y=data.y,
        key=np.asarray([f"{t}\x00{r['label']}" for t, r in zip(texts, data.rows, strict=True)]),
        extra=data.extra,
    )


E17_STAGE1 = ["N", "ALL", "SHUF", *[f"f:{n}" for n in FACT_NAMES]]
E16_STAGE1 = [
    "N",
    "ENV:F:1000",
    "ENV:F:3000",
    "ENV:FS:1000",
    "ENV:FS:3000",
    "ENV:D:1000",
    "ENV:B:1000",
    "ENV:D:3000",
    "ENV:B:3000",
    "ENVSHUF+ENV:FS:1000",
]


def features(texts, max_length):
    return EmbeddingCache(laya_agent).get(texts, "mean", max_length)


def max_len(name: str) -> int:
    return 1024 if parse_cell(name)["cap"] > 1000 else 512


# -------------------------------------------------------------------------------- stage 1


def cmd_stage1(args):
    from dataclasses import replace as _replace

    from backbone_embed import embed
    from jevmon.config import load_config
    from jevmon.laya.head import TunedHead, resolve_head
    from jevmon.monitor import build_decider
    from settle_eval import RECIPES

    data = load(args.root)
    sets = sets_of(data)
    names = E17_STAGE1 if args.which == "e17" else E16_STAGE1
    out: dict[str, dict] = {}

    def record(model, cell, pred):
        for s, idx in sets.items():
            out.setdefault(model, {}).setdefault(cell, {})[s] = metrics.score(
                to_rows(data, idx, pred[idx])
            )["macro_f1"]

    texts = {n: cell_texts(data, n) for n in names}
    head = TunedHead.load(resolve_head("lab-head-v4"))
    for n in names:
        record("shipped head v4", n, head.probabilities(features(texts[n], max_len(n))).argmax(1))
    cfg = load_config()
    cfg = _replace(cfg, judge=_replace(cfg.judge, tuned=""))
    decider = build_decider(cfg)
    decider.load()
    for n in names:
        labels = []
        for s in range(0, len(texts[n]), 32):
            labels += [
                d.label if d.available else "safe"
                for d in decider.decide_batch(texts[n][s : s + 32])
            ]
        record("zero-shot laya", n, np.asarray([LABELS.index(x) for x in labels]))
    # E14 backbone heads fitted on the old format (old train rows of the E15 capture, text A)
    base = [render(data, i, parse_cell("N")) for i in range(len(data.rows))]
    tr = sets["train"]
    pick = [
        n for n in names if n in ("N", "ALL", "SHUF", "ENV:FS:1000", "ENV:D:1000", "ENV:B:1000")
    ]
    for name, (src, var, spec, model_id, kind, _role) in RECIPES.items():
        if name in ("laya", "laya-wd03"):
            continue
        store = pickle.loads(Path(f"cache/bb-{src}-512.pkl").read_bytes())
        old_idx = [i for i in tr if base[i] in store]
        if len(old_idx) < 50:
            raise SystemExit(f"old-format cache covers only {len(old_idx)} train rows")
        xtr = np.stack([store[base[i]][var] for i in old_idx]).astype(np.float32)
        ytr = data.y[old_idx]
        ktr = data.key[old_idx]
        for n in pick:
            uniq = sorted(set(texts[n]))
            vec = embed(src, model_id, uniq, kind=kind, max_length=max_len(n), device="cuda")[var]
            ix = {t: j for j, t in enumerate(uniq)}
            xte = np.stack([vec[ix[t]] for t in texts[n]]).astype(np.float32)
            record(f"{name} (old-format head)", n, fit_predict(spec, xtr, ytr, ktr, xte).argmax(1))
    path = Path(f"cache/{args.which}-stage1.json")
    path.write_text(json.dumps(out, indent=1))
    for model, per in out.items():
        for n, row in per.items():
            base_row = per["N"]
            cells = " ".join(
                f"{s} {row[s]:.3f}({row[s] - base_row[s]:+.3f})"
                for s in ("train", "old", "re", "fresh", "facts", "P1")
            )
            print(f"{model:34s} {n:22s} {cells}")
        print()


# -------------------------------------------------------------------------------- stage 2


def cell_x(data, name):
    return features(cell_texts(data, name), max_len(name))


def cmd_cv(args):
    data = load(args.root)
    tr = sets_of(data)["train"]
    names = [c for c in args.cells.split(";")] if args.cells else None
    results: dict[str, dict] = {}

    def cv(name):
        if name in results:
            return results[name]
        d = keyed(data, cell_texts(data, name))
        sc = cv_scores(SPEC, cell_x(data, name), d, tr, folds=4, repeats=5, base_seed=0)
        results[name] = sc
        print(f"{name:40s} {sc['unique_f1']:.3f} +- {sc['unique_f1_sd']:.3f}", flush=True)
        return sc

    if names:
        for n in names:
            cv(n)
    else:
        fixed = ["N", "ALL", "SHUF", "CONTAINER", "LAUNCH"]
        singles = [f"f:{n}" for n in FACT_NAMES]
        for n in [*fixed, *singles]:
            cv(n)
        top4 = sorted(FACT_NAMES, key=lambda n: -results[f"f:{n}"]["unique_f1"])[:4]
        for i, a in enumerate(top4):
            for b in top4[i + 1 :]:
                cv("f:" + ",".join(sorted([a, b])))
        # greedy forward selection by train-CV, stop when the gain is below 1 sd
        chosen: list[str] = []
        best = results["N"]["unique_f1"]
        sd = results["N"]["unique_f1_sd"]
        while len(chosen) < len(FACT_NAMES):
            trial = {n: cv("f:" + ",".join(chosen + [n])) for n in FACT_NAMES if n not in chosen}
            n, sc = max(trial.items(), key=lambda kv: kv[1]["unique_f1"])
            if sc["unique_f1"] - best < sd:
                break
            chosen.append(n)
            best, sd = sc["unique_f1"], sc["unique_f1_sd"]
        results["__greedy__"] = {"chosen": chosen, "f1": best}
        print("greedy subset:", chosen, round(best, 3))
        subset = chosen or [max(FACT_NAMES, key=lambda n: results[f"f:{n}"]["unique_f1"])]
        for n in subset:
            cv("f:" + ",".join(m for m in subset if m != n)) if len(subset) > 1 else None
    Path(f"cache/{args.which}-cv.json").write_text(json.dumps(results, indent=1))
    print("cells tried:", len([k for k in results if not k.startswith("__")]))


def cmd_score(args):
    final = Path(f"cache/{args.which}-final.json")
    if final.exists() and not args.force:
        raise SystemExit(f"{final} exists: held-out is scored once (--force)")
    data = load(args.root)
    sets = sets_of(data)
    names = args.cells.split(";")
    if names[0] != "N":
        names = ["N", *names]
    tr = sets["train"]
    preds = {}
    for n in names:
        d = keyed(data, cell_texts(data, n))
        x = cell_x(data, n)
        preds[n] = fit_predict(SPEC, x[tr], d.y[tr], d.key[tr], x).argmax(1)
    cv_path = Path(f"cache/{args.which}-cv.json")
    cv = json.loads(cv_path.read_text()) if cv_path.exists() else {}
    table = {}
    for s, idx in sets.items():
        if s == "train":
            continue
        rng = np.random.default_rng(SEED)
        fam = data.family[idx]
        fams = sorted(set(fam))
        members = [np.flatnonzero(fam == f) for f in fams]
        draws = [
            np.concatenate([members[i] for i in rng.integers(0, len(fams), len(fams))])
            for _ in range(BOOT)
        ]
        y = data.y[idx]
        vals = {n: np.asarray([macro_f1(y[d], preds[n][idx][d]) for d in draws]) for n in names}
        table[s] = {}
        for n in names:
            sc = metrics.score(to_rows(data, idx, preds[n][idx]))
            m = data.y[idx] == LABELS.index("safe")
            e = {
                "macro_f1": sc["macro_f1"],
                "rows": len(idx),
                "families": len(set(fam)),
                "safe_recall": float((preds[n][idx][m] == LABELS.index("safe")).mean()),
                "per_class": {k: [v["precision"], v["recall"]] for k, v in sc["per_class"].items()},
                "lo": float(np.percentile(vals[n], 2.5)),
                "hi": float(np.percentile(vals[n], 97.5)),
            }
            if n != "N":
                dlt = vals[n] - vals["N"]
                e |= {
                    "diff_lo": float(np.percentile(dlt, 2.5)),
                    "diff_hi": float(np.percentile(dlt, 97.5)),
                    "p_better": float((dlt > 0).mean()),
                }
            table[s][n] = e
    decision = {}
    for n in names:
        if n == "N":
            continue
        t = table
        cv_ok = (cv.get(n, {}).get("unique_f1", 0) - cv.get("N", {}).get("unique_f1", 0)) >= -0.03
        a = (
            t["facts"][n]["diff_lo"] > 0
            and t["facts"][n]["macro_f1"] > t["facts"].get("SHUF", {"macro_f1": -1})["macro_f1"]
            and t["P1"][n]["macro_f1"] - t["P1"]["N"]["macro_f1"] >= -0.02
            and t["pool"]["N"]["safe_recall"] - t["pool"][n]["safe_recall"] <= 0.03
            and cv_ok
        )
        b = t["pool"][n]["diff_lo"] > 0
        decision[n] = {"rule_a": bool(a), "rule_b": bool(b), "adopt": bool(a or b)}
    final.write_text(json.dumps({"table": table, "cv": cv, "decision": decision}, indent=1))
    for s, per in table.items():
        for n, e in per.items():
            d = (
                f" diff {e['macro_f1'] - per['N']['macro_f1']:+.3f} "
                f"[{e['diff_lo']:+.3f},{e['diff_hi']:+.3f}] P{e['p_better']:.2f}"
                if "diff_lo" in e
                else ""
            )
            print(
                f"{s:6s} {n:28s} F1 {e['macro_f1']:.3f} [{e['lo']:.2f},{e['hi']:.2f}]{d} "
                f"safeR {e['safe_recall']:.2f}"
            )
    print(json.dumps(decision, indent=1))
    for n in names:
        run = RunLog.start(
            args.project,
            f"{args.which}-final-{n.replace(':', '_')}-{time.strftime('%H%M%S')}",
            {"cell": n, "recipe": SPEC.name(), "experiment": args.which},
            directory=args.runs,
        )
        flat = {}
        for s, per in table.items():
            flat[f"{s}/macro_f1"] = per[n]["macro_f1"]
            flat[f"{s}/safe_recall"] = per[n]["safe_recall"]
        run.log(0, **flat)
        run.finish()


def cmd_show(args):
    data = load(args.root)
    for s, idx in sets_of(data).items():
        fams = len(set(data.family[idx]))
        print(
            f"{s:6s} rows {len(idx):4d} families {fams:3d} classes "
            f"{np.bincount(data.y[idx], minlength=3).tolist()}"
        )
    for n in ("N", "ALL", "ENV:FS:1000"):
        print(n, "|", cell_texts(data, n)[int(sets_of(data)["facts"][0])][:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("show", "stage1", "cv", "score"))
    ap.add_argument("--which", choices=("e17", "e16"), default="e17")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--cells", default="")
    ap.add_argument("--project", default="jevmon-context")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--runs", type=Path, default=Path(os.environ.get("JEVMON_RUNS", "runs")))
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    {"show": cmd_show, "stage1": cmd_stage1, "cv": cmd_cv, "score": cmd_score}[args.mode](args)


if __name__ == "__main__":
    main()

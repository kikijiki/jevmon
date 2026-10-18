"""Train the three shipped judge heads: a linear head on frozen embeddings.

Each shipped judge model is a linear classifier over a frozen embedding of a rendered launch
context: `laya` (the local Laya encoder, mean-pooled), `ettin` (`jhu-clsp/ettin-encoder-1b`) and
`qwen3e` (`Qwen/Qwen3-Embedding-8B`), the latter two reading the concatenation of the mean of the
last layer and of the layer at 3/4 depth (`mean+mean75`). All three are fitted on the mixed train
pool (original lab train rows + E19b consensus train rows + the shell-safety and cli-command-risk
own train rows) and scored once on the lab, E19b and external held-out sets.

Three subcommands:

    just gpu-run-lab python lab/scripts/train_heads.py embed-laya   # resumable, iGPU, Laya mean
    just run-lab     python lab/scripts/train_heads.py run          # CPU, writes lab-data-e20
    just run-lab     python lab/scripts/train_heads.py export       # CPU, writes the shipped heads

`embed-laya` fills `lab/cache/e20/laya-mean-512.pkl` for every text this pipeline trains on or
scores that is not already in `lab/cache/emb-english-mean-512.pkl`. `run` fits the three fixed
recipes on the mixed train pool and scores the held-out sets with a family-clustered bootstrap and
the zero-shot Laya and always-safe references. `export` fits the same recipes on the whole train
pool and writes `app/src/jevmon/laya/heads/{laya,ettin,qwen3e}.safetensors`, recording each
held-out macro-F1.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from improve_lib import Spec, fit_predict, load_rows, render, sample_weights
from jevmon_lab.manifest import LABELS

# ------------------------------------------------------------------ paths and constants

# The one place the archived lab captures are read from (`data/archive/lab-data-*`).
ARCHIVE = ROOT / "data" / "archive"
LAB = ("lab-data-3c-r6", "lab-data-3c-r7")
SETTLE_OLD = "lab-data-settle-old"
SETTLE_NEW = "lab-data-settle-new"
GEN2 = "lab-data-gen2-2026-10-17"
EXT_DIRS = {
    "shell-safety": "lab-data-ext-shell-safety",
    "cli-risk": "lab-data-ext-cli-risk",
    "quasarnix": "lab-data-ext-quasarnix",
    "splunk": "lab-data-ext-splunk",
}
EXT_TRAIN = ("shell-safety", "cli-risk")
OUT_DIR = ROOT / "lab-data-e20"
LAYA_NEW = ROOT / "lab/cache/e20/laya-mean-512.pkl"
APP_HEADS = ROOT / "app/src/jevmon/laya/heads"

# The three shipped judge models and their fixed recipes. `name`: packaged file stem; `parts`:
# (backbone, variant) feature columns concatenated in order; `config`: the feature-config name;
# `backbone_id`: HF id recorded in the head ("" = the local Laya encoder, no HF download).
HEADS = {
    "laya": (
        (("laya", "mean"),),
        Spec(wd=0.001, balanced=True),
        "laya:mean",
        "",
    ),
    "ettin": (
        (("ettin-1b", "mean"), ("ettin-1b", "mean75")),
        Spec(wd=0.001, balanced=True),
        "ettin-1b:mean+mean75",
        "jhu-clsp/ettin-encoder-1b",
    ),
    "qwen3e": (
        (("qwen3e-8b", "mean"), ("qwen3e-8b", "mean75")),
        Spec(wd=0.1, balanced=True),
        "qwen3e-8b:mean+mean75",
        "Qwen/Qwen3-Embedding-8B",
    ),
}

BOOT = 10000
SEED = 20

# ------------------------------------------------------------------ data


@dataclass
class Row:
    source: str
    family: str
    label: str
    text: str
    group: str = ""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def lab_train() -> list[Row]:
    out = []
    for d in LAB:
        for r in load_rows(ARCHIVE / d / "dataset.jsonl"):
            if r["split"] == "train":
                out.append(Row("lab", "lab::" + r["family"], r["label"], render(r, "shipped")))
    return out


def lab_pool() -> list[Row]:
    out = []
    for d in LAB:
        for r in load_rows(ARCHIVE / d / "dataset.jsonl"):
            if r["split"] == "heldout":
                out.append(
                    Row("lab", "lab::" + r["family"], r["label"], render(r, "shipped"), "old")
                )
    for group, d in (("re", SETTLE_OLD), ("fresh", SETTLE_NEW)):
        for r in load_rows(ARCHIVE / d / "dataset.jsonl"):
            assert r["split"] == "heldout", r["family"]
            out.append(Row("lab", "lab::" + r["family"], r["label"], render(r, "shipped"), group))
    return out


def gen2() -> tuple[list[Row], list[Row]]:
    rows = read_jsonl(ROOT / GEN2 / "dataset.jsonl")
    train = [
        Row("gen2", "gen2::" + r["family"], r["label"], r["context"])
        for r in rows
        if r["consensus"] and r["split"] == "train"
    ]
    held = [
        Row("gen2", "gen2::" + r["family"], r["label"], r["context"])
        for r in rows
        if r["consensus"] and r["split"] == "heldout"
    ]
    return train, held


def ext_train() -> list[Row]:
    out = []
    for src in EXT_TRAIN:
        for r in read_jsonl(ROOT / EXT_DIRS[src] / "dataset.jsonl"):
            if r["role"] == "aug_train" and r["label"] and r["kind"] in ("driver", "direct"):
                out.append(Row(f"ext-{src}", f"{src}::" + r["family"], r["label"], r["context"]))
    return out


def training_rows() -> list[Row]:
    return lab_train() + gen2()[0] + ext_train()


@dataclass
class Group:
    name: str
    rows: list[Row]
    labelled: bool = True
    note: str = ""


def eval_groups() -> list[Group]:
    import ext_eval

    pool = lab_pool()
    _, held = gen2()
    groups = [
        Group(
            "lab/pool", pool, note="original lab held-out pool (old+recapture+fresh, 61 families)"
        ),
        Group("lab/old", [r for r in pool if r.group == "old"], note="r6+r7 held-out, 22 families"),
        Group("lab/re", [r for r in pool if r.group == "re"], note="settle recapture"),
        Group("lab/fresh", [r for r in pool if r.group == "fresh"], note="settle fresh families"),
        Group("gen2/heldout", held, note="E19b consensus held-out (multi-judge consensus labels)"),
    ]
    for s in ext_eval.load_sets([ROOT / d for d in EXT_DIRS.values()]):
        rows = [
            Row(s.source, r["family"], r["label"], r["context"]) for r in s.rows if r.get("context")
        ]
        groups.append(Group(s.name, rows, labelled=s.labelled, note=s.note))
    return groups


# ------------------------------------------------------------------ features


_STORES: dict[str, dict] = {}


def store(name: str) -> dict:
    if name not in _STORES:
        if name == "laya":
            paths = [ROOT / "lab/cache/emb-english-mean-512.pkl", LAYA_NEW]
        else:
            paths = [
                ROOT / f"lab/cache/bb-{name}-512.pkl",
                ROOT / f"lab/cache/e20/bb-{name}-512.pkl",
            ]
        merged: dict = {}
        for p in paths:
            if p.exists():
                merged.update(pickle.loads(p.read_bytes()))
        _STORES[name] = merged
    return _STORES[name]


def vector(name: str, variant: str, text: str) -> np.ndarray:
    v = store(name)[text]
    return v if name == "laya" else v[variant]


def matrix(parts: tuple[tuple[str, str], ...], texts: list[str]) -> np.ndarray:
    cols = []
    for name, variant in parts:
        missing = [t for t in texts if t not in store(name)]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} texts not cached (run embed-laya first)")
        cols.append(np.stack([vector(name, variant, t) for t in texts]))
    return np.concatenate(cols, axis=1).astype(np.float32)


def head_key(name: str) -> str:
    """The results.json key of a shipped head's held-out entry (`config|recipe`)."""
    _, spec, config_name, _ = HEADS[name]
    return f"{config_name}|{spec.name()}"


# ------------------------------------------------------------------ embed-laya


def cmd_embed_laya(args: argparse.Namespace) -> None:
    train = training_rows()
    groups = eval_groups()
    texts = sorted({r.text for r in train} | {r.text for g in groups for r in g.rows})
    have = set(store("laya"))
    todo = [t for t in texts if t not in have]
    print(f"{len(texts)} texts, {len(todo)} missing from the Laya cache", flush=True)
    if not todo:
        return
    import laya

    from jevmon.laya.head import embed_states

    agent = laya.load(
        model_id_or_path="convaiinnovations/laya",
        device=args.device,
        revision=laya.PINNED_REVISIONS["convaiinnovations/laya"],
    )
    cur = pickle.loads(LAYA_NEW.read_bytes()) if LAYA_NEW.exists() else {}
    batch = args.batch
    started = time.perf_counter()
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        vecs = embed_states(agent, chunk, pooling="mean", max_length=512, batch_size=batch)
        for t, v in zip(chunk, vecs, strict=True):
            cur[t] = v.astype(np.float32)
        if i % (batch * 20) == 0:
            LAYA_NEW.parent.mkdir(parents=True, exist_ok=True)
            LAYA_NEW.write_bytes(pickle.dumps(cur))
            done = min(i + batch, len(todo))
            rate = done / max(time.perf_counter() - started, 1e-9)
            print(f"  {done}/{len(todo)} ({rate:.1f}/s)", flush=True)
    LAYA_NEW.write_bytes(pickle.dumps(cur))
    print(f"wrote {len(cur)} vectors to {LAYA_NEW}", flush=True)


# ------------------------------------------------------------------ metrics


def _installed() -> tuple:
    import ext_eval

    return ext_eval.confusion, ext_eval.macro_f1_from


def _metric(conf: np.ndarray, present: list[int]) -> np.ndarray:
    """Macro-F1 over the present classes; for a single-class set its recall (as ext_eval does)."""
    if len(present) == 1:
        c = present[0]
        return conf[..., c, c] / np.maximum(conf[..., c, :].sum(axis=-1), 1)
    _, macro_f1_from = _installed()
    return macro_f1_from(conf, present)


def score_point(y: np.ndarray, cand: np.ndarray) -> float:
    """The headline point metric (no bootstrap): macro-F1, or recall for a single-class set."""
    confusion, _ = _installed()
    return float(_metric(confusion(y, cand), sorted(set(y.tolist()))))


def summarize(y: np.ndarray, cand: np.ndarray, fam: np.ndarray) -> dict:
    """A head's score with a family-cluster bootstrap CI.

    A multi-class set scores macro-F1 over the present classes; a single-class set scores recall
    (QuasarNix is crit only), matching `ext_eval`.
    """
    confusion, _ = _installed()
    present = sorted(set(y.tolist()))
    fams = sorted(set(fam.tolist()))
    members = [np.flatnonzero(fam == f) for f in fams]
    boot_c = np.stack([confusion(y[m], cand[m]) for m in members])
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, len(fams), size=(BOOT, len(fams)))
    f1_c = _metric(boot_c[draw].sum(axis=1), present)
    conf = confusion(y, cand)
    safe_only = np.zeros(len(y), dtype=int)
    out = {
        "n": len(y),
        "families": len(fams),
        "metric": "recall" if len(present) == 1 else "macro_f1",
        "support": {LABELS[c]: int((y == c).sum()) for c in present},
        "macro_f1": float(_metric(conf, present)),
        "macro_f1_ci": [float(np.percentile(f1_c, 2.5)), float(np.percentile(f1_c, 97.5))],
        "accuracy": float((y == cand).mean()),
        "always_safe": float(_metric(confusion(y, safe_only), present)),
        "confusion": conf.tolist(),
        "predicted": {LABELS[c]: int((cand == c).sum()) for c in range(3)},
        "per_class": {},
    }
    for c in present:
        tp = conf[c, c]
        out["per_class"][LABELS[c]] = {
            "precision": float(tp / conf[:, c].sum()) if conf[:, c].sum() else 0.0,
            "recall": float(tp / conf[c].sum()),
        }
    return out


def predicted_mix(pred: np.ndarray) -> dict:
    return {LABELS[c]: int((pred == c).sum()) for c in range(3)} | {"n": len(pred)}


# ------------------------------------------------------------------ run


def cmd_run(args: argparse.Namespace) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    import torch

    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    train = training_rows()
    groups = eval_groups()
    print(
        "train rows",
        len(train),
        "| groups " + ", ".join(f"{g.name}({len(g.rows)})" for g in groups),
    )
    train_texts = [r.text for r in train]
    y_train = np.asarray([LABELS.index(r.label) for r in train])
    key_train = np.asarray([f"{r.text}\x00{r.label}" for r in train])
    all_texts = [r.text for g in groups for r in g.rows]

    preds: dict[str, np.ndarray] = {}
    for name, (parts, spec, _config, _backbone) in HEADS.items():
        x_tr = matrix(parts, train_texts)
        x_eval = matrix(parts, all_texts)
        preds[name] = fit_predict(spec, x_tr, y_train, key_train, x_eval).argmax(1)
        print(f"fitted {name:7s} ({spec.name()}) in {time.perf_counter() - t0:.0f}s", flush=True)

    zero = zero_shot_reference()
    results: dict = {"sets": {}, "heads": list(HEADS)}
    offset = 0
    for g in groups:
        sl = slice(offset, offset + len(g.rows))
        offset += len(g.rows)
        entry: dict = {"n": len(g.rows), "labelled": g.labelled, "note": g.note, "judges": {}}
        ref = zero.get(g.name)
        if ref is not None:
            entry["zero_shot_laya"] = ref
        if g.labelled:
            y = np.asarray([LABELS.index(r.label) for r in g.rows])
            entry["support"] = {LABELS[c]: int((y == c).sum()) for c in range(3)}
            fam = np.asarray([r.family for r in g.rows])
            for name in HEADS:
                entry["judges"][head_key(name)] = summarize(y, preds[name][sl], fam)
        else:
            entry["always_safe"] = 0.0
            for name in HEADS:
                entry["judges"][head_key(name)] = predicted_mix(preds[name][sl])
        results["sets"][g.name] = entry
        print(f"{g.name:46s} n={len(g.rows):5d}", flush=True)

    results["zero_shot_reference"] = zero
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1))
    (OUT_DIR / "summary.md").write_text(render_report(results))
    print(f"wrote {OUT_DIR / 'summary.md'} in {time.perf_counter() - t0:.0f}s")


def fit_linear(spec: Spec, x_tr: np.ndarray, y: np.ndarray, key: np.ndarray) -> tuple:
    """The exact objective of `improve_lib._logreg`, returned as (weight, bias) on the raw input.

    Kept in lock-step with `fit_predict`: same prep, sample weights, normalisation and LBFGS call,
    so a head exported here scores its cached features exactly like the `run` candidate did. A
    `standardize` step (not used by the shipped recipes) is folded into weight and bias.
    """
    import torch
    import torch.nn.functional as F

    from improve_lib import _prep

    raw = x_tr
    xs, _ = _prep(spec, x_tr, x_tr)
    w = sample_weights(spec, y, key, len(LABELS))
    w = w / max(w.mean(), 1e-12)
    xt, yt, wt = (
        torch.from_numpy(xs),
        torch.from_numpy(y),
        torch.from_numpy(w.astype(np.float32)),
    )
    wg = torch.zeros(len(LABELS), xs.shape[1], requires_grad=True)
    b = torch.zeros(len(LABELS), requires_grad=True)
    opt = torch.optim.LBFGS([wg, b], lr=1.0, line_search_fn="strong_wolfe", max_iter=200)

    def closure():
        opt.zero_grad()
        ce = F.cross_entropy(xt @ wg.T + b, yt, reduction="none", label_smoothing=spec.smoothing)
        loss = (ce * wt).sum() / wt.sum() + 0.5 * spec.wd * (wg * wg).sum()
        loss.backward()
        return loss

    opt.step(closure)
    wn, bn = wg.detach().numpy(), b.detach().numpy()
    if spec.standardize:
        mu, sd = raw.mean(0, keepdims=True), raw.std(0, keepdims=True) + 1e-6
        wn = wn / sd
        bn = bn - (wn * mu).sum(1)
    return wn.astype(np.float32), bn.astype(np.float32)


def cmd_export(args: argparse.Namespace) -> None:
    """Fit the three shipped heads on the mixed train pool and write packaged safetensors.

    Each head is the exact recipe already scored in `lab-data-e20/results.json`; this only fits it
    on the whole mixed train pool and records the per-set held-out macro-F1 in the head's metadata.
    It never reads held-out labels to change anything - the recipes are fixed constants.
    """
    sys.stdout.reconfigure(line_buffering=True)
    import torch

    from jevmon.laya.head import TunedHead

    torch.set_num_threads(args.threads)
    train = training_rows()
    groups = eval_groups()
    texts = [r.text for r in train]
    y = np.asarray([LABELS.index(r.label) for r in train])
    key = np.asarray([f"{r.text}\x00{r.label}" for r in train])
    train_set = (
        f"mixed pool: original lab train + E19b consensus train + external "
        f"(shell-safety, cli-risk) train rows ({len(train)} rows)"
    )
    results = json.loads((OUT_DIR / "results.json").read_text()) if args.check else {}
    APP_HEADS.mkdir(parents=True, exist_ok=True)
    for name, (parts, spec, config_name, backbone_id) in HEADS.items():
        x_tr = matrix(parts, texts)
        weight, bias = fit_linear(spec, x_tr, y, key)
        metrics: dict[str, float] = {}
        for g in groups:
            if not g.labelled:
                continue
            gtexts = [r.text for r in g.rows]
            pred = (matrix(parts, gtexts) @ weight.T + bias).argmax(1)
            gy = np.asarray([LABELS.index(r.label) for r in g.rows])
            fam = np.asarray([r.family for r in g.rows])
            metrics[g.name] = float(summarize(gy, pred, fam)["macro_f1"])
        candidate = f"{config_name}|{spec.name()}"
        head = TunedHead(
            labels=tuple(LABELS),
            weight=weight,
            bias=bias,
            max_length=512,
            variants=tuple(variant for _, variant in parts),
            backbone=backbone_id,
            recipe=spec.name(),
            training_set=train_set,
            metrics=tuple(metrics.items()),
        )
        out = APP_HEADS / f"{name}.safetensors"
        head.save(out)
        note = ""
        if results:
            ref = results["sets"]["lab/pool"]["judges"].get(candidate, {}).get("macro_f1")
            here = metrics["lab/pool"]
            if ref is not None:
                note = f" | results.json lab/pool {ref:.6f} ({'ok' if abs(ref - here) < 5e-4 else 'MISMATCH'})"
        print(
            f"{name:7s} dim {weight.shape[1]:5d} {candidate} -> {out.name}"
            f" | lab/pool macro-F1 {metrics['lab/pool']:.6f}{note}",
            flush=True,
        )


# ------------------------------------------------------------------ reference + report


def zero_shot_reference() -> dict[str, dict]:
    """Zero-shot Laya macro-F1 per eval set where a cached prediction exists (not recomputed).

    External rows come from `lab-data-ext-results/preds.json`; the E19b held-out from
    `stage1-preds.jsonl`, whose `consensus` field is stale (`false` on every row), so records are
    joined to `dataset.jsonl` by their index `i` before filtering on consensus.
    """
    import ext_eval

    confusion, macro_f1_from = _installed()
    out: dict[str, dict] = {}
    preds_path = ROOT / "lab-data-ext-results/preds.json"
    if preds_path.exists():
        cache = json.loads(preds_path.read_text()).get("laya-zero-shot", {})
        for s in ext_eval.load_sets([ROOT / d for d in EXT_DIRS.values()]):
            if not s.labelled or not s.rows or any(r["context"] not in cache for r in s.rows):
                continue
            y = np.asarray([LABELS.index(r["label"]) for r in s.rows])
            pred = np.asarray([int(np.argmax(cache[r["context"]])) for r in s.rows])
            present = sorted(set(y.tolist()))
            out[s.name] = {
                "macro_f1": float(macro_f1_from(confusion(y, pred), present)),
                "n": len(y),
            }
    stage1 = ROOT / GEN2 / "stage1-preds.jsonl"
    if stage1.exists():
        ds = read_jsonl(ROOT / GEN2 / "dataset.jsonl")
        sp = {r["i"]: r for r in read_jsonl(stage1)}
        held_idx = [i for i, r in enumerate(ds) if r["consensus"] and r["split"] == "heldout"]
        y = np.asarray([LABELS.index(ds[i]["label"]) for i in held_idx])
        pred = np.asarray([LABELS.index(sp[i]["zero_shot_laya"]) for i in held_idx])
        conf = confusion(y, pred)
        out["gen2/heldout"] = {
            "macro_f1": float(macro_f1_from(conf, sorted(set(y.tolist())))),
            "n": len(y),
        }
    return out


def render_report(results: dict) -> str:
    lines = [
        "# Shipped heads on mixed command styles",
        "",
        "The three heads are fitted on the mixed train pool only (original lab train + E19b",
        "consensus train + external train rows) and scored once on the held-out / external /",
        "consensus-held-out sets. Bootstrap: family-cluster,",
        f"{BOOT} draws, seed {SEED}.",
        "",
    ]
    for name, entry in results["sets"].items():
        lines.append(f"### {name} (n = {entry['n']})")
        if entry["note"]:
            lines.append(entry["note"])
        lines.append("")
        if not entry["labelled"]:
            lines.append("| head | safe | warn | crit |")
            lines.append("| --- | --- | --- | --- |")
            for j, m in entry["judges"].items():
                lines.append(f"| {j} | {m['safe']} | {m['warn']} | {m['crit']} |")
            lines.append("")
            continue
        lines.append("| head | macro-F1 [95% CI] | always-safe | safe P/R | warn P/R | crit P/R |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for j, m in entry["judges"].items():
            if "macro_f1" not in m:
                continue
            label = j if m.get("metric") != "recall" else f"{j} (recall)"
            pc = " | ".join(
                f"{m['per_class'].get(c, {}).get('precision', float('nan')):.2f} / "
                f"{m['per_class'].get(c, {}).get('recall', float('nan')):.2f}"
                for c in LABELS
            )
            lines.append(
                f"| {label} | {m['macro_f1']:.3f} [{m['macro_f1_ci'][0]:.2f}, {m['macro_f1_ci'][1]:.2f}] "
                f"| {m['always_safe']:.3f} | {pc} |"
            )
        if "zero_shot_laya" in entry:
            lines.append(f"| zero-shot Laya | {entry['zero_shot_laya']['macro_f1']:.3f} | | |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed-laya")
    e.add_argument("--device", default="cuda")
    e.add_argument("--batch", type=int, default=32)
    e.set_defaults(func=cmd_embed_laya)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=16)
    r.set_defaults(func=cmd_run)
    x = sub.add_parser("export")
    x.add_argument("--threads", type=int, default=16)
    x.add_argument(
        "--check",
        action="store_true",
        help="compare the fitted lab/pool macro-F1 against lab-data-e20/results.json",
    )
    x.set_defaults(func=cmd_export)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

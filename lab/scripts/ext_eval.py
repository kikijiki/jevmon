"""External test sets: score the shipped judges, no training, no tuning.

    just gpu-run-lab python lab/scripts/ext_eval.py \
        --data lab-data-ext-shell-safety lab-data-ext-cli-risk lab-data-ext-quasarnix \
               lab-data-ext-splunk --out lab-data-ext-results

Judges: zero-shot Laya and the three packaged heads in `app/src/jevmon/laya/heads/` (`laya` on
Laya embeddings; `ettin` and `qwen3e` on their own backbone's embeddings of the same text). The
text of every row is the one `ingest.Renderer` produced (shipped format, variant A). Embeddings
and verdicts are cached per text, so a rerun only scores new rows. Every set is test only; nothing
is fitted. Each source is reported on its own and never pooled with another source or the lab pool.

Metrics: macro-F1 over the classes present in the set's labels, with a 95% family-clustered
bootstrap CI (families resampled with replacement, 10000 draws); per-class precision/recall; the
confusion matrix (true rows, predicted columns); the always-safe baseline. Sets with one label
class report recall. The sensitivity rows drop labels the mapping marks uncertain.
"""

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import backbone_embed
from jevmon.laya.head import TunedHead, embed_states
from jevmon_lab import ingest

LABELS = ("safe", "warn", "crit")
BOOT = 10000
SEED = 20
# backbone cache name -> (packaged head name, HF id, encoder/decoder)
BACKBONES = {
    "ettin-1b": ("ettin", "jhu-clsp/ettin-encoder-1b", "encoder"),
    "qwen3e-8b": ("qwen3e", "Qwen/Qwen3-Embedding-8B", "decoder"),
}
LAYERS = ("laya-zero-shot", "laya")


@dataclass
class EvalSet:
    name: str  # source/stratum
    source: str
    rows: list[dict]
    labelled: bool = True
    note: str = ""


# ---------------------------------------------------------------- sets


def load_sets(dirs: list[Path]) -> list[EvalSet]:
    sets: list[EvalSet] = []
    for d in dirs:
        rows = ingest.read_dataset(d / "dataset.jsonl")
        src = rows[0]["source"]

        def pick(kind: str, role: str = "test", rows=rows):
            return [r for r in rows if r["role"] == role and r["kind"] == kind and r["label"]]

        if src == "splunk-attack-data-linux":
            rev = [
                dict(r, label=r["provisional_label"])
                for r in rows
                if r.get("in_review") and r.get("provisional_label") in LABELS
            ]
            sets.append(
                EvalSet(
                    "splunk/reviewed",
                    src,
                    rev,
                    note="300-row sample, labels written by the assistant, unreviewed",
                )
            )
            sets.append(
                EvalSet(
                    "splunk/reviewed, label independent of parent",
                    src,
                    [r for r in rev if not r.get("provisional_note", "").startswith("[parent]")],
                    note="drops rows whose note says the label rests on the parent process",
                )
            )
            sets.append(
                EvalSet(
                    "splunk/folder labels (weak)",
                    src,
                    pick("launch"),
                    note="technique-folder label for every non-background launch",
                )
            )
            sets.append(
                EvalSet(
                    "splunk/background (unlabelled)",
                    src,
                    [r for r in rows if r["role"] == "background"],
                    labelled=False,
                )
            )
            continue
        sets.append(EvalSet(f"{src}/driver", src, pick("driver")))
        sets.append(EvalSet(f"{src}/direct", src, pick("direct")))
        if src == "quasarnix":
            sets.append(
                EvalSet(
                    "quasarnix/nl2bash (unlabelled)",
                    src,
                    [r for r in rows if r["role"] == "background" and r["kind"] == "driver"],
                    labelled=False,
                )
            )
        else:
            primary = sets[-2]
            sets.append(
                EvalSet(
                    f"{src}/driver, certain labels only",
                    src,
                    [r for r in primary.rows if not r["uncertain"]],
                    note="drops rows whose mapping rule is flagged uncertain",
                )
            )
    return sets


# ---------------------------------------------------------------- judges


def laya_decider(tuned: str = ""):
    """The real LayaDecider on shipped defaults (an empty config file, so no user overrides)."""
    import tempfile

    from jevmon.config import load_config
    from jevmon.monitor import build_decider

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write("")
    cfg = load_config(fh.name)
    cfg = replace(cfg, judge=replace(cfg.judge, tuned=tuned, checkpoint="english"))
    decider = build_decider(cfg)
    decider.load()
    if decider.error:
        raise SystemExit(f"model failed to load: {decider.error}")
    return decider


def score_laya(texts: list[str], cache: dict) -> None:
    """Zero-shot Laya (probabilities per class) and the packaged `laya` head for uncached texts."""
    todo = [t for t in texts if t not in cache.get("laya-zero-shot", {})]
    if not todo:
        return
    decider = laya_decider("")
    agent = decider._agent(decider._router)
    head = TunedHead.load("laya")
    zs = cache.setdefault("laya-zero-shot", {})
    hd = cache.setdefault("laya", {})
    for i in range(0, len(todo), 32):
        chunk = todo[i : i + 32]
        for t, dec in zip(chunk, decider.decide_batch(chunk), strict=True):
            zs[t] = [float(dec.probabilities.get(c, 0.0)) for c in LABELS]
        vecs = embed_states(agent, chunk, pooling=head.pooling, max_length=head.max_length)
        probs = head.probabilities(vecs)
        order = [head.labels.index(c) for c in LABELS]
        for t, p in zip(chunk, probs, strict=True):
            hd[t] = [float(p[j]) for j in order]
        print(f"  laya {min(i + 32, len(todo))}/{len(todo)}", flush=True)


def score_backbone(name: str, texts: list[str], cache: dict, device: str) -> None:
    todo = [t for t in texts if t not in cache.get(name, {})]
    if not todo:
        return
    head_name, model_id, kind = BACKBONES[name]
    head = TunedHead.load(head_name)
    variants = head.feature_variants
    backbone_embed.CACHE_DIR = Path("lab/cache/ext")
    backbone_embed.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feats = backbone_embed.embed(
        name,
        model_id,
        sorted(todo),
        kind=kind,
        max_length=head.max_length,
        device=device,
        batch_size=8,
    )
    vectors = np.concatenate([feats[v] for v in variants], axis=1).astype(np.float32)
    probs = head.probabilities(vectors)
    order = [head.labels.index(c) for c in LABELS]
    out = cache.setdefault(name, {})
    for t, p in zip(sorted(todo), probs, strict=True):
        out[t] = [float(p[j]) for j in order]


# ---------------------------------------------------------------- metrics


def confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    m = np.zeros((3, 3), dtype=np.int64)
    np.add.at(m, (y, p), 1)
    return m


def macro_f1_from(conf: np.ndarray, present: list[int]) -> float:
    f1s = []
    for c in present:
        tp = conf[..., c, c]
        fp = conf[..., :, c].sum(-1) - tp
        fn = conf[..., c, :].sum(-1) - tp
        denom = 2 * tp + fp + fn
        f1s.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return np.mean(f1s, axis=0)


def summarize(y: np.ndarray, p: np.ndarray, fam: np.ndarray) -> dict:
    present = sorted(set(y.tolist()))
    conf = confusion(y, p)
    fams = sorted(set(fam.tolist()))
    per_fam = np.stack([confusion(y[fam == f], p[fam == f]) for f in fams])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(fams), size=(BOOT, len(fams)))
    boot = per_fam[idx].sum(axis=1)  # (BOOT, 3, 3)
    out = {
        "n": len(y),
        "families": len(fams),
        "support": {LABELS[c]: int((y == c).sum()) for c in present},
        "accuracy": float((y == p).mean()),
        "confusion": conf.tolist(),
        "per_class": {},
    }
    for c in present:
        tp = conf[c, c]
        pred_n = conf[:, c].sum()
        rec_b = boot[:, c, c] / np.maximum(boot[:, c, :].sum(axis=1), 1)
        out["per_class"][LABELS[c]] = {
            "precision": float(tp / pred_n) if pred_n else 0.0,
            "recall": float(tp / conf[c].sum()),
            "recall_ci": [float(np.percentile(rec_b, 2.5)), float(np.percentile(rec_b, 97.5))],
        }
    if len(present) > 1:
        out["macro_f1"] = float(macro_f1_from(conf, present))
        vals = macro_f1_from(boot, present)
        out["macro_f1_ci"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        safe_only = np.zeros(len(y), dtype=int)
        out["always_safe_macro_f1"] = float(macro_f1_from(confusion(y, safe_only), present))
    out["predicted"] = {LABELS[c]: int((p == c).sum()) for c in range(3)}
    return out


def verdict_mix(p: np.ndarray) -> dict:
    return {LABELS[c]: int((p == c).sum()) for c in range(3)} | {"n": len(p)}


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judges", default="laya,ettin-1b,qwen3e-8b")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    args.out.mkdir(parents=True, exist_ok=True)
    sets = load_sets(args.data)
    texts = sorted({r["context"] for s in sets for r in s.rows})
    print(f"{len(sets)} sets, {len(texts)} distinct texts", flush=True)

    cache_path = args.out / "preds.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    wanted = args.judges.split(",")
    if "laya" in wanted:
        score_laya(texts, cache)
        cache_path.write_text(json.dumps(cache))
    for name in BACKBONES:
        if name in wanted:
            print(f"backbone {name}", flush=True)
            score_backbone(name, texts, cache, args.device)
            cache_path.write_text(json.dumps(cache))

    judges = [j for j in (*LAYERS, *BACKBONES) if j in cache]
    results: dict = {"sets": {}, "texts": len(texts)}
    for s in sets:
        entry: dict = {"source": s.source, "note": s.note, "n": len(s.rows), "judges": {}}
        if not s.rows:
            continue
        fam = np.array([r["family"] for r in s.rows])
        for j in judges:
            probs = np.array([cache[j][r["context"]] for r in s.rows])
            pred = probs.argmax(1)
            if s.labelled:
                y = np.array([LABELS.index(r["label"]) for r in s.rows])
                entry["judges"][j] = summarize(y, pred, fam)
            else:
                entry["judges"][j] = verdict_mix(pred)
        if s.labelled:
            entry["uncertain_rows"] = sum(1 for r in s.rows if r.get("uncertain"))
            entry["label_counts"] = {c: sum(1 for r in s.rows if r["label"] == c) for c in LABELS}
        results["sets"][s.name] = entry
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    (args.out / "summary.md").write_text(render_markdown(results, judges))
    print((args.out / "summary.md").read_text())


def fmt_ci(d: dict) -> str:
    lo, hi = d["macro_f1_ci"]
    return f"{d['macro_f1']:.3f} [{lo:.2f}, {hi:.2f}]"


def render_markdown(results: dict, judges: list[str]) -> str:
    lines: list[str] = []
    for name, entry in results["sets"].items():
        first = next(iter(entry["judges"].values()))
        lines.append(f"### {name} (n = {entry['n']})\n")
        if "support" not in first:
            lines.append("| judge | " + " | ".join(LABELS) + " |")
            lines.append("| --- | --- | --- | --- |")
            for j in judges:
                m = entry["judges"][j]
                lines.append(f"| {j} | " + " | ".join(str(m[c]) for c in LABELS) + " |")
            lines.append("")
            continue
        lines.append(
            f"families {first['families']}, labels {entry['label_counts']}, "
            f"uncertain-mapped rows {entry['uncertain_rows']}. {entry['note']}\n"
        )
        multi = "macro_f1" in first
        head = "| judge | acc |" + (" macro-F1 [95% CI] | always-safe |" if multi else "")
        head += " " + " | ".join(f"{c} P / R" for c in LABELS) + " |"
        lines.append(head)
        lines.append("| --- |" + " --- |" * (head.count("|") - 2))
        for j in judges:
            m = entry["judges"][j]
            cells = [j, f"{m['accuracy']:.3f}"]
            if multi:
                cells += [fmt_ci(m), f"{m['always_safe_macro_f1']:.3f}"]
            for c in LABELS:
                pc = m["per_class"].get(c)
                cells.append(f"{pc['precision']:.2f} / {pc['recall']:.2f}" if pc else "-")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Confusion (true rows, predicted columns safe / warn / crit):\n")
        for j in judges:
            c = entry["judges"][j]["confusion"]
            lines.append(f"- {j}: {c}")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

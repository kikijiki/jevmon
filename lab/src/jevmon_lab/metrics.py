"""Held-out scoring for judge verdicts, with the class-mix artefacts always in view.

The lab's labelled contexts are dominated by `safe`, so accuracy alone flatters any judge
that collapses to one class. Every score therefore carries the always-safe baseline
(`safe` predicted for every row) and the majority-class rate next to the model's own
numbers, plus per-class precision/recall: a class-mix artefact cannot hide behind them.

Rows are mappings with `label` (ground truth, `None` for unlabelled) and `verdict` (the judge's
class). Unlabelled and unjudged rows are the caller's business: score what you pass.
"""

from collections import Counter

from jevmon_lab.manifest import LABELS

Row = dict


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score(rows: list[Row], labels: tuple[str, ...] = LABELS) -> dict:
    """Accuracy, macro-F1 and per-class precision/recall, each next to its baseline.

    The always-safe baseline is scored with the same function on the same rows, so the two
    numbers always describe the same class mix.
    """
    truth = [str(row["label"]) for row in rows]
    pred = [str(row["verdict"]) for row in rows]
    support = Counter(truth)
    if not truth:
        raise ValueError("no rows to score")
    majority_class, majority_n = support.most_common(1)[0]
    active = [lab for lab in labels if support.get(lab) or lab in pred]
    per_class: dict[str, dict] = {}
    f1s: list[float] = []
    for lab in active:
        tp = sum(1 for t, p in zip(truth, pred, strict=True) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(truth, pred, strict=True) if t != lab and p == lab)
        fn = support.get(lab, 0) - tp
        precision, recall, f1 = _prf(tp, fp, fn)
        per_class[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.get(lab, 0),
            "predicted": sum(1 for p in pred if p == lab),
        }
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    base_f1s: list[float] = []
    for lab in active:
        tp = sum(1 for t in truth if t == lab and lab == "safe")
        fp = sum(1 for t in truth if t != lab) if lab == "safe" else 0
        fn = support.get(lab, 0) - tp
        precision, recall, f1 = _prf(tp, fp, fn)
        base_f1s.append(f1)
    return {
        "n": len(truth),
        "support": {lab: support.get(lab, 0) for lab in labels},
        "accuracy": sum(1 for t, p in zip(truth, pred, strict=True) if t == p) / len(truth),
        "always_safe_accuracy": sum(1 for t in truth if t == "safe") / len(truth),
        "majority_class": majority_class,
        "majority_rate": majority_n / len(truth),
        "macro_f1": macro_f1,
        "always_safe_macro_f1": sum(base_f1s) / len(base_f1s) if base_f1s else 0.0,
        "per_class": per_class,
    }


def lines(result: dict) -> list[str]:
    """Markdown lines for one score: headline next to its baselines, then per-class P/R."""
    out = [
        f"| n | {result['n']} |",
        "| --- | --- |",
        (
            f"| accuracy | {result['accuracy']:.3f} "
            f"(always-safe {result['always_safe_accuracy']:.3f}, "
            f"majority {result['majority_class']} {result['majority_rate']:.3f}) |"
        ),
        (
            f"| macro-F1 | {result['macro_f1']:.3f} "
            f"(always-safe {result['always_safe_macro_f1']:.3f}) |"
        ),
        "",
        "| class | support | predicted | precision | recall | F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for lab, values in result["per_class"].items():
        out.append(
            f"| {lab} | {values['support']} | {values['predicted']} | {values['precision']:.3f} "
            f"| {values['recall']:.3f} | {values['f1']:.3f} |"
        )
    return out

"""Held-out scoring: every headline carries the always-safe baseline and majority rate."""

import pytest

from jevmon_lab.metrics import lines, score


def rows(pairs: list[tuple[str, str]]) -> list[dict]:
    return [{"label": truth, "verdict": pred} for truth, pred in pairs]


def test_perfect_prediction_beats_the_always_safe_baseline() -> None:
    result = score(
        rows(
            [
                ("safe", "safe"),
                ("safe", "safe"),
                ("crit", "crit"),
                ("warn", "warn"),
            ]
        )
    )
    assert result["accuracy"] == 1.0 and result["macro_f1"] == 1.0
    assert result["always_safe_accuracy"] == 0.5  # 2 of 4 rows are safe
    assert result["majority_class"] == "safe" and result["majority_rate"] == 0.5
    # always-safe: precision 2/4, recall 1.0 -> F1 2/3 for safe, 0 for the rest
    assert result["always_safe_macro_f1"] == pytest.approx((2 / 3) / 3)


def test_collapse_to_safe_is_exposed_by_macro_f1_and_recall() -> None:
    result = score(rows([("safe", "safe")] * 8 + [("crit", "safe")] * 2))
    assert result["accuracy"] == 0.8
    assert result["accuracy"] == result["always_safe_accuracy"]
    assert result["macro_f1"] == result["always_safe_macro_f1"]
    assert result["per_class"]["crit"]["recall"] == 0.0
    assert result["per_class"]["safe"]["precision"] == 0.8


def test_per_class_precision_and_recall_are_reported() -> None:
    result = score(
        rows(
            [
                ("crit", "crit"),
                ("crit", "safe"),
                ("safe", "crit"),
                ("safe", "safe"),
            ]
        )
    )
    assert result["per_class"]["crit"]["precision"] == 0.5
    assert result["per_class"]["crit"]["recall"] == 0.5
    assert result["per_class"]["safe"]["precision"] == 0.5
    assert result["per_class"]["safe"]["recall"] == 0.5
    assert result["n"] == 4


def test_lines_show_the_baseline_next_to_every_headline() -> None:
    text = "\n".join(lines(score(rows([("safe", "safe"), ("warn", "warn")]))))
    assert "always-safe 0.5" in text and "majority safe 0.5" in text
    assert "always-safe 0.333" in text  # macro-F1 of the always-safe predictor

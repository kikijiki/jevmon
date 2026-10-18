"""Head-space and metric helpers of `lab/scripts/train_heads.py`: pure functions, no data needed."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import learning_curve as lc
import train_heads as th
from improve_lib import Spec

VARIANTS = {
    "laya": {"mean"},
    "ettin-1b": {"mean", "mean75", "cls"},
    "qwen3e-8b": {"mean", "mean75"},
}


def config_name(parts) -> str:
    by_backbone: dict[str, list[str]] = {}
    for backbone, variant in parts:
        by_backbone.setdefault(backbone, []).append(variant)
    return "+".join(f"{b}:{'+'.join(vs)}" for b, vs in by_backbone.items())


def test_heads_are_unique_and_well_formed() -> None:
    assert set(th.HEADS) == {"laya", "ettin", "qwen3e"}
    for name, (parts, spec, config, backbone) in th.HEADS.items():
        assert parts, name
        assert spec.balanced
        for bb, variant in parts:
            assert variant in VARIANTS[bb], (name, bb, variant)
        assert config == config_name(parts)
        assert (
            backbone
            == {
                "laya": "",
                "ettin": "jhu-clsp/ettin-encoder-1b",
                "qwen3e": "Qwen/Qwen3-Embedding-8B",
            }[name]
        )
    assert th.head_key("laya") == "laya:mean|logreg/wd0.001/bal"
    assert th.head_key("qwen3e") == "qwen3e-8b:mean+mean75|logreg/wd0.1/bal"


def test_summarize_perfect_candidate_is_best() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    fam = np.array(["a", "a", "b", "b", "c", "c"])
    good = th.summarize(y, y.copy(), fam)
    assert good["macro_f1"] == 1.0
    assert good["macro_f1_ci"][0] <= good["macro_f1"] <= good["macro_f1_ci"][1]
    assert good["n"] == 6 and good["families"] == 3
    bad = th.summarize(y, np.zeros_like(y), fam)
    assert bad["macro_f1"] < good["macro_f1"]
    assert bad["always_safe"] < 1.0


def test_summarize_single_class_is_recall() -> None:
    y = np.full(4, 2)
    fam = np.array(["a", "a", "b", "b"])
    pred = np.array([2, 2, 0, 2])
    out = th.summarize(y, pred, fam)
    assert out["metric"] == "recall"
    assert abs(out["macro_f1"] - 0.75) < 1e-9
    assert out["always_safe"] == 0.0
    assert "crit" in out["per_class"] and "safe" not in out["per_class"]


def test_predicted_mix_counts_every_row() -> None:
    mix = th.predicted_mix(np.array([0, 1, 1, 2, 2, 2]))
    assert mix["safe"] == 1 and mix["warn"] == 2 and mix["crit"] == 3 and mix["n"] == 6


def test_head_key_is_config_plus_recipe() -> None:
    spec = Spec(wd=0.001, balanced=True)
    assert th.head_key("ettin") == f"ettin-1b:mean+mean75|{spec.name()}"


def test_sample_families_is_stratified_and_deterministic() -> None:
    by_source = {"a": ["a1", "a2", "a3", "a4"], "b": ["b1", "b2", "b3", "b4"]}
    rng = np.random.default_rng(0)
    full = lc.sample_families(by_source, 1.0, rng)
    assert full == ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"]
    half = lc.sample_families(by_source, 0.5, np.random.default_rng(0))
    assert len(half) == 4 and half == lc.sample_families(by_source, 0.5, np.random.default_rng(0))
    assert {f[0] for f in half} == {"a", "b"}  # every source stays represented


def test_draw_keeps_all_three_classes() -> None:
    by_source = {"a": ["f1", "f2", "f3"]}
    fams = np.repeat(["f1", "f2", "f3"], 3)  # each family carries every class
    labels = np.tile([0, 1, 2], 3)
    for frac in (0.05, 0.34, 0.67, 1.0):
        chosen = lc.draw(by_source, frac, 3, labels, fams)
        assert chosen == lc.draw(by_source, frac, 3, labels, fams)
        assert {int(v) for v in labels[np.isin(fams, chosen)]} == {0, 1, 2}

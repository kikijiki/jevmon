"""Pure helpers of `lab/scripts/lora_check.py`: dev split, batching and the paired test, no data."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import lora_check as lc


def test_dev_split_is_deterministic_and_disjoint() -> None:
    families = [f"f{i}" for i in range(200)]
    fit, dev = lc.dev_split(families)
    fit2, dev2 = lc.dev_split(list(reversed(families)))
    assert (fit, dev) == (fit2, dev2)
    assert not (fit & dev)
    assert fit | dev == set(families)
    assert 0 < len(dev) < len(families)


def test_make_batches_covers_every_index_once() -> None:
    lengths = [10, 20, 30, 40, 50, 60, 5, 15, 25]
    batches = lc.make_batches(lengths, np.random.default_rng(0), batch=3)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert all(len(b) <= 3 for b in batches)
    assert batches == lc.make_batches(lengths, np.random.default_rng(0), batch=3)


def test_make_batches_sorts_within_chunks_without_shuffle() -> None:
    lengths = list(range(10, 0, -1))
    batches = lc.make_batches(lengths, rng=None, batch=4, chunk=100)
    for batch in batches:
        got = [lengths[i] for i in batch]
        assert got == sorted(got)


def test_paired_diff_of_identical_predictions_is_zero() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    fam = np.array(["a", "a", "b", "b", "c", "c"])
    diff = lc.paired_diff(y, y.copy(), y.copy(), fam)
    assert diff["point"] == 0.0
    assert diff["ci"] == [0.0, 0.0]
    assert not diff["win"] and not diff["lost"]


def test_paired_diff_flags_a_clear_win() -> None:
    y = np.array([0, 0, 1, 1, 2, 2] * 4)
    fam = np.array([f"f{i}" for i in range(6)] * 4)
    head = np.zeros_like(y)  # always safe
    lora = y.copy()  # perfect
    diff = lc.paired_diff(y, lora, head, fam)
    assert diff["point"] > 0 and diff["win"] and diff["p_better"] > 0.9


def test_ettin_backbone_config() -> None:
    cfg = lc.BACKBONES["ettin"]
    assert cfg["out"].name == "lab-data-lora-ettin"
    assert cfg["out"] != lc.BACKBONES["laya"]["out"]
    assert cfg["model_id"] == "jhu-clsp/ettin-encoder-1b"
    assert cfg["cached_parts"] == (("ettin-1b", "mean"), ("ettin-1b", "mean75"))
    assert lc.LORA_TARGETS == ("Wqkv", "Wo", "Wi")
    assert len(cfg["grid"]) == 4


def test_ettin_grid_is_the_four_best_laya_dev_cells() -> None:
    assert lc.GRID_ETTIN == ((16, 3e-4, 6), (8, 1e-3, 3), (8, 3e-4, 6), (16, 3e-4, 3))
    assert all(cell in lc.GRID for cell in lc.GRID_ETTIN)


def test_out_dir_switches_backbone_and_smoke() -> None:
    from types import SimpleNamespace

    assert lc.out_dir(SimpleNamespace(backbone="laya", smoke=False)).name == "lab-data-lora"
    ettin = lc.out_dir(SimpleNamespace(backbone="ettin", smoke=False))
    assert ettin.name == "lab-data-lora-ettin"
    assert lc.out_dir(SimpleNamespace(backbone="ettin", smoke=True)).name == "smoke"


def test_stratified_subset_keeps_classes_and_is_deterministic() -> None:
    from types import SimpleNamespace

    rows = [SimpleNamespace(label=c) for c in ("safe", "warn", "crit") * 10]
    got = lc.stratified_subset(rows, 7)
    assert len(got) == 7
    assert {r.label for r in got} == {"safe", "warn", "crit"}
    assert got == lc.stratified_subset(rows, 7)
    assert lc.stratified_subset(rows, 99) == rows

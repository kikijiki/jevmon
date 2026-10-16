"""Shipped frozen-backbone heads in lab/heads load and carry their provenance metadata."""

import json
from pathlib import Path

import pytest
from safetensors import safe_open

from jevmon.laya.head import TunedHead

HEADS = Path(__file__).resolve().parents[1] / "heads"
KEYS = (
    "backbone",
    "layer",
    "recipe",
    "train_rows",
    "train_families",
    "text_format",
    "heldout_rows",
    "pool_macro_f1",
    "date",
)
FILES = sorted(HEADS.glob("*.safetensors"))


def test_three_heads_are_shipped() -> None:
    assert {f.stem for f in FILES} == {"foundation-sec-8b", "qwen3e-8b", "ettin-1b"}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_head_loads_with_metadata(path: Path) -> None:
    head = TunedHead.load(path)
    assert head.labels == ("safe", "warn", "crit")
    assert head.weight.shape[0] == 3
    with safe_open(str(path), framework="numpy") as fh:
        meta = fh.metadata()
    assert all(meta.get(k) for k in KEYS), meta
    assert json.loads(meta["labels"]) == ["safe", "warn", "crit"]
    assert 0.5 < float(meta["pool_macro_f1"]) < 1.0
    assert meta["heldout_rows"] == "420"

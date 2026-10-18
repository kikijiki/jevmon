"""The shipped judge heads: load, provenance, and reproduce the held-out numbers.

`lab/scripts/train_heads.py export` writes the three heads into `app/src/jevmon/laya/heads/`. The
reproduction test scores the packaged head over the cached lab/cache/e20 features and checks the
held-out macro-F1 matches the candidate already scored in lab-data-e20/results.json. It is skipped
when the (gitignored) results and caches are not present.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from jevmon.laya.head import TunedHead

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_heads as th

RESULTS = th.OUT_DIR / "results.json"
NAMES = ("laya", "ettin", "qwen3e")
# The shipped files: the three frozen-backbone heads plus the two LoRA judges (adapter + head).
PACKAGED = set(NAMES) | {"laya-lora", "ettin-lora"}


def test_packaged_heads_are_the_three_models() -> None:
    assert {p.stem for p in th.APP_HEADS.glob("*.safetensors")} == PACKAGED
    for name in NAMES:
        parts, spec, _config, backbone = th.HEADS[name]
        head = TunedHead.load(name)
        assert head.labels == ("safe", "warn", "crit")
        assert head.feature_variants == tuple(variant for _, variant in parts)
        assert head.backbone == backbone and head.recipe == spec.name()
        assert "mixed pool" in head.training_set and head.max_length == 512
        assert dict(head.metrics).get("lab/pool") is not None


def _cache(name: str) -> bool:
    """At least one of the two cache paths `train_heads.store` merges for a backbone."""
    return (th.ROOT / f"lab/cache/{name}.pkl").exists() or (
        th.ROOT / f"lab/cache/e20/{name}.pkl"
    ).exists()


def _artefacts_present() -> bool:
    needed = [RESULTS, th.ROOT / th.GEN2]
    needed += [th.ROOT / d for d in th.EXT_DIRS.values()]
    needed += [th.ARCHIVE / d for d in (*th.LAB, th.SETTLE_OLD, th.SETTLE_NEW)]
    if not all(p.exists() for p in needed):
        return False
    # The embeddings the reproduction scores over: Laya plus both HF backbones (gitignored).
    laya = _cache("emb-english-mean-512") or _cache("e20/laya-mean-512")
    backbones = all(
        _cache(f"bb-{name}-512") or _cache(f"e20/bb-{name}-512")
        for name in ("ettin-1b", "qwen3e-8b")
    )
    return laya and backbones


@pytest.mark.skipif(not _artefacts_present(), reason="results/caches not present")
def test_exported_heads_reproduce_results_json() -> None:
    results = json.loads(RESULTS.read_text())
    groups = th.eval_groups()
    for name in NAMES:
        parts, _spec, _config, _backbone = th.HEADS[name]
        head = TunedHead.load(name)
        candidate = th.head_key(name)
        stored = dict(head.metrics)
        for group in groups:
            if not group.labelled:
                continue
            reference = (
                results["sets"]
                .get(group.name, {})
                .get("judges", {})
                .get(candidate, {})
                .get("macro_f1")
            )
            if reference is None:
                continue
            texts = [row.text for row in group.rows]
            predictions = head.probabilities(th.matrix(parts, texts)).argmax(1)
            y = np.asarray([th.LABELS.index(row.label) for row in group.rows])
            family = np.asarray([row.family for row in group.rows])
            scored = th.summarize(y, predictions, family)["macro_f1"]
            assert abs(scored - reference) < 5e-4, (name, group.name, scored, reference)
            assert abs(stored[group.name] - reference) < 5e-4, (name, group.name)

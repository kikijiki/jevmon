"""The Hugging Face backbone judges: config wiring, the unavailable path, and feature parity.

The parity test compares the app's frozen features for a few lab texts against the cached
`lab/cache/e20/bb-*-512.pkl` features. It needs a GPU and the model in the Hugging Face cache
(`HF_HOME` must point at the hub, e.g. `HF_HOME=~/.cache/huggingface`), so it is skipped in the
plain `just check` CPU run and meant to be driven from the GPU env.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from jevmon.laya.backbone import MODELS, BackboneConfig, BackboneDecider, embed_variants
from jevmon.laya.decider import LayaDecider

CATEGORIES = {"safe": "a", "warn": "b", "crit": "c"}
# The lab cache names each backbone store by its E20 feature name, not the judge model key.
CACHE_NAMES = {"ettin": "ettin-1b", "qwen3e": "qwen3e-8b"}


def _config(key: str, **overrides) -> BackboneConfig:
    overrides.setdefault("categories", CATEGORIES)
    return BackboneConfig(key=key, **overrides)


def test_build_decider_selects_the_backbone(tmp_path) -> None:
    from jevmon.config import load_config
    from jevmon.monitor import build_decider

    path = tmp_path / "config.toml"
    path.write_text('[judge]\nmodel = "ettin"\n')
    decider = build_decider(load_config(path))
    assert isinstance(decider, BackboneDecider)
    assert decider.spec.model_id == "jhu-clsp/ettin-encoder-1b"
    assert (decider.config.device, decider.config.dtype) == ("auto", "auto")
    assert isinstance(build_decider(load_config()), LayaDecider)


def test_unknown_model_is_a_config_error(tmp_path) -> None:
    from jevmon.config import ConfigError, load_config

    path = tmp_path / "config.toml"
    path.write_text('[judge]\nmodel = "gpt"\n')
    with pytest.raises(ConfigError, match=r"\[judge\] model must be"):
        load_config(path)


def test_broken_head_makes_the_backbone_unavailable_without_fallback(tmp_path) -> None:
    decider = BackboneDecider(_config("ettin", head=str(tmp_path / "missing.safetensors")))
    decider.load()  # the packaged model is never downloaded: the head fails first
    assert decider.ready and decider.error
    assert "ettin unavailable" in decider.status
    assert not decider.decide("state").available


def test_head_for_another_backbone_is_refused(tmp_path) -> None:
    from jevmon.laya.head import TunedHead

    path = tmp_path / "bb.safetensors"
    TunedHead(
        labels=("safe", "warn", "crit"),
        weight=np.zeros((3, 4), dtype=np.float32),
        bias=np.zeros(3, dtype=np.float32),
        backbone="some/other-model",
        variants=("mean",),
    ).save(path)
    decider = BackboneDecider(_config("ettin", head=str(path)))
    decider.load()
    assert "backbone" in (decider.error or "")
    assert not decider.decide_batch(["a", "b"])[0].available


def test_reconfigure_keeps_the_model_and_adopts_categories() -> None:
    decider = BackboneDecider(_config("ettin"))
    decider.reconfigure(_config("ettin", categories={"safe": "x", "warn": "y", "crit": "z"}))
    assert decider.config.categories == {"safe": "x", "warn": "y", "crit": "z"}


def test_monitor_status_surfaces_the_backbone_and_its_error(tmp_path) -> None:
    from jevmon.config import load_config
    from jevmon.monitor import Monitor, build_decider

    path = tmp_path / "config.toml"
    broken = tmp_path / "missing.safetensors"
    path.write_text(f'[judge]\nmodel = "ettin"\ntuned = "{broken}"\n')
    cfg = load_config(path)
    decider = build_decider(cfg)
    decider.load()
    status = Monitor(cfg, None, decider=decider).status()
    assert status["model_status"] == decider.status
    assert "ettin" in status["model_status"] and "unavailable" in status["model_status"]


# -- feature parity against the lab cache -----------------------------------------------------

HF_HOME = os.environ.get("HF_HOME")


def _cached(model_id: str) -> bool:
    if not HF_HOME:
        return False
    return (Path(HF_HOME) / "hub" / ("models--" + model_id.replace("/", "--"))).is_dir()


def _require_gpu_and_cache(key: str) -> None:
    if not HF_HOME:
        pytest.skip("HF_HOME not set: model cache not reachable")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU: the cached features are bf16-on-GPU")
    if not _cached(MODELS[key].model_id):
        pytest.skip(f"{MODELS[key].model_id} not in the Hugging Face cache")


@pytest.mark.slow
@pytest.mark.parametrize("key", ["ettin", "qwen3e"])
def test_app_features_match_the_lab_cache(key: str) -> None:
    _require_gpu_and_cache(key)
    import pickle

    cache = Path("lab/cache/e20") / f"bb-{CACHE_NAMES[key]}-512.pkl"
    if not cache.exists():
        pytest.skip(f"{cache} not present")
    store = pickle.loads(cache.read_bytes())
    texts = sorted(store)[:5]
    decider = BackboneDecider(_config(key, device="gpu"))
    decider.load()
    assert decider.ready and decider.error is None, decider.status
    head = decider._head
    feats = embed_variants(
        decider._tok,
        decider._model,
        texts,
        kind=decider.spec.kind,
        variants=head.feature_variants,
        max_length=head.max_length,
        device=decider._device,
        batch_size=8,
    )
    for variant in head.feature_variants:
        got = feats[variant]
        want = np.stack([store[t][variant] for t in texts]).astype(np.float32)
        assert got.shape == want.shape
        cosine = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
        assert cosine.min() > 0.999, (key, variant, cosine.min())
        # The cache is float16 of a bf16 forward: component-wise agreement is only meaningful
        # relative to the vector norm (small components are cancellation noise from summing
        # bf16 values), so compare the relative L2 error.
        relative = np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)
        assert relative.max() < 0.02, (key, variant, float(relative.max()))


def test_ettin_lora_is_a_selectable_backbone_with_a_bundled_head() -> None:
    from jevmon.detect.judge import JudgeSettings
    from jevmon.laya.head import LoraHead
    from jevmon.monitor import backbone_config

    assert MODELS["ettin-lora"].model_id == MODELS["ettin"].model_id
    assert MODELS["ettin-lora"].lora == "ettin-lora"
    config = backbone_config(JudgeSettings(model="ettin-lora"))
    assert config.key == "ettin-lora" and config.head == ""
    assert backbone_config(JudgeSettings(model="ettin")).head == "ettin"
    lora = LoraHead.load("ettin-lora")
    assert lora.head.backbone == MODELS["ettin-lora"].model_id
    assert lora.head.labels == ("safe", "warn", "crit")

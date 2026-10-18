"""The shipped LoRA adapter (`LoraHead`): save/load, the in-memory merge maths, and wiring.

No peft and no real weights: the merge is exercised on a tiny synthetic module, and the decider
merge path runs against a fake agent whose `model.encoder` is a tiny module tree.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from jevmon.laya.decider import LayaConfig, LayaDecider
from jevmon.laya.head import HeadLoadError, LoraHead, TunedHead, merge_lora_

LABELS = ("safe", "warn", "crit")
CATEGORIES = {"safe": "a", "warn": "b", "crit": "c"}


class TinyEncoder:
    """A module tree whose names match an adapter key (`layers.0.attn.Wo`); needs torch."""

    def __init__(self, in_dim: int = 4, out_dim: int = 4) -> None:
        torch = pytest.importorskip("torch")

        class Root(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [
                        torch.nn.ModuleDict(
                            {
                                "attn": torch.nn.ModuleDict(
                                    {"Wo": torch.nn.Linear(in_dim, out_dim, bias=False)}
                                )
                            }
                        )
                    ]
                )

        self.module = Root()

    def named_modules(self):
        return self.module.named_modules()

    def weight(self):
        return self.module.layers[0].attn["Wo"].weight


def head(dim: int = 4) -> TunedHead:
    return TunedHead(
        labels=LABELS,
        weight=np.zeros((3, dim), dtype=np.float32),
        bias=np.zeros(3, dtype=np.float32),
    )


def lora(adapter, rank: int = 2, alpha: float = 4.0) -> LoraHead:
    return LoraHead(head=head(), rank=rank, alpha=alpha, target_modules=("Wo",), adapter=adapter)


def test_merge_lora_applies_scale_times_b_at_a() -> None:
    torch = pytest.importorskip("torch")
    encoder = TinyEncoder()
    before = encoder.weight().detach().clone()
    a = np.arange(8, dtype=np.float32).reshape(2, 4) / 8.0
    b = np.arange(8, dtype=np.float32).reshape(4, 2) / 8.0
    merged = lora((("layers.0.attn.Wo", a, b),))
    assert merged.scale == 2.0
    assert merge_lora_(encoder, merged) == 1
    want = before + merged.scale * (torch.from_numpy(b) @ torch.from_numpy(a))
    assert torch.allclose(encoder.weight().detach(), want, atol=1e-6)


def test_merge_lora_rejects_a_missing_target() -> None:
    pytest.importorskip("torch")
    encoder = TinyEncoder()
    a = np.zeros((2, 4), dtype=np.float32)
    b = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(HeadLoadError, match="not found"):
        merge_lora_(encoder, lora((("layers.9.attn.Wo", a, b),)))


def test_lora_head_save_load_round_trip(tmp_path) -> None:
    a = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((4, 2)).astype(np.float32)
    original = LoraHead(
        head=TunedHead(
            labels=LABELS,
            weight=np.zeros((3, 4), dtype=np.float32),
            bias=np.zeros(3, dtype=np.float32),
            recipe="lora/r2/lr0.0003/e6/seed0",
            training_set="mixed pool",
            metrics=(("lab/pool", 0.5),),
        ),
        rank=2,
        alpha=4.0,
        target_modules=("Wo",),
        revision="abc123",
        adapter=(("layers.0.attn.Wo", a, b),),
        per_seed=(("lab/pool", (0.4, 0.5, 0.6)),),
    )
    path = tmp_path / "laya-lora.safetensors"
    original.save(path)
    loaded = LoraHead.load(path)
    assert loaded.rank == 2 and loaded.alpha == 4.0 and loaded.scale == 2.0
    assert loaded.target_modules == ("Wo",) and loaded.revision == "abc123"
    assert loaded.head.pooling == "mean" and loaded.head.labels == LABELS
    assert dict(loaded.head.metrics) == {"lab/pool": 0.5}
    assert dict(loaded.per_seed) == {"lab/pool": (0.4, 0.5, 0.6)}
    assert len(loaded.adapter) == 1
    np.testing.assert_array_equal(loaded.adapter[0][1], a)
    np.testing.assert_array_equal(loaded.adapter[0][2], b)


def test_lora_head_refuses_a_plain_head_file(tmp_path) -> None:
    path = tmp_path / "head.safetensors"
    head().save(path)
    with pytest.raises(HeadLoadError, match="not a lora head"):
        LoraHead.load(path)


def test_lora_head_refuses_bad_rank() -> None:
    a = np.zeros((2, 4), dtype=np.float32)
    b = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(HeadLoadError, match="rank/alpha"):
        LoraHead(head=head(), rank=0, alpha=4.0, adapter=(("layers.0.attn.Wo", a, b),))
    with pytest.raises(HeadLoadError, match="rank"):
        LoraHead(head=head(), rank=3, alpha=4.0, adapter=(("layers.0.attn.Wo", a, b),))


def test_build_decider_routes_laya_lora_to_the_laya_decider(tmp_path) -> None:
    from jevmon.config import load_config
    from jevmon.monitor import build_decider

    path = tmp_path / "config.toml"
    path.write_text('[judge]\nmodel = "laya-lora"\n')
    decider = build_decider(load_config(path))
    assert isinstance(decider, LayaDecider)
    assert decider.config.lora == "laya-lora" and decider.config.tuned == ""
    # The default `laya` keeps no adapter and its packaged frozen head.
    default = build_decider(load_config())
    assert isinstance(default, LayaDecider) and default.config.lora == ""
    assert default.config.tuned == "laya"


class FakeRouter:
    def __init__(self, encoder: TinyEncoder) -> None:
        self._agents = {"english": SimpleNamespace(model=SimpleNamespace(encoder=encoder))}


def test_decider_merges_the_shipped_lora_adapter(tmp_path) -> None:
    pytest.importorskip("torch")
    a = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((4, 2)).astype(np.float32)
    path = tmp_path / "laya-lora.safetensors"
    lora((("layers.0.attn.Wo", a, b),)).save(path)
    encoder = TinyEncoder()
    before = encoder.weight().detach().clone()
    decider = LayaDecider(LayaConfig(lora=str(path), categories=CATEGORIES))
    decider._router = FakeRouter(encoder)
    decider._load_lora()
    assert decider._lora_note is None and decider._head is not None
    assert set(decider._head.labels) == set(CATEGORIES)
    assert not bool((encoder.weight().detach() == before).all())  # weights changed in place


def test_a_broken_lora_adapter_degrades_to_zero_shot(tmp_path) -> None:
    # A missing file fails in `LoraHead.load`, before any encoder is touched (no torch needed).
    decider = LayaDecider(
        LayaConfig(lora=str(tmp_path / "missing.safetensors"), categories=CATEGORIES)
    )
    decider._load_lora()
    assert decider._head is None and decider._embed is None
    assert decider._lora_note and "lora adapter failed" in decider._lora_note

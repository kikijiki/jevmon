"""Fine-tuned head: save/load round trip, softmax shape and the guards against a bad file."""

import numpy as np
import pytest

from jevmon.laya.head import HeadLoadError, TunedHead

LABELS = ("safe", "warn", "crit")


def head(bias: tuple[float, ...] = (0.0, 0.0, 9.0), dim: int = 4) -> TunedHead:
    return TunedHead(
        labels=LABELS,
        weight=np.zeros((3, dim), dtype=np.float32),
        bias=np.asarray(bias, dtype=np.float32),
    )


def test_save_and_load_round_trip(tmp_path) -> None:
    path = tmp_path / "head.safetensors"
    head().save(path)
    loaded = TunedHead.load(path)
    assert loaded.labels == LABELS and loaded.max_length == 512
    np.testing.assert_array_equal(loaded.weight, np.zeros((3, 4), dtype=np.float32))
    np.testing.assert_allclose(loaded.bias, np.asarray((0.0, 0.0, 9.0)))


def test_probabilities_are_a_softmax_over_the_labels() -> None:
    probs = head().probabilities(np.ones((3, 4), dtype=np.float32))
    assert probs.shape == (3, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert LABELS[int(probs[0].argmax())] == "crit"
    assert probs[0, 2] > 0.99


def test_dimension_mismatch_is_refused() -> None:
    with pytest.raises(HeadLoadError, match="embedding dim"):
        head().probabilities(np.ones((2, 7), dtype=np.float32))


def test_shape_and_finiteness_guards() -> None:
    with pytest.raises(HeadLoadError, match="weight shape"):
        TunedHead(labels=LABELS, weight=np.zeros((4, 4)), bias=np.zeros(3))
    with pytest.raises(HeadLoadError, match="bias shape"):
        TunedHead(labels=LABELS, weight=np.zeros((3, 4)), bias=np.zeros(5))
    with pytest.raises(HeadLoadError, match="non-finite"):
        TunedHead(
            labels=LABELS,
            weight=np.full((3, 4), np.nan, dtype=np.float32),
            bias=np.zeros(3, dtype=np.float32),
        )


def test_load_reports_missing_and_malformed_files(tmp_path) -> None:
    with pytest.raises(HeadLoadError, match="no such head"):
        TunedHead.load(tmp_path / "missing.safetensors")
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    with pytest.raises(HeadLoadError):
        TunedHead.load(broken)


def test_shipped_default_head_loads_with_the_three_classes() -> None:
    from jevmon.config import load_config

    name = load_config().judge.tuned
    loaded = TunedHead.load(name)
    assert set(loaded.labels) == {"safe", "warn", "crit"}
    assert loaded.weight.shape[0] == 3 and np.isfinite(loaded.weight).all()


def test_wrong_format_version_and_missing_metadata_are_refused(tmp_path) -> None:
    from safetensors.numpy import save_file

    tensors = {"weight": np.zeros((3, 4), np.float32), "bias": np.zeros(3, np.float32)}
    old = tmp_path / "old.safetensors"
    save_file(tensors, str(old), metadata={"format_version": "0"})
    with pytest.raises(HeadLoadError, match="format_version"):
        TunedHead.load(old)
    bare = tmp_path / "bare.safetensors"
    save_file(tensors, str(bare), metadata={"format_version": "1"})
    with pytest.raises(HeadLoadError, match="lacks"):
        TunedHead.load(bare)

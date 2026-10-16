"""Fine-tuned classification head over frozen Laya encoder embeddings.

A head is one linear map from a pooling of the encoder's last hidden state to the judge's
categories plus a softmax. Weights are trained offline from labelled lab contexts
(`lab/scripts/finetune_head`) and selected with `[judge] tuned` (a path, empty by default = the
zero-shot model decides). Inference is a matrix multiply, so the head never leaves the judge's
path: `LayaDecider` feeds its output through the same `_decision` the zero-shot path uses,
with the same confidence bands and the same `Decision` type.

`embed_states` is the one embedder for both training and serving - a head trained on one
pooling scores with the same pooling, recorded in the file - so what was fitted is what runs.

Nothing process-specific lives here: the file (safetensors) is embedding-shaped weights plus the
category names and pooling they were trained for, and `TunedHead.load` refuses a head whose categories
do not match the configured ones.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# Poolings of the encoder's last hidden state. `mean` is laya's own `embed_fn_from_agent`
# (padding excluded); `mean+cls` keeps the whole vector for a linear head to weigh.
POOLINGS = ("mean", "cls", "mean+cls")
FORMAT_VERSION = "1"  # safetensors metadata key `format_version`


HEADS_DIR = Path(__file__).parent / "heads"  # heads shipped in the package (tiny, tracked)
DEFAULT_HEAD = "lab-head-v4"


def resolve_head(spec: str | Path) -> Path:
    """A bare name (no directory, no suffix) is a head shipped in `HEADS_DIR`; else a path."""
    text = str(spec)
    if "/" not in text and "\\" not in text and not Path(text).suffix:
        return HEADS_DIR / f"{text}.safetensors"
    return Path(text)


class HeadLoadError(ValueError):
    """The weights file is missing, malformed or trained for different categories."""


@dataclass(frozen=True, slots=True)
class TunedHead:
    labels: tuple[str, ...]  # category names, in the column order of the weights
    weight: np.ndarray  # (classes, dim)
    bias: np.ndarray  # (classes,)
    max_length: int = 512  # tokenizer budget the embeddings were trained with
    pooling: str = "mean"  # how the embeddings were pooled (see POOLINGS)

    def __post_init__(self) -> None:
        if self.weight.ndim != 2 or self.weight.shape[0] != len(self.labels):
            raise HeadLoadError(
                f"weight shape {self.weight.shape} does not match {len(self.labels)} labels"
            )
        if self.bias.shape != (len(self.labels),):
            raise HeadLoadError(f"bias shape {self.bias.shape} != ({len(self.labels)},)")
        if not self.labels:
            raise HeadLoadError("head has no labels")
        if self.pooling not in POOLINGS:
            raise HeadLoadError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        if not np.isfinite(self.weight).all() or not np.isfinite(self.bias).all():
            raise HeadLoadError("head weights contain non-finite values")

    def probabilities(self, vectors: np.ndarray) -> np.ndarray:
        """(n, classes) softmax over `vectors @ weight.T + bias`, rows summing to 1."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.shape[1] != self.weight.shape[1]:
            raise HeadLoadError(
                f"embedding dim {vectors.shape[1]} != head dim {self.weight.shape[1]}"
            )
        logits = vectors @ self.weight.T + self.bias
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    @classmethod
    def load(cls, path: str | Path) -> "TunedHead":
        """Read a `.safetensors` written by `save` (numpy only, no torch).

        `path` may be a shipped head's bare name (see `resolve_head`).
        """
        file = resolve_head(path)
        if not file.exists():
            raise HeadLoadError(f"no such head: {file}")
        try:
            with safe_open(str(file), framework="numpy") as fh:
                meta = fh.metadata() or {}
                if meta.get("format_version") != FORMAT_VERSION:
                    raise HeadLoadError(
                        f"unsupported head format_version {meta.get('format_version')!r}"
                        f" (expected {FORMAT_VERSION!r})"
                    )
                missing = [k for k in ("labels", "pooling", "max_length") if k not in meta]
                if missing:
                    raise HeadLoadError(f"head metadata lacks {missing}")
                if set(fh.keys()) != {"weight", "bias"}:
                    raise HeadLoadError(f"unexpected tensors {sorted(fh.keys())}")
                weight, bias = fh.get_tensor("weight"), fh.get_tensor("bias")
            labels = json.loads(meta["labels"])
            if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
                raise HeadLoadError("labels must be a JSON list of strings")
            if weight.dtype != np.float32 or bias.dtype != np.float32:
                raise HeadLoadError("weight and bias must be float32")
            return cls(
                labels=tuple(labels),
                weight=weight,
                bias=bias,
                max_length=int(meta["max_length"]),
                pooling=meta["pooling"],
            )
        except HeadLoadError:
            raise
        except Exception as exc:
            raise HeadLoadError(f"{file}: {type(exc).__name__}: {exc}") from exc

    def save(self, path: str | Path) -> None:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "weight": np.ascontiguousarray(self.weight, dtype=np.float32),
                "bias": np.ascontiguousarray(self.bias, dtype=np.float32),
            },
            str(file),
            metadata={
                "format_version": FORMAT_VERSION,
                "labels": json.dumps(list(self.labels)),
                "pooling": self.pooling,
                "max_length": str(self.max_length),
            },
        )


def embed_states(
    agent,
    states: Sequence[str],
    *,
    pooling: str = "mean",
    max_length: int = 512,
    batch_size: int = 32,
) -> np.ndarray:
    """(n, dim) embeddings of `states` from the frozen encoder already on `agent`.

    Same maths as laya's `embed_fn_from_agent` (padding excluded from the mean, inference mode,
    current `agent.device`) with the pooling choice the head records: `mean`, the `cls` token,
    or both concatenated. Training and serving both call this, so a head never scores with a
    pooling it was not fitted on.
    """
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    import torch
    from laya.common import encode_text

    rows = ["" if state is None else str(state) for state in states]
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    tok, encoder = agent.tok, agent.model.encoder
    device = agent.device
    parts: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        encoded = encode_text(
            tok,
            rows[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            if pooling == "mean":
                pooled = mean
            elif pooling == "cls":
                pooled = hidden[:, 0]
            else:
                pooled = torch.cat([mean, hidden[:, 0]], dim=-1)
        parts.append(pooled.float().cpu().numpy())
    return np.concatenate(parts, axis=0)


def make_embedder(
    agent,
    pooling: str = "mean",
    max_length: int = 512,
    batch_size: int = 32,
) -> Callable[[Sequence[str]], np.ndarray]:
    """A `states -> (n, dim)` callable for `embed_states` (the shape the judge needs)."""

    def embedder(states: Sequence[str]) -> np.ndarray:
        return embed_states(
            agent, states, pooling=pooling, max_length=max_length, batch_size=batch_size
        )

    return embedder

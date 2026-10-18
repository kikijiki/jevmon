"""Fine-tuned classification head over frozen Laya encoder embeddings.

A head is one linear map from a pooling of the encoder's last hidden state to the judge's
categories plus a softmax. Weights are trained offline from the mixed labelled pool
(`lab/scripts/train_heads.py`) and selected with `[judge] tuned` (a path, empty by default = the
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
# Feature variants a shipped head may name, one column per variant concatenated in order.
# `lab/scripts/backbone_embed.py` computes these per backbone kind: decoders give `mean`,
# `mean75` (mean of the layer at 3/4 depth), `last` and `last75`; encoders give `mean`, `mean75`
# and `cls`. A Laya head only ever uses `mean` (`pooling`); a backbone head records the
# concatenation it was fitted on in `variants`.
FEATURE_VARIANTS = ("mean", "cls", "last", "mean75", "last75")
_POOLING_VARIANTS = {"mean": ("mean",), "cls": ("cls",), "mean+cls": ("mean", "cls")}
FORMAT_VERSION = "1"  # safetensors metadata key `format_version`
# Packaged LoRA files (a `LoraHead`): a separate `kind`, same metadata conventions as a head.
LORA_FORMAT_VERSION = "1"
LORA_KIND = "lora"


HEADS_DIR = Path(__file__).parent / "heads"  # heads shipped in the package (tiny, tracked)
DEFAULT_HEAD = "laya"  # the packaged head for `[judge] model = "laya"`
DEFAULT_LORA_HEAD = "laya-lora"  # the packaged adapter+head for `[judge] model = "laya-lora"`


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
    pooling: str = "mean"  # how the embeddings were pooled (see POOLINGS); laya heads only
    # HF id of the backbone the head embeds with ("" = the local Laya encoder). A backbone head
    # concatenates `variants` in order into the weight's input dimension.
    backbone: str = ""
    variants: tuple[str, ...] = ()  # feature columns, in concatenation order (see FEATURE_VARIANTS)
    recipe: str = ""  # the fit recipe, e.g. "logreg/wd0.001/bal"
    training_set: str = ""  # plain description of the train pool the head was fitted on
    metrics: tuple[tuple[str, float], ...] = ()  # (eval set, macro-F1) on held-out sets

    def __post_init__(self) -> None:
        if self.weight.ndim != 2 or self.weight.shape[0] != len(self.labels):
            raise HeadLoadError(
                f"weight shape {self.weight.shape} does not match {len(self.labels)} labels"
            )
        if self.bias.shape != (len(self.labels),):
            raise HeadLoadError(f"bias shape {self.bias.shape} != ({len(self.labels)},)")
        if not self.labels:
            raise HeadLoadError("head has no labels")
        if self.pooling not in POOLINGS and not self.variants:
            raise HeadLoadError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        for variant in self.variants:
            if variant not in FEATURE_VARIANTS:
                raise HeadLoadError(
                    f"unknown feature variant {variant!r}: expected one of {FEATURE_VARIANTS}"
                )
        if not np.isfinite(self.weight).all() or not np.isfinite(self.bias).all():
            raise HeadLoadError("head weights contain non-finite values")

    @property
    def feature_variants(self) -> tuple[str, ...]:
        """The feature columns the head reads, in order (a legacy `pooling` names its single one)."""
        return self.variants or _POOLING_VARIANTS.get(self.pooling, (self.pooling,))

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
                missing = [k for k in ("labels", "max_length") if k not in meta]
                if not meta.get("pooling") and not meta.get("variants"):
                    missing.append("pooling")
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
            variants = tuple(json.loads(meta["variants"])) if meta.get("variants") else ()
            if not all(isinstance(x, str) for x in variants):
                raise HeadLoadError("variants must be a JSON list of strings")
            metrics_raw = json.loads(meta["metrics"]) if meta.get("metrics") else {}
            if not isinstance(metrics_raw, dict):
                raise HeadLoadError("metrics must be a JSON object")
            metrics = tuple((str(k), float(v)) for k, v in metrics_raw.items())
            pooling = meta.get("pooling") or "+".join(variants) or "mean"
            return cls(
                labels=tuple(labels),
                weight=weight,
                bias=bias,
                max_length=int(meta["max_length"]),
                pooling=pooling,
                backbone=str(meta.get("backbone", "")),
                variants=variants,
                recipe=str(meta.get("recipe", "")),
                training_set=str(meta.get("training_set", "")),
                metrics=tuple(sorted(metrics)),
            )
        except HeadLoadError:
            raise
        except Exception as exc:
            raise HeadLoadError(f"{file}: {type(exc).__name__}: {exc}") from exc

    def save(self, path: str | Path) -> None:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        variants = self.feature_variants
        save_file(
            {
                "weight": np.ascontiguousarray(self.weight, dtype=np.float32),
                "bias": np.ascontiguousarray(self.bias, dtype=np.float32),
            },
            str(file),
            metadata={
                "format_version": FORMAT_VERSION,
                "labels": json.dumps(list(self.labels)),
                # pooling records the laya convention; variants is the full concatenation.
                "pooling": self.pooling if not self.variants else "+".join(variants),
                "variants": json.dumps(list(variants)),
                "max_length": str(self.max_length),
                "backbone": self.backbone,
                "recipe": self.recipe,
                "training_set": self.training_set,
                "metrics": json.dumps(dict(self.metrics)),
            },
        )


@dataclass(frozen=True, slots=True)
class LoraHead:
    """A LoRA adapter over the Laya encoder plus its classifier head, merged at load time.

    One packaged `.safetensors`: the classifier `weight`/`bias` (as in `TunedHead`), and one
    `.lora_A` / `.lora_B` pair per target module (PEFT's `Linear` LoRA layout: `A` is (rank, in),
    `B` is (out, rank)). `merge_lora_` folds `scale * B @ A` into the encoder weights in memory,
    so serving never imports `peft` and the merged model costs what the frozen encoder costs.

    Metadata: `rank`, `alpha` (scaling is `alpha / rank`), `target_modules`, the Laya `revision`
    the adapter was trained against, `pooling`/`max_length` (as in a head), `labels`, and
    `metrics` / `per_seed` — per-seed macro-F1 per eval set, the mean in `metrics` and the seed
    values in `per_seed`.
    """

    head: TunedHead  # the classifier, and the pooling to read after the merge
    rank: int
    alpha: float
    target_modules: tuple[str, ...] = ()
    revision: str = ""  # the Laya revision the adapter was trained against
    adapter: tuple[tuple[str, np.ndarray, np.ndarray], ...] = ()  # (module, A, B) per target
    per_seed: tuple[tuple[str, tuple[float, ...]], ...] = ()  # (eval set, per-seed macro-F1)

    def __post_init__(self) -> None:
        if self.rank <= 0 or not np.isfinite(self.alpha) or self.alpha <= 0:
            raise HeadLoadError(f"lora rank/alpha must be positive, got {self.rank}/{self.alpha}")
        if not self.adapter:
            raise HeadLoadError("lora file has no adapter tensors")
        for name, a, b in self.adapter:
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != self.rank or b.shape[1] != self.rank:
                raise HeadLoadError(f"lora tensors for {name!r} do not match rank {self.rank}")

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def load(cls, path: str | Path) -> "LoraHead":
        file = resolve_head(path)
        if not file.exists():
            raise HeadLoadError(f"no such lora head: {file}")
        try:
            with safe_open(str(file), framework="numpy") as fh:
                meta = fh.metadata() or {}
                if meta.get("format_version") != LORA_FORMAT_VERSION:
                    raise HeadLoadError(
                        f"unsupported lora format_version {meta.get('format_version')!r}"
                        f" (expected {LORA_FORMAT_VERSION!r})"
                    )
                if meta.get("kind") != LORA_KIND:
                    raise HeadLoadError(f"not a lora head (kind {meta.get('kind')!r})")
                missing = [k for k in ("labels", "rank", "alpha", "pooling") if k not in meta]
                if missing:
                    raise HeadLoadError(f"lora metadata lacks {missing}")
                keys = set(fh.keys())
                if not {"weight", "bias"} <= keys:
                    raise HeadLoadError("lora file lacks the classifier weight/bias")
                adapter = []
                for key in sorted(k for k in keys if k.endswith(".lora_A")):
                    module = key[: -len(".lora_A")]
                    if module + ".lora_B" not in keys:
                        raise HeadLoadError(f"lora target {module!r} has no lora_B")
                    adapter.append((module, fh.get_tensor(key), fh.get_tensor(module + ".lora_B")))
                if not adapter:
                    raise HeadLoadError("lora file has no *.lora_A tensors")
                weight, bias = fh.get_tensor("weight"), fh.get_tensor("bias")
            labels = json.loads(meta["labels"])
            variants = tuple(json.loads(meta["variants"])) if meta.get("variants") else ()
            metrics = json.loads(meta["metrics"]) if meta.get("metrics") else {}
            per_seed = json.loads(meta["per_seed"]) if meta.get("per_seed") else {}
            targets = (
                tuple(json.loads(meta["target_modules"])) if meta.get("target_modules") else ()
            )
            head = TunedHead(
                labels=tuple(labels),
                weight=weight,
                bias=bias,
                max_length=int(meta.get("max_length", 512)),
                pooling=meta.get("pooling", "mean"),
                backbone=str(meta.get("backbone", "")),
                variants=variants,
                recipe=str(meta.get("recipe", "")),
                training_set=str(meta.get("training_set", "")),
                metrics=tuple((str(k), float(v)) for k, v in metrics.items()),
            )
            return cls(
                head=head,
                rank=int(meta["rank"]),
                alpha=float(meta["alpha"]),
                target_modules=targets,
                revision=str(meta.get("revision", "")),
                adapter=tuple(adapter),
                per_seed=tuple((str(k), tuple(float(x) for x in v)) for k, v in per_seed.items()),
            )
        except HeadLoadError:
            raise
        except Exception as exc:
            raise HeadLoadError(f"{file}: {type(exc).__name__}: {exc}") from exc

    def save(self, path: str | Path) -> None:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, np.ndarray] = {
            "weight": np.ascontiguousarray(self.head.weight, dtype=np.float32),
            "bias": np.ascontiguousarray(self.head.bias, dtype=np.float32),
        }
        for module, a, b in self.adapter:
            tensors[f"{module}.lora_A"] = np.ascontiguousarray(a, dtype=np.float32)
            tensors[f"{module}.lora_B"] = np.ascontiguousarray(b, dtype=np.float32)
        save_file(
            tensors,
            str(file),
            metadata={
                "format_version": LORA_FORMAT_VERSION,
                "kind": LORA_KIND,
                "labels": json.dumps(list(self.head.labels)),
                "pooling": self.head.pooling,
                "variants": json.dumps(list(self.head.feature_variants)),
                "max_length": str(self.head.max_length),
                "backbone": self.head.backbone,
                "rank": str(self.rank),
                "alpha": repr(float(self.alpha)),
                "target_modules": json.dumps(list(self.target_modules)),
                "revision": self.revision,
                "recipe": self.head.recipe,
                "training_set": self.head.training_set,
                "metrics": json.dumps(dict(self.head.metrics)),
                "per_seed": json.dumps({k: list(v) for k, v in self.per_seed}),
            },
        )


def merge_lora_(encoder, lora: LoraHead) -> int:
    """Merge `lora` into `encoder` in place (`W += scale * B @ A`, fp32 then cast); module count.

    `encoder` is any module tree whose `named_modules()` names match the adapter keys (the Laya
    `ModernBertModel`, e.g. `layers.0.attn.Wo`). The delta is computed in fp32 on the weight's
    device and cast back to the weight dtype, so a bf16 encoder keeps bf16 weights and stays
    close to the peft-wrapped model that was trained.
    """
    import torch

    modules = dict(encoder.named_modules())
    merged = 0
    for module, a, b in lora.adapter:
        target = modules.get(module)
        weight = getattr(target, "weight", None)
        if target is None or weight is None:
            raise HeadLoadError(f"lora target {module!r} not found on the encoder")
        delta = lora.scale * (
            torch.from_numpy(np.asarray(b, dtype=np.float32))
            @ torch.from_numpy(np.asarray(a, dtype=np.float32))
        )
        if tuple(delta.shape) != tuple(weight.shape):
            raise HeadLoadError(
                f"lora target {module!r}: delta {tuple(delta.shape)} != weight {tuple(weight.shape)}"
            )
        with torch.no_grad():
            weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))
        merged += 1
    return merged


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

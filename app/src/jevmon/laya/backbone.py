"""Shipped frozen-backbone judges: a Hugging Face model plus a packaged linear head.

Judge models selectable here (`[judge] model`): `ettin` (`jhu-clsp/ettin-encoder-1b`), `ettin-lora`
(the same encoder with a packaged LoRA adapter merged into its weights at load, see
`jevmon.laya.head.LoraHead`) and `qwen3e` (`Qwen/Qwen3-Embedding-8B`); `laya` and `laya-lora` are
served by `LayaDecider`. This module implements the Hugging Face ones. It loads the model lazily (only when selected),
computes the same frozen features as `lab/scripts/backbone_embed.py` -- a decoder is left-padded
with EOS appended after truncation, an encoder is a padded batch; variants are the mean of the last
layer and of the layer at 3/4 depth, plus `last`/`last75` for a decoder and `cls` for an encoder --
and applies the packaged head (a `TunedHead` whose `variants` say which columns the head reads).

`transformers` is an optional dependency: install it with `uv sync --extra backbone`. It is
imported lazily and a missing install makes the judge report itself unavailable; there is no
silent fallback to Laya. The first use downloads the model from Hugging Face.
"""

import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from jevmon.detect.judge import SAFE
from jevmon.laya.decider import Decision
from jevmon.sanitize import shorten

MODEL_MAX = 1_000  # characters of state the model ever sees (the judge already bounds it)
EXTRA_HINT = "backbone judges need the optional dependency: run `uv sync --extra backbone`"


@dataclass(frozen=True, slots=True)
class Backbone:
    """One selectable Hugging Face judge backbone."""

    key: str
    model_id: str  # the Hugging Face id the head was trained against
    kind: str  # "encoder" (padded batch) or "decoder" (left-pad, EOS after truncation)
    lora: str = ""  # packaged LoRA adapter (bare name) merged into the model at load; "" = none


# The Hugging Face options. Laya is not here: it is served by the `laya` package (LayaDecider).
MODELS = {
    "ettin": Backbone("ettin", "jhu-clsp/ettin-encoder-1b", "encoder"),
    "ettin-lora": Backbone("ettin-lora", "jhu-clsp/ettin-encoder-1b", "encoder", lora="ettin-lora"),
    "qwen3e": Backbone("qwen3e", "Qwen/Qwen3-Embedding-8B", "decoder"),
}


@dataclass(frozen=True, slots=True)
class BackboneConfig:
    """Everything the backbone judge needs: which model, which head, and where it runs."""

    key: str  # "ettin", "ettin-lora" or "qwen3e"
    categories: dict[str, str]
    routine: str = SAFE
    head: str = ""  # a head path or a shipped bare name; "" = the packaged head for `key`
    #   (for a LoRA key, "" = the adapter's bundled classifier)
    device: str = "auto"  # "auto" (GPU when a HIP device works) | "cpu" | "gpu"
    dtype: str = "auto"  # "auto" (bf16 on GPU, fp32 on CPU) | "fp32" | "fp16" | "bf16"
    threads: int = 8  # torch intra-op threads while on CPU (tokenizing/forward)
    threads_gpu: int = 2  # intra-op threads while on the GPU (CPU only tokenizes)
    batch_size: int = 8  # texts per forward pass


def _text_model(model):
    """The text stack of a multimodal wrapper (Qwen3.5 / openjev), else the model itself.

    Same rule as `lab/scripts/backbone_embed.py`, so serving and the cached features agree.
    """
    for attr in ("language_model",):
        inner = getattr(model, attr, None)
        if inner is not None:
            return inner
    inner = getattr(getattr(model, "model", None), "language_model", None)
    return inner if inner is not None else model


def embed_variants(
    tok,
    model,
    texts: Sequence[str],
    *,
    kind: str,
    variants: Sequence[str],
    max_length: int = 512,
    device: str = "cpu",
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """{variant: (n, dim) float32} for `texts`, exactly as `backbone_embed.embed` computes them.

    A decoder is left-padded and gets the EOS token appended after truncation; an encoder is a
    padded batch. Only the requested `variants` are kept.
    """
    import torch

    stack = _text_model(model)
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else (eos or 0)
    collected: list[dict[str, np.ndarray] | None] = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for s in range(0, len(order), batch_size):
        index = order[s : s + batch_size]
        chunk = [texts[i] for i in index]
        if kind == "decoder":
            ids = [
                tok(text, add_special_tokens=False)["input_ids"][: max_length - 1] + [eos]
                for text in chunk
            ]
            width = max(len(i) for i in ids)
            inp = torch.tensor([[pad] * (width - len(i)) + i for i in ids])
            att = torch.tensor([[0] * (width - len(i)) + [1] * len(i) for i in ids])
        else:
            enc = tok(
                chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            inp, att = enc["input_ids"], enc["attention_mask"]
        with torch.inference_mode():
            result = stack(
                input_ids=inp.to(device),
                attention_mask=att.to(device),
                output_hidden_states=True,
            )
            hidden = result.hidden_states
            mask = att.to(device).unsqueeze(-1).float()
            n_layers = len(hidden) - 1
            mid = hidden[(3 * n_layers) // 4].float()
            last = result.last_hidden_state.float()

            def mean(h, mask=mask):
                return ((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().float().numpy()

            feats = {"mean": mean(last), "mean75": mean(mid)}
            if kind == "decoder":
                feats["last"] = last[:, -1].cpu().float().numpy()
                feats["last75"] = mid[:, -1].cpu().float().numpy()
            else:
                feats["cls"] = last[:, 0].cpu().float().numpy()
        for j, i in enumerate(index):
            collected[i] = {v: feats[v][j] for v in variants}
    return {
        v: np.stack([c[v] for c in collected if c is not None]).astype(np.float32) for v in variants
    }


class BackboneDecider:
    """The `Decider` protocol over a frozen HF backbone and its packaged head."""

    def __init__(self, config: BackboneConfig) -> None:
        if config.key not in MODELS:
            raise ValueError(f"unknown backbone {config.key!r}: expected {sorted(MODELS)}")
        self.config = config
        self.spec = MODELS[config.key]
        self._tok = None
        self._model = None
        self._head = None
        self._device = "cpu"
        self._active_device = ""  # "gpu"/"cpu": resolved when the model loads
        self._active_dtype = ""  # "bf16"/"fp32"/"fp16"
        self._device_note: str | None = None
        self._error: str | None = None
        self._ready = threading.Event()
        self._threads_applied = threading.local()

    def reconfigure(self, config: BackboneConfig) -> None:
        """Adopt a new category/question config; the resident model and head stay until restart."""
        self.config = config

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def active_device(self) -> str:
        return self._active_device

    @property
    def active_dtype(self) -> str:
        return self._active_dtype

    @property
    def status(self) -> str:
        name = self.spec.key
        if not self.ready:
            return f"{name} loading"
        if self._model is None or self._head is None:
            return f"{name} unavailable ({self._error or 'not loaded'})"
        out = f"{name} ready ({self._active_device}"
        if self._active_dtype:
            out += f", {self._active_dtype}"
        out += ")"
        if self._device_note:
            out += f"; {self._device_note}"
        return out

    def _resolve_device(self, torch) -> tuple[str, str | None]:
        want = self.config.device if self.config.device in ("auto", "cpu", "gpu") else "auto"
        if want == "cpu":
            return "cpu", None
        try:
            available = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - a broken HIP stack reads as "no GPU"
            available = False
        if not available:
            note = "gpu requested but no HIP device; using cpu" if want == "gpu" else None
            return "cpu", note
        return "cuda", None

    def _torch_dtype(self, torch, device: str):
        want = (
            self.config.dtype if self.config.dtype in ("auto", "fp32", "fp16", "bf16") else "auto"
        )
        if want == "auto":
            want = "bf16" if device == "cuda" else "fp32"
        if want == "bf16":
            return torch.bfloat16, "bf16"
        if want == "fp16":
            return torch.float16, "fp16"
        return torch.float32, "fp32"

    def _load_on(self, torch, device_arg: str, note: str | None) -> None:
        """Load tokenizer + model + head on `device_arg`; raises on any failure."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(EXTRA_HINT) from exc

        from jevmon.laya.head import HeadLoadError, LoraHead, TunedHead, merge_lora_

        lora = LoraHead.load(self.spec.lora) if self.spec.lora else None
        head = TunedHead.load(self.config.head) if self.config.head else None
        if head is None:
            head = lora.head if lora else TunedHead.load(self.spec.key)
        if head.backbone and head.backbone != self.spec.model_id:
            raise HeadLoadError(
                f"head backbone {head.backbone!r} != selected model {self.spec.model_id!r}"
            )
        if set(head.labels) != set(self.config.categories):
            raise HeadLoadError(
                f"head categories {sorted(head.labels)} != configured "
                f"{sorted(self.config.categories)}"
            )
        dtype, dtype_name = self._torch_dtype(torch, device_arg)
        device = torch.device(device_arg)
        tok = AutoTokenizer.from_pretrained(self.spec.model_id)
        model = AutoModel.from_pretrained(self.spec.model_id, dtype=dtype)
        model = model.to(device).eval()
        if lora is not None:
            if lora.head.backbone and lora.head.backbone != self.spec.model_id:
                raise HeadLoadError(
                    f"lora backbone {lora.head.backbone!r} != selected model {self.spec.model_id!r}"
                )
            merged = merge_lora_(model, lora)
            if merged != len(lora.adapter):
                raise HeadLoadError(f"merged {merged} of {len(lora.adapter)} lora targets")
        self._tok, self._model, self._head = tok, model, head
        self._device = device
        self._active_device = "gpu" if device_arg == "cuda" else "cpu"
        self._active_dtype = dtype_name
        self._device_note = note

    def _load_blocking(self) -> None:
        try:
            import torch
        except ImportError as exc:  # optional extra, report and stay unavailable
            raise RuntimeError(EXTRA_HINT) from exc

        device_arg, note = self._resolve_device(torch)
        if device_arg == "cuda":
            try:
                self._load_on(torch, "cuda", note)
                return
            except Exception as exc:  # noqa: BLE001 - GPU load failure: CPU fallback below
                note = (
                    f"gpu load failed ({type(exc).__name__}: {shorten(str(exc), 120)}); using cpu"
                )
        self._load_on(torch, "cpu", note)

    def load(self) -> None:
        """Load lazily; call from a worker thread. Never raises: `status` reports the outcome."""
        try:
            self._load_blocking()
            self._error = None
        except Exception as exc:  # noqa: BLE001 - no fallback to Laya: report unavailable
            self._tok = self._model = self._head = None
            self._error = f"{type(exc).__name__}: {shorten(str(exc), 200)}"
        finally:
            self._ready.set()

    def _limit_threads(self) -> None:
        torch = sys.modules.get("torch")
        if torch is None or getattr(self._threads_applied, "done", False):
            return
        threads = self.config.threads_gpu if self._active_device == "gpu" else self.config.threads
        torch.set_num_threads(max(1, threads))
        self._threads_applied.done = True

    def _unavailable(self) -> Decision:
        return Decision(self.config.routine, 0.0, 0.0, False, "model unavailable", False)

    def _decisions(self, probabilities: np.ndarray) -> list[Decision]:
        labels = tuple(self._head.labels)
        out = []
        for row in probabilities:
            label = labels[int(row.argmax())]
            probs = {str(name): float(value) for name, value in zip(labels, row, strict=True)}
            out.append(
                Decision(
                    label,
                    probs[label],
                    0.0,
                    label != self.config.routine,
                    f"model: {label} {probs[label]:.2f}",
                    probabilities=probs,
                )
            )
        return out

    def decide(self, state: str) -> Decision:
        return self.decide_batch([state])[0]

    def decide_batch(self, states: Sequence[str]) -> list[Decision]:
        self._ready.wait()
        if not states:
            return []
        if self._model is None or self._head is None:
            return [self._unavailable() for _ in states]
        self._limit_threads()
        rows = [shorten(state, MODEL_MAX) for state in states]
        feats = embed_variants(
            self._tok,
            self._model,
            rows,
            kind=self.spec.kind,
            variants=self._head.feature_variants,
            max_length=self._head.max_length,
            device=self._device,
            batch_size=self.config.batch_size,
        )
        vectors = np.concatenate([feats[v] for v in self._head.feature_variants], axis=1).astype(
            np.float32
        )
        if vectors.shape[1] != self._head.weight.shape[1]:
            raise RuntimeError(
                f"backbone features dim {vectors.shape[1]} != head dim {self._head.weight.shape[1]}"
            )
        return self._decisions(self._head.probabilities(vectors))


__all__ = ["MODELS", "Backbone", "BackboneConfig", "BackboneDecider", "embed_variants"]

"""Laya decision model wrapper: load once (bounded), answer typed questions about a text state.

The verdict is advisory (the judge is zero-shot until fine-tuned on labels from this box), so
callers treat it as advisory. The wrapper is domain-neutral: the questions and categories come
from `LayaConfig`, and the state is any short text the caller built (already sanitized).

The model runs on the GPU when `device` says so (or "auto" and a working HIP device is present)
with a PyTorch ROCm wheel; any GPU failure at load or inference time demotes the model to CPU and
says so in `status` -- a monitoring verdict must keep flowing. Active device and dtype are on
`active_device`/`active_dtype`.
"""

import contextlib
import os
import re
import sys
import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

from jevmon.sanitize import shorten

MODEL_MAX = 1_000  # characters of state the model ever sees
# States per forward pass inside one `predict_batch` call. A pass pads every state to the longest
# one in it and, measured on this CPU, costs exactly that padded length: one state per pass is the
# floor (795 ms/ctx) and every extra state per pass adds its padding -- pairs 851, fours 883, and
# one padded slab for 16 states 1513 (bench_judge.py --cap-sweep). Pairs are
# the best multi-state setting measured, so a drained batch is evaluated as length-sorted pairs.
PASS_SIZE = 2
# On the iGPU a forward pass is mostly fixed cost, so unlike on CPU a wider pass wins, but only
# up to a point: a drained batch of 32 measured 39 ms/ctx at pass 4 and 43/43/46/51 at 2/8/16/32
# (pass 1, one forward per state, was 82 -- bench_judge.py --pass-sweep).
GPU_PASS_SIZE = 4

_DEVICES = ("auto", "cpu", "gpu")
_DTYPES = ("auto", "fp32", "fp16", "bf16")
# Errors that mean the GPU itself failed (HIP/kernel/OOM), as opposed to a bad request: only these
# demote the model to CPU; anything else propagates to the judge, which degrades the batch.
_GPU_ERROR = re.compile(
    r"hip|cuda|rocm|gfx\d|out of memory|memory exceeded|allocation failed|device-side|out of resource",
    re.IGNORECASE,
)


def _default_categories() -> dict[str, str]:
    return {
        "routine": "normal operation, informational",
        "suspicious": "possible attack, intrusion or abuse",
        "fault": "error, crash, failure or misconfiguration",
    }


@dataclass(frozen=True, slots=True)
class LayaConfig:
    categories: dict[str, str] = field(default_factory=_default_categories)
    routine: str = "routine"  # the category that means "nothing to see"
    attention_question: str = "Does this need a human's attention?"
    attack_question: str = "Is this evidence of an attack or unauthorized access?"
    attention_min: float = 0.95  # strict by default: zero-shot confidence is badly calibrated
    attack_min: float = 1.0  # 1.0 disables the attack trigger
    checkpoint: str = "english"
    # Local directory holding a fine-tuned checkpoint (a `laya-train` output), registered on the
    # router as `checkpoint`. Empty = the built-in source for that name. Load-time.
    checkpoint_dir: str = ""
    offline: bool = False  # never touch the network; cache only
    load_timeout: float = 60.0
    threads: int = 4  # torch intra-op threads; see JudgeSettings.threads
    threads_gpu: int = 2  # intra-op threads while the model is on the GPU (CPU does tokenizing)
    device: str = "auto"  # "auto" (GPU when a HIP device works) | "cpu" | "gpu"
    dtype: str = "auto"  # "auto" (laya's per-device default) | "fp32" | "fp16" | "bf16"
    pass_size: int = 0  # states per forward pass; 0 = auto: PASS_SIZE, GPU_PASS_SIZE on the GPU
    # Path to a fine-tuned head (`jevmon.laya.head.TunedHead`, trained by lab/scripts/finetune_head).
    # Empty = the zero-shot model decides. Load-time, like device/dtype.
    tuned: str = ""


@dataclass(frozen=True, slots=True)
class Decision:
    label: str
    confidence: float  # probability of `label`
    attack: float  # probability of the attack question
    flagged: bool  # crosses the configured thresholds and is not routine
    summary: str
    available: bool = True
    probabilities: dict[str, float] = field(default_factory=dict)  # per-category probabilities


def questions(config: LayaConfig) -> dict:
    return {
        "attention": {
            "type": "choice",
            "instructions": config.attention_question,
            "criteria": dict(config.categories),
        },
        "attack": {"type": "noul", "instructions": config.attack_question},
    }


class LayaDecider:
    def __init__(self, config: LayaConfig | None = None) -> None:
        self.config = config or LayaConfig()
        self._questions = questions(self.config)
        self._router = None
        self._ready = threading.Event()
        self._error: str | None = None
        self._threads_applied = threading.local()
        self._active_device = ""  # "", "cpu" or "gpu": resolved when the router is built
        self._active_dtype = ""  # "", "fp32", "fp16" or "bf16": read back from the agent
        self._device_note: str | None = None  # set when the GPU was asked for but is not in use
        self._head = None  # TunedHead when [judge] tuned loaded one; None = zero-shot
        self._embed = None  # the laya embedder feeding the head
        self._tuned_note: str | None = None  # why the configured head is not in use
        self._checkpoint = self.config.checkpoint  # effective checkpoint name (see _register)
        self._checkpoint_note: str | None = None  # why checkpoint_dir is not in use

    def reconfigure(self, config: LayaConfig) -> None:
        """Adopt a new question/category config without reloading the checkpoint.

        `device`, `dtype` and `tuned` are load-time: a new value is stored and applied to the
        status strings that read `config`, but the resident model and head keep what they were
        loaded with until restart.
        """
        self.config = config
        self._questions = questions(config)

    @property
    def active_device(self) -> str:
        """The device the model actually runs on: "gpu", "cpu", or "" before the load."""
        return self._active_device

    @property
    def active_dtype(self) -> str:
        """The precision the model actually runs in ("" before the load)."""
        return self._active_dtype

    def _agent(self, router: object) -> object | None:
        """The loaded `laya.Agent` for the effective checkpoint (None for a fake router)."""
        agents = getattr(router, "_agents", None)
        if not isinstance(agents, dict):
            return None
        return agents.get(self._checkpoint)

    def _register_checkpoint(self, router: object) -> None:
        """Point `checkpoint` at `checkpoint_dir` on the router before anything loads.

        A directory that will not register (missing, unreadable, a router without `register`)
        falls back to the shipped built-in checkpoint and records why, so a bad path degrades
        to the zero-shot judge instead of making every call fail.
        """
        self._checkpoint = self.config.checkpoint
        self._checkpoint_note = None
        if not self.config.checkpoint_dir:
            return
        try:
            register = getattr(router, "register", None)
            if not callable(register):
                raise TypeError("router cannot register checkpoints")
            register(
                self.config.checkpoint,
                self.config.checkpoint_dir,
                description="jevmon fine-tune",
            )
        except Exception as exc:  # noqa: BLE001 - a bad path must not take the judge down
            self._checkpoint = LayaConfig().checkpoint
            self._checkpoint_note = (
                f"checkpoint_dir failed ({type(exc).__name__}: {shorten(str(exc), 120)}); "
                f"using {self._checkpoint}"
            )

    def _resolve_device(self, torch: object) -> tuple[str, str | None]:
        """Map `config.device` to the Router device argument; never raises.

        No HIP device (CPU-only torch wheel, no `/dev/kfd` access) is the normal case for
        "auto" and just means CPU; only an explicit "gpu" that cannot be honoured earns a note.
        """
        want = self.config.device if self.config.device in _DEVICES else "auto"
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

    def _read_agent_state(self, router: object, torch: object) -> tuple[str, str]:
        """(device, dtype) the agent is actually on after the load, from laya itself."""
        agent = self._agent(router)
        if agent is None:
            return "", ""
        device = getattr(agent.device, "type", "cpu")
        if not getattr(agent, "amp_enabled", False):
            dtype = "fp32"
        elif getattr(agent, "dtype", None) == torch.float16:
            dtype = "fp16"
        elif getattr(agent, "dtype", None) == torch.bfloat16:
            dtype = "bf16"
        else:
            dtype = "fp32"
        return ("gpu" if device == "cuda" else "cpu"), dtype

    def _apply_dtype(self, router: object, torch: object) -> None:
        """Apply `config.dtype` to the loaded agent; "auto" leaves laya's per-device default."""
        want = self.config.dtype if self.config.dtype in _DTYPES else "auto"
        if want == "auto":
            return
        agent = self._agent(router)
        if agent is None:  # a fake router or a layout change: nothing to retune
            return
        if want == "fp32":
            agent.amp_enabled = False
        elif want == "fp16":
            agent.dtype, agent.amp_enabled = torch.float16, True
        else:
            agent.dtype, agent.amp_enabled = torch.bfloat16, True

    def _construct(self, router_cls, device_arg: str, torch: object) -> object:
        """Router(device=...) plus one warm-up predict (which loads the checkpoint)."""
        router = router_cls(device=device_arg)
        self._register_checkpoint(router)
        if device_arg == "cuda" or self.config.dtype != "auto":
            # Load the agent first so the dtype is applied before the warm-up forward runs.
            preload = getattr(router, "preload", None)
            if callable(preload):
                # No preload (old laya): the warm-up predict loads the agent a moment later.
                with contextlib.suppress(Exception):
                    preload([self.config.checkpoint])
        self._apply_dtype(router, torch)
        router.predict("startup", self._questions, model=self._checkpoint)
        return router

    def _build(self):
        """Construct and warm up the router. Slow; may download a checkpoint.

        With a GPU device this is also the load-time fallback: any failure while building on the
        GPU rebuilds on CPU and records why, so a bad HIP stack degrades instead of dying.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import torch
            from laya import Router

            device_arg, note = self._resolve_device(torch)
            router = None
            if device_arg == "cuda":
                try:
                    router = self._construct(Router, "cuda", torch)
                except Exception as exc:  # noqa: BLE001 - GPU load failure: CPU fallback below
                    note = f"gpu load failed ({type(exc).__name__}: {shorten(str(exc), 120)}); using cpu"
                    router = None
            if router is None:
                if device_arg == "cuda":
                    device_arg = "cpu"
                router = self._construct(Router, "cpu", torch)
            active, dtype = self._read_agent_state(router, torch)
            if note is None and device_arg == "cuda" and active == "cpu":
                # laya itself fell back while placing the model; mirror its warning as our note.
                why = next(
                    (
                        str(w.message)
                        for w in caught
                        if "cpu" in str(w.message).lower() and "fall" in str(w.message).lower()
                    ),
                    "gpu placement failed",
                )
                note = f"gpu fell back to cpu ({shorten(why, 120)})"
            if not active:  # no agent to introspect (fake/foreign router): trust the argument
                active = "gpu" if device_arg == "cuda" else "cpu"
            self._active_device, self._active_dtype, self._device_note = active, dtype, note
            return router

    def _load_tuned(self) -> None:
        """Load the configured fine-tuned head next to the resident checkpoint.

        A missing, malformed or category-mismatched head degrades to the zero-shot path and
        records why in `_tuned_note` (shown by `status`): the judge keeps answering instead of
        going unavailable over a bad path.
        """
        if not self.config.tuned:
            return
        try:
            from jevmon.laya.head import HeadLoadError, TunedHead, make_embedder

            head = TunedHead.load(self.config.tuned)
            if set(head.labels) != set(self.config.categories):
                raise HeadLoadError(
                    f"head categories {sorted(head.labels)} != "
                    f"configured {sorted(self.config.categories)}"
                )
            agent = self._agent(self._router)
            if agent is None:
                raise HeadLoadError("no loaded agent to embed states with")
            self._embed = make_embedder(agent, pooling=head.pooling, max_length=head.max_length)
            self._head, self._tuned_note = head, None
        except Exception as exc:  # noqa: BLE001 - a bad head must not take the judge down
            self._head, self._embed = None, None
            self._tuned_note = (
                f"tuned head failed ({type(exc).__name__}: {shorten(str(exc), 120)}); "
                "zero-shot model"
            )

    def _load_blocking(self) -> None:
        try:
            self._router = self._build()
            self._error = None
        except Exception as exc:  # noqa: BLE001 - the model is optional; degrade, do not crash
            self._error = f"{type(exc).__name__}: {exc}"
            return
        self._load_tuned()

    def load(self) -> None:
        """Load the checkpoint; call from a worker thread. Never blocks past `load_timeout`.

        A missing checkpoint with no network makes the hub client retry or hang, so the load runs
        in a helper thread and we give up after the timeout. If it finishes later the model is
        picked up then.
        """
        cfg = self.config
        if cfg.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
        worker = threading.Thread(target=self._load_blocking, daemon=True, name="laya-load")
        worker.start()
        try:
            worker.join(cfg.load_timeout)
            if worker.is_alive():
                self._error = f"load timed out after {cfg.load_timeout:g}s (offline?)"
        finally:
            self._ready.set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def status(self) -> str:
        if not self.ready:
            return "loading"
        if self._router is None:
            return f"unavailable ({self._error or 'not loaded'})"
        if not self._active_device:  # a router injected for tests, not built by us
            return "ready" + self._tuned_status
        out = f"ready ({self._active_device}"
        if self._active_dtype:
            out += f", {self._active_dtype}"
        out += ")"
        if self._device_note:
            out += f"; {self._device_note}"
        return out + self._tuned_status

    @property
    def _tuned_status(self) -> str:
        """What `[judge] tuned`/`checkpoint_dir` are doing: loaded, off, or why not."""
        if self._checkpoint_note:
            return f"; {self._checkpoint_note}"
        if self._tuned_note:
            return f"; {self._tuned_note}"
        if self._head is not None:
            name = self.config.tuned.rsplit("/", 1)[-1]
            return f"; tuned head ({name})"
        if self.config.checkpoint_dir and self._checkpoint != LayaConfig().checkpoint:
            return f"; checkpoint {self._checkpoint}"
        return ""

    def _limit_threads(self) -> None:
        """Cap torch's intra-op pool for the calling thread (once per thread, per device)."""
        torch = sys.modules.get("torch")  # present once the router is built; never import it here
        if torch is None or getattr(self._threads_applied, "done", False):
            return
        threads = self.config.threads_gpu if self._active_device == "gpu" else self.config.threads
        torch.set_num_threads(max(1, threads))
        self._threads_applied.done = True

    @staticmethod
    def _gpu_error(exc: BaseException) -> bool:
        """Is this error the GPU itself failing (rather than a bad request)?"""
        if type(exc).__name__ == "OutOfMemoryError" or isinstance(exc, MemoryError):
            return True
        return bool(_GPU_ERROR.search(f"{type(exc).__name__}: {exc}"))

    def _demote_to_cpu(self, exc: BaseException) -> bool:
        """Move a GPU resident model to CPU after a GPU failure; True when the retry may run."""
        if self._active_device != "gpu" or self._router is None:
            return False
        agent = self._agent(self._router)
        if agent is None:
            return False
        try:
            import torch

            deaccelerate = getattr(agent, "deaccelerate", None)
            if callable(deaccelerate):
                deaccelerate()  # a backend's copied device weights must go first
            agent.device = torch.device("cpu")
            agent.dtype = torch.float32
            agent.amp_enabled = False
            agent.model.to(agent.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - could not demote: the original error propagates
            return False
        self._active_device = "cpu"
        self._active_dtype = "fp32"
        self._device_note = (
            f"gpu inference failed ({type(exc).__name__}: {shorten(str(exc), 120)}); using cpu"
        )
        self._threads_applied.done = False  # re-cap the pool for the CPU thread count
        return True

    def _gpu_call(self, fn):
        """Run a router call; on a GPU error demote to CPU and retry once, else propagate."""
        try:
            return fn()
        except Exception as exc:  # only a GPU-shaped error demotes; anything else propagates
            if not self._gpu_error(exc) or not self._demote_to_cpu(exc):
                raise
            return fn()

    def _unavailable(self) -> Decision:
        return Decision(self.config.routine, 0.0, 0.0, False, "model unavailable", False)

    def _requests(self, states: Sequence[str]) -> list[dict]:
        return [
            {
                "state": shorten(state, MODEL_MAX),
                "questions": self._questions,
                "model": self._checkpoint,
            }
            for state in states
        ]

    def _decision(self, result: object) -> Decision:
        """Build the Decision from one prediction result (shared by `decide` and `decide_batch`).

        A result without usable answers (a failed or skipped batch item) degrades to unavailable.
        """
        answers = (result or {}).get("answers") if isinstance(result, dict) else None
        if not answers:
            return self._unavailable()
        attention = answers.get("attention") or {}
        label = str(attention.get("choice", self.config.routine))
        probabilities = {
            str(name): float(prob) for name, prob in (attention.get("probabilities") or {}).items()
        }
        conf = probabilities.get(label, 0.0)
        attack = float((answers.get("attack") or {}).get("noul", 0.0))
        cfg = self.config
        flagged = attack >= cfg.attack_min or (label != cfg.routine and conf >= cfg.attention_min)
        return Decision(
            label,
            conf,
            attack,
            flagged,
            f"model: {label} {conf:.2f}, attack {attack:.2f}",
            probabilities=probabilities,
        )

    def _tuned_decision(self, states: Sequence[str]) -> list[Decision]:
        """Answer for every state through the fine-tuned head instead of the typed question.

        The head only replaces where the category probabilities come from: the result is fed
        through the same `_decision` as the zero-shot path, so the confidence bands, the
        routine category and the attack threshold all apply unchanged.
        """
        import numpy as np

        rows = [shorten(state, MODEL_MAX) for state in states]
        probs = np.asarray(self._head.probabilities(self._embed(rows)), dtype=np.float32)
        if len(probs) != len(rows):
            raise RuntimeError(f"tuned head returned {len(probs)} rows for {len(rows)} states")
        labels = tuple(self._head.labels)
        results = [
            {
                "answers": {
                    "attention": {
                        "choice": labels[int(row.argmax())],
                        "probabilities": {
                            str(name): float(value) for name, value in zip(labels, row, strict=True)
                        },
                    },
                    "attack": {"noul": 0.0},
                }
            }
            for row in probs
        ]
        return [self._decision(result) for result in results]

    def decide(self, state: str) -> Decision:
        """Answer the configured questions about `state` (sanitize it first)."""
        self._ready.wait()
        if self._router is None:
            return self._unavailable()
        self._limit_threads()
        if self._head is not None:
            return self._tuned_decision([state])[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._gpu_call(
                lambda: self._router.predict(
                    shorten(state, MODEL_MAX), self._questions, model=self._checkpoint
                )
            )
        return self._decision(result)

    def _pass_size(self) -> int:
        """States per forward pass: the configured one, else per-device (auto)."""
        if self.config.pass_size > 0:
            return self.config.pass_size
        return GPU_PASS_SIZE if self._active_device == "gpu" else PASS_SIZE

    def decide_batch(self, states: Sequence[str]) -> list[Decision]:
        """Answer for every state in one `predict_batch` call; results follow the input order.

        Lengths are sorted inside the call (restoring input order) so similar states share a
        forward pass of at most `pass_size` states (auto: pairs on CPU, wider on the GPU).
        Unavailable model: unavailable decisions for all. Errors from the call propagate to the
        caller (the judge degrades them); a GPU failure demotes to CPU first (see `status`).
        """
        self._ready.wait()
        if not states:
            return []
        if self._router is None:
            return [self._unavailable() for _ in states]
        self._limit_threads()
        if self._head is not None:
            return self._tuned_decision(states)
        requests = self._requests(states)
        pass_size = self._pass_size()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self._gpu_call(
                lambda: self._router.predict_batch(
                    requests, batch_size=pass_size, sort_by_length=True
                )
            )
        return [self._decision(result) for result in results]

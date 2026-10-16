"""Judge exec contexts with the decision model. The model is the only thing that assigns
severity; this module carries the context to it, memoises decisions and stores the verdict.

Design:

- Questions and classes come from config (`[judge]`); the model answers a typed choice question.
  The answer is the severity (safe 0, warn 1, crit 2); its confidence rides along, with no
  thresholds on top.
- Volume is handled by judging each distinct situation once: contexts are memoised by signature
  with a TTL and re-judged when the signature changes. The model runs on one worker thread behind a
  bounded queue; full-queue contexts are counted and stored as `judged=false`, never filtered.
- The worker judges in small batches: it takes the first waiting context, then drains up to the
  batch limit minus one more without waiting (idle latency stays at one context, a backlog fills
  the batch) and asks the decider for all of them in one call. The limit is
  `[judge] batch_size`, or auto per model device when that is 0 (32 on the GPU, 1 on CPU). An
  exception in a batch degrades those items to `judged=false`; the worker survives.
- With no model (offline, load failure, timeout) contexts are recorded as safe events with
  `judged=false` and the status says why. There is no heuristic fallback.
"""

import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Protocol

from jevmon.detect.context import Context
from jevmon.events import Event, Scalar, Severity, severity_of
from jevmon.laya.decider import Decision

WINDOW_BATCHES = 64  # recent model calls behind avg_batch / ms_per_context
CLASSES = tuple(m.name.lower() for m in Severity)  # ("safe", "warn", "crit")
SAFE = CLASSES[0]  # the class that means "nothing to see"; also what a non-verdict reads as
GPU_BATCH_SIZE = 32  # batch_size=0 (auto) drains this many per call while the model is on the GPU


class Decider(Protocol):
    """The slice of `LayaDecider` the judge needs (a fake stands in for tests).

    `decide_batch` is optional: a decider without it is driven one context per `decide` call.
    """

    @property
    def ready(self) -> bool: ...

    @property
    def status(self) -> str: ...

    def decide(self, state: str) -> Decision: ...


@dataclass(frozen=True, slots=True)
class JudgeSettings:
    # Which judge runs: "laya" is the shipped model, "fake" a deterministic stand-in with no
    # model, no GPU and no judgement (see detect/fake.py) for demos, tests and screenshots.
    mode: str = "laya"
    question: str = "Which of these categories best describes this process launch?"
    # The three classes and what each means (prompt content, generic: no process names or paths).
    # The label the model answers IS the severity (see events.Severity): safe 0, warn 1, crit 2.
    categories: dict[str, str] = field(
        default_factory=lambda: {
            "safe": (
                "expected everyday activity here: shells, builds, package installs, scheduled "
                "jobs, ssh sessions and other development tooling; no action needed"
            ),
            "warn": (
                "out of the ordinary for this workstation but plausibly harmless: a one-off "
                "script, the first run of a newly installed tool, a debugger; worth a glance"
            ),
            "crit": (
                "looks hostile from the launch alone: downloading and running remote code, "
                "encoded or obfuscated commands, execution from a temporary, world-writable or "
                "hidden location, a shell spawned by an unexpected parent"
            ),
        }
    )
    # Path to a fine-tuned classification head over the frozen encoder (`.safetensors` written by
    # `lab/scripts/finetune_head`) or a shipped head's bare name. Empty = the zero-shot model decides.
    # Load-time: the resident head stays until restart, like `device`/`dtype`.
    tuned: str = ""
    # Only for `mode = "fake"` (demos and screenshots): a context containing one of these
    # fragments gets that category instead of the hash-derived one, so a mocked run can show
    # plausible verdicts. Ignored by the model judge.
    fake_verdicts: dict[str, str] = field(default_factory=dict)
    # Checkpoint the model answers from: a built-in name ("english", "multilingual",
    # "typed-decisions") or a local fine-tune registered through `checkpoint_dir`.
    checkpoint: str = "english"
    # Local directory of a fine-tune (a `laya-train` output), registered on the router as
    # `checkpoint`. Empty = the built-in source for that name. Load-time.
    checkpoint_dir: str = ""
    memo_ttl_seconds: float = 300.0
    queue_capacity: int = 50_000
    # Model device: "auto" uses the GPU when a working HIP device (PyTorch ROCm wheel) is
    # present, else CPU; "cpu"/"gpu" force it. A GPU failure at load or inference time falls
    # back to CPU and says so in `ctl status` -- the worker never dies over it.
    device: str = "auto"
    # Autocast precision: "auto" (laya's per-device default: bf16 on GPU, fp32 on CPU),
    # "fp32", "fp16" or "bf16".
    dtype: str = "auto"
    # Contexts the worker drains per model call: 0 = auto -- 32 on the GPU (one forward pass
    # there is mostly fixed cost: measured 65 ms/ctx single, 55 at drain 8, 44 at 32 with
    # bf16), 1 on CPU (batching only adds padding there, measured 2-22%
    # slower than 1). A positive number forces that drain size on both devices. The worker
    # never waits to fill a batch, so an idle judge still answers one at a time.
    batch_size: int = 0
    # torch intra-op threads for the model: measured 2.3x faster than 4 (720 vs 1642 ms/context)
    # and 8 <= CPUQuota/100 in the shipped unit. Keep this at or below the
    # service's CPUQuota in cores: torch's default (one per core) thrashes against a quota
    # (5.9 s/call vs 0.96 s measured).
    threads: int = 8
    # Intra-op threads while the model is on the GPU: there the CPU only tokenizes and feeds,
    # so the default is low on purpose -- a monitoring tool must not peg cores while the iGPU
    # does the work.
    threads_gpu: int = 2

    def __post_init__(self) -> None:
        if self.mode not in ("laya", "fake"):
            raise ValueError(f"mode must be 'laya' or 'fake', got {self.mode!r}")
        if self.device not in ("auto", "cpu", "gpu"):
            raise ValueError(f"device must be 'auto', 'cpu' or 'gpu', got {self.device!r}")
        if self.dtype not in ("auto", "fp32", "fp16", "bf16"):
            raise ValueError(f"dtype must be 'auto', 'fp32', 'fp16' or 'bf16', got {self.dtype!r}")
        if not isinstance(self.tuned, str):
            raise TypeError(f"tuned must be a path string, got {self.tuned!r}")
        for name in self.categories:
            if name not in CLASSES:
                raise ValueError(f"unknown class {name!r}: expected {', '.join(CLASSES)}")
        if SAFE not in self.categories:
            raise ValueError(f"categories must include {SAFE!r}")
        for fragment, category in self.fake_verdicts.items():
            if category not in self.categories:
                raise ValueError(f"fake_verdicts[{fragment!r}]: unknown category {category!r}")
        if not isinstance(self.checkpoint, str) or not self.checkpoint:
            raise ValueError(f"checkpoint must be a non-empty name, got {self.checkpoint!r}")
        if not isinstance(self.checkpoint_dir, str):
            raise TypeError(f"checkpoint_dir must be a path string, got {self.checkpoint_dir!r}")


@dataclass(frozen=True, slots=True)
class Verdict:
    label: str
    confidence: float
    severity: int
    judged: bool
    cached: bool
    available: bool
    summary: str
    probabilities: dict[str, float] = field(default_factory=dict)


class Judge:
    """Memoise and run the decider over contexts on one worker thread."""

    def __init__(self, decider: Decider, settings: JudgeSettings, emit=None) -> None:
        self.decider = decider
        self.settings = settings
        self._emit = emit if emit is not None else (lambda event: None)
        # queued as (context, monotonic enqueue time): staleness must not depend on Context.time
        self._queue: queue.Queue[tuple[Context, float]] = queue.Queue(
            maxsize=settings.queue_capacity
        )
        self._cache: dict[str, tuple[Verdict, float]] = {}
        # signature -> contexts that arrived while the same situation was already queued or being
        # judged; they share its verdict instead of queueing duplicates behind a slow model.
        self._pending: dict[str, list[Context]] = {}
        # (contexts per model call, its wall ms) over the recent window behind the status metrics.
        self._samples: deque[tuple[int, float]] = deque(maxlen=WINDOW_BATCHES)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.judged = 0
        self.cached = 0
        self.dropped = 0
        self.unavailable = 0
        self._paused = False

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="jevmon-judge", daemon=True)
            self._thread.start()

    def drain(self, timeout: float = 2.0) -> None:
        """Wait (bounded) until nothing is queued or in flight.

        Every submitted context holds an entry in `_pending` until its verdict is emitted, so an
        empty map plus an empty queue means the store has every verdict. Shutdown uses this so the
        last contexts of a run are not dropped when the source stops ahead of the worker.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                pending = bool(self._pending)
            if not pending and self._queue.qsize() == 0:
                return
            time.sleep(0.005)

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self.drain()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    @property
    def status(self) -> str:
        return (
            f"decider[{self.decider.status}] judged={self.judged} cached={self.cached} "
            f"dropped={self.dropped} unavailable={self.unavailable}"
        )

    @property
    def depth(self) -> int:
        """Contexts waiting for the judge worker."""
        return self._queue.qsize()

    @property
    def oldest_age(self) -> float:
        """Seconds since the oldest waiting context was enqueued (verdict staleness)."""
        with self._queue.mutex:
            head = self._queue.queue[0] if self._queue.queue else None
        return max(0.0, time.monotonic() - head[1]) if head is not None else 0.0

    @property
    def avg_batch(self) -> float:
        """Average contexts per model call over the recent window (1.0 with no batching)."""
        with self._lock:
            samples = list(self._samples)
        return sum(n for n, _ in samples) / len(samples) if samples else 0.0

    @property
    def ms_per_context(self) -> float:
        """Measured wall ms per judged context over the recent window (0.0 before the first call)."""
        with self._lock:
            samples = list(self._samples)
        contexts = sum(n for n, _ in samples)
        return sum(ms for _, ms in samples) / contexts if contexts else 0.0

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Stop asking the model but keep accepting contexts: they are recorded judged=false."""
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def submit(self, context: Context) -> Verdict | None:
        """Judge (or reuse a cached verdict for) a context. Returns the verdict when known now."""
        if self._paused:
            verdict = Verdict(SAFE, 0.0, Severity.SAFE, False, False, False, "paused")
            self._emit(self._event(context, verdict, reason="paused"))
            return verdict
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(context.signature)
            if entry is not None and entry[1] > now:
                self.cached += 1
                verdict = replace(entry[0], cached=True)
            elif context.signature in self._pending:
                self._pending[context.signature].append(context)
                return None
            else:
                self._pending[context.signature] = []
        if entry is not None and entry[1] > now:
            self._emit(self._event(context, verdict))
            return verdict
        try:
            self._queue.put_nowait((context, time.monotonic()))
        except queue.Full:
            with self._lock:
                self.dropped += 1
                self._pending.pop(context.signature, None)
            verdict = Verdict(SAFE, 0.0, Severity.SAFE, False, False, False, "queue full")
            self._emit(self._event(context, verdict, reason="dropped: judge queue full"))
            return verdict
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                context, _ = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # Take the first waiting context, then drain the backlog without waiting: idle latency
            # stays at one context, a queue behind a slow model fills the batch naturally.
            batch = [context]
            while len(batch) < self._batch_limit():
                try:
                    waiting, _ = self._queue.get_nowait()
                except queue.Empty:
                    break
                batch.append(waiting)
            for queued, verdict in zip(batch, self._judge_batch(batch), strict=True):
                self._emit(self._event(queued, verdict))
                with self._lock:
                    waiters = self._pending.pop(queued.signature, [])
                    if verdict.judged:
                        self.cached += len(waiters)
                shared = replace(verdict, cached=True) if verdict.judged else verdict
                for waiter in waiters:
                    self._emit(self._event(waiter, shared))
                self._queue.task_done()

    def _batch_limit(self) -> int:
        """Contexts the worker may drain per model call (never blocks to fill it).

        `batch_size = 0` follows the model's actual device: GPU_BATCH_SIZE while the decider
        runs on the GPU, 1 on CPU (no decider attribute counts as CPU).
        """
        size = self.settings.batch_size
        if size == 0:
            size = GPU_BATCH_SIZE if getattr(self.decider, "active_device", "") == "gpu" else 1
        return max(1, size)

    def _judge_batch(self, batch: list[Context]) -> list[Verdict]:
        """Judge one drained batch in a single decider call, one item per call when there is none."""
        if not self.decider.ready:
            self._count_unavailable(len(batch))
            return [self._unavailable(f"model unavailable ({self.decider.status})") for _ in batch]
        batched = getattr(self.decider, "decide_batch", None)
        started = time.perf_counter()
        if not callable(batched):
            verdicts = [self._judge(context) for context in batch]
        else:
            try:
                decisions = list(batched([context.state for context in batch]))
            except Exception as exc:  # noqa: BLE001 - a broken model must not stop the judge
                self._count_unavailable(len(batch))
                verdicts = [self._unavailable(f"model error: {type(exc).__name__}") for _ in batch]
            else:
                verdicts = [
                    self._verdict(context, decisions[i] if i < len(decisions) else None)
                    for i, context in enumerate(batch)
                ]
        self._record_batch(len(batch), (time.perf_counter() - started) * 1000)
        return verdicts

    def _judge(self, context: Context) -> Verdict:
        """Judge one context through a decider without `decide_batch`."""
        try:
            decision = self.decider.decide(context.state)
        except Exception as exc:  # noqa: BLE001 - a broken model must not stop the judge
            self._count_unavailable(1)
            return self._unavailable(f"model error: {type(exc).__name__}")
        return self._verdict(context, decision)

    def _verdict(self, context: Context, decision: Decision | None) -> Verdict:
        if decision is None or not decision.available:
            self._count_unavailable(1)
            return self._unavailable("model unavailable")
        verdict = Verdict(
            label=decision.label,
            confidence=decision.confidence,
            severity=severity_of(decision.label),
            judged=True,
            cached=False,
            available=True,
            summary=decision.summary,
            probabilities=dict(decision.probabilities),
        )
        with self._lock:
            self.judged += 1
            self._cache[context.signature] = (
                verdict,
                time.monotonic() + self.settings.memo_ttl_seconds,
            )
        return verdict

    def _unavailable(self, summary: str) -> Verdict:
        return Verdict(
            SAFE,
            0.0,
            Severity.SAFE,
            judged=False,
            cached=False,
            available=False,
            summary=summary,
        )

    def _count_unavailable(self, count: int) -> None:
        with self._lock:
            self.unavailable += count

    def _record_batch(self, contexts: int, ms: float) -> None:
        with self._lock:
            self._samples.append((contexts, ms))

    def _event(self, context: Context, verdict: Verdict, reason: str | None = None) -> Event:
        fields: dict[str, Scalar] = dict(context.fields)
        fields.update(
            {
                "context": context.state,
                "signature": context.signature,
                "label": verdict.label,
                "confidence": round(verdict.confidence, 4),
                "judged": int(verdict.judged),
                "cached": int(verdict.cached),
                "available": int(verdict.available),
                "verdict": verdict.summary,
                "model_status": self.decider.status,
            }
        )
        if reason:
            fields["reason"] = reason
        if verdict.probabilities:
            fields["probabilities"] = json.dumps(
                verdict.probabilities, ensure_ascii=True, separators=(",", ":")
            )
        message = f"judge: {verdict.label} {verdict.confidence:.2f}"
        if reason:
            message += f" ({reason})"
        return Event(
            time=context.time,
            source="exec",
            subject=context.subject,
            kind="verdict",
            severity=verdict.severity,
            message=message,
            fields=fields,
            command=context.command,
        )

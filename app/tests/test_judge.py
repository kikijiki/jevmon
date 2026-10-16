"""Judge: memoisation, bounded queue drops, no-model path, severity bands (fake decider)."""

import threading
import time

from jevmon.detect.context import Context
from jevmon.detect.judge import Judge, JudgeSettings
from jevmon.events import Severity
from jevmon.laya.decider import Decision


class FakeDecider:
    def __init__(
        self,
        label: str = "crit",
        confidence: float = 0.95,
        available: bool = True,
        ready: bool = True,
        block: threading.Event | None = None,
    ) -> None:
        self.label = label
        self.confidence = confidence
        self.available = available
        self._ready = ready
        self.block = block
        self.calls = 0
        self.states: list[str] = []

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def status(self) -> str:
        return "fake-ready" if self._ready else "fake-unavailable"

    def decide(self, state: str) -> Decision:
        self.calls += 1
        self.states.append(state)
        if self.block is not None:
            self.block.wait(3)
        return Decision(
            self.label, self.confidence, 0.0, False, "fake verdict", available=self.available
        )


class BatchDecider(FakeDecider):
    """A decider with the batched API: records each call, optionally gated, failing or slow."""

    def __init__(
        self, gate: threading.Event | None = None, fail: int = 0, delay: float = 0.0, **kw
    ) -> None:
        super().__init__(**kw)
        self.gate = gate
        self.fail = fail
        self.delay = delay
        self.batches: list[list[str]] = []

    def decide_batch(self, states) -> list[Decision]:
        self.batches.append(list(states))
        self.calls += len(states)
        self.states.extend(states)
        if self.gate is not None:
            self.gate.wait(3)
        if self.delay:
            time.sleep(self.delay)
        if self.fail > 0:
            self.fail -= 1
            raise RuntimeError("boom")
        return [
            Decision(
                self.label, self.confidence, 0.0, False, "fake verdict", available=self.available
            )
            for _ in states
        ]


def ctx(signature: str = "s1") -> Context:
    return Context(
        time=1.0,
        kind="exec",
        subject="1",
        state=f"state {signature}",
        signature=signature,
        fields={"exe": "/opt/x"},
    )


def settings(**kw) -> JudgeSettings:
    return JudgeSettings(**kw)


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_judges_and_emits_with_context_and_verdict():
    emitted = []
    decider = FakeDecider()
    judge = Judge(decider, settings(), emit=emitted.append)
    judge.start()
    judge.submit(ctx("a"))
    assert wait_for(lambda: judge.judged == 1 and emitted)
    judge.stop()
    event = emitted[0]
    assert event.severity == Severity.CRIT and event.fields["label"] == "crit"
    assert event.fields["judged"] == 1 and event.fields["context"] == "state a"
    assert decider.states == ["state a"]
    assert "judged=1" in judge.status


def test_memoisation_judges_a_repeated_situation_once():
    emitted = []
    decider = FakeDecider()
    judge = Judge(decider, settings(), emit=emitted.append)
    judge.start()
    judge.submit(ctx("same"))
    assert wait_for(lambda: judge.judged == 1)
    judge.submit(ctx("same"))
    assert wait_for(lambda: judge.cached == 1)
    judge.stop()
    assert decider.calls == 1
    assert any(e.fields["cached"] == 1 for e in emitted)


def test_bounded_queue_counts_drops():
    block = threading.Event()
    decider = FakeDecider(block=block)
    judge = Judge(decider, settings(queue_capacity=1), emit=lambda e: None)
    judge.start()
    judge.submit(ctx("a"))
    assert wait_for(lambda: decider.calls == 1)
    judge.submit(ctx("b"))  # fills the queue while the worker is blocked
    judge.submit(ctx("c"))  # dropped
    block.set()
    assert wait_for(lambda: judge.dropped >= 1)
    judge.stop()
    assert judge.dropped >= 1


def test_no_model_records_context_without_verdict():
    emitted = []
    judge = Judge(FakeDecider(ready=False), settings(), emit=emitted.append)
    judge.start()
    judge.submit(ctx("a"))
    assert wait_for(lambda: judge.unavailable == 1 and emitted)
    judge.stop()
    event = emitted[0]
    assert event.severity == Severity.SAFE
    assert event.fields["judged"] == 0 and event.fields["available"] == 0
    assert "unavailable" in judge.status


def test_the_label_is_the_severity_whatever_the_confidence():
    for label, severity in (
        ("safe", Severity.SAFE),
        ("warn", Severity.WARN),
        ("crit", Severity.CRIT),
    ):
        for confidence in (0.4, 0.99):
            emitted = []
            judge = Judge(
                FakeDecider(label=label, confidence=confidence), settings(), emit=emitted.append
            )
            judge.start()
            judge.submit(ctx(f"{label}{confidence}"))
            assert wait_for(lambda emitted=emitted: bool(emitted))
            judge.stop()
            assert emitted[0].severity == severity
            assert emitted[0].fields["confidence"] == confidence


def test_identical_situation_in_flight_is_judged_once_and_shared():
    emitted = []
    gate = threading.Event()
    decider = FakeDecider(block=gate)
    judge = Judge(decider, settings(), emit=emitted.append)
    judge.start()
    for _ in range(4):
        judge.submit(ctx("same"))
    assert wait_for(lambda: decider.calls == 1)  # one model call while the rest wait behind it
    assert judge.depth == 0
    gate.set()
    assert wait_for(lambda: len(emitted) == 4)
    judge.stop()
    assert decider.calls == 1
    assert judge.judged == 1 and judge.cached == 3
    assert sum(1 for e in emitted if e.fields.get("cached")) == 3


def test_in_flight_waiters_are_released_when_the_queue_is_full():
    gate = threading.Event()
    judge = Judge(FakeDecider(block=gate), settings(queue_capacity=1), emit=lambda e: None)
    judge.start()
    judge.submit(ctx("a"))
    assert wait_for(lambda: judge.depth == 0)  # "a" is now with the model
    judge.submit(ctx("b"))  # queued
    judge.submit(ctx("c"))  # queue full -> dropped, must not stay registered as pending
    assert judge.dropped == 1 and "c" not in judge._pending
    gate.set()
    judge.stop()


def test_oldest_age_counts_from_enqueue_even_for_zero_time_contexts():
    gate = threading.Event()
    judge = Judge(FakeDecider(block=gate), settings(), emit=lambda e: None)
    assert judge.oldest_age == 0.0  # empty
    judge.start()
    judge.submit(ctx("a"))
    assert wait_for(lambda: judge.depth == 0)  # "a" is with the model
    # baseline census contexts carry Context.time == 0.0: age must not come from it
    zero = Context(time=0.0, kind="exec", subject="w", state="s", signature="w0", fields={})
    judge.submit(zero)
    assert judge.depth == 1
    time.sleep(0.3)
    age = judge.oldest_age
    assert 0.25 <= age < 5.0
    gate.set()
    judge.stop()


def test_worker_drains_up_to_batch_size_without_waiting():
    gate = threading.Event()
    emitted = []
    decider = BatchDecider(gate=gate, delay=0.02)
    judge = Judge(decider, settings(batch_size=4), emit=emitted.append)
    judge.start()
    judge.submit(ctx("a"))
    assert wait_for(lambda: decider.batches == [["state a"]])  # first call is in flight, blocked
    for sig in "bcde":
        judge.submit(ctx(sig))
    gate.set()  # releases every later call: the backlog must arrive as one batch
    assert wait_for(lambda: len(emitted) == 5)
    judge.stop()
    assert decider.batches[0] == ["state a"]
    assert decider.batches[1] == ["state b", "state c", "state d", "state e"]
    assert all(len(batch) <= 4 for batch in decider.batches)
    assert [event.fields["context"] for event in emitted] == [f"state {s}" for s in "abcde"]
    assert judge.judged == 5
    assert judge.avg_batch == 2.5 and judge.ms_per_context > 0.0


def test_batch_verdicts_are_emitted_in_order_and_shared_with_waiters():
    gate = threading.Event()
    emitted = []
    decider = BatchDecider(gate=gate)
    judge = Judge(decider, settings(batch_size=4), emit=emitted.append)
    judge.start()
    for _ in range(4):
        judge.submit(ctx("same"))
    assert wait_for(lambda: decider.batches == [["state same"]])
    assert judge.depth == 0  # the duplicates wait as pending, they are not queued
    gate.set()
    assert wait_for(lambda: len(emitted) == 4)
    judge.stop()
    assert decider.batches == [["state same"]]
    assert judge.judged == 1 and judge.cached == 3
    assert [event.fields["cached"] for event in emitted] == [0, 1, 1, 1]
    assert all(event.fields["judged"] == 1 for event in emitted)


def test_batch_exception_degrades_to_unjudged_and_the_worker_survives():
    emitted = []
    decider = BatchDecider(fail=1)
    judge = Judge(decider, settings(batch_size=4), emit=emitted.append)
    judge.start()
    judge.submit(ctx("a"))
    assert wait_for(lambda: len(emitted) == 1)
    assert emitted[0].fields["judged"] == 0 and emitted[0].fields["available"] == 0
    assert "model error: RuntimeError" in emitted[0].fields["verdict"]
    assert judge.unavailable == 1
    judge.submit(ctx("b"))  # a broken batch must not kill the worker
    assert wait_for(lambda: len(emitted) == 2)
    assert emitted[1].fields["judged"] == 1
    assert judge.unavailable == 1 and judge.judged == 1
    judge.stop()
    assert judge._thread is None


def test_falls_back_to_single_decide_when_the_decider_has_no_decide_batch():
    emitted = []
    decider = FakeDecider()
    assert not hasattr(decider, "decide_batch")
    judge = Judge(decider, settings(batch_size=8), emit=emitted.append)
    judge.start()
    for sig in "abc":
        judge.submit(ctx(sig))
    assert wait_for(lambda: len(emitted) == 3)
    judge.stop()
    assert decider.states == ["state a", "state b", "state c"]
    assert all(event.fields["judged"] == 1 for event in emitted)
    assert judge.avg_batch >= 1.0 and judge.ms_per_context > 0.0


def test_auto_batch_limit_follows_the_model_device():
    from jevmon.detect.judge import GPU_BATCH_SIZE

    cpu, gpu = BatchDecider(), BatchDecider()
    gpu.active_device = "gpu"
    assert Judge(cpu, JudgeSettings(batch_size=0))._batch_limit() == 1
    assert Judge(gpu, JudgeSettings(batch_size=0))._batch_limit() == GPU_BATCH_SIZE == 32
    assert Judge(gpu, JudgeSettings(batch_size=4))._batch_limit() == 4  # explicit wins


def test_verdict_event_carries_command_and_class_severity():
    emitted = []
    judge = Judge(FakeDecider(label="crit", confidence=0.3), settings(), emit=emitted.append)
    judge.start()
    context = Context(
        time=1.0,
        kind="exec",
        subject="1",
        state="state c",
        signature="c",
        fields={"exe": "/opt/x"},
        command="x --flag",
    )
    judge.submit(context)
    assert wait_for(lambda: bool(emitted))
    judge.stop()
    event = emitted[0]
    assert event.command == "x --flag"
    assert event.severity == Severity.CRIT  # low confidence does not demote

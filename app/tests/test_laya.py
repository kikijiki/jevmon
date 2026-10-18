"""Laya wrapper tests with a fake router (no model download), plus the device selection and
GPU -> CPU fallback paths against fake `torch`/`laya` modules (no HIP device needed)."""

import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from jevmon.laya.decider import (
    GPU_PASS_SIZE,
    PASS_SIZE,
    LayaConfig,
    LayaDecider,
    questions,
)


class FakeRouter:
    def __init__(self, label: str, conf: float, attack: float) -> None:
        self.label, self.conf, self.attack = label, conf, attack
        self.seen: list[str] = []

    def predict(self, text, qs, model="english"):
        self.seen.append(text)
        return {
            "answers": {
                "attention": {"choice": self.label, "probabilities": {self.label: self.conf}},
                "attack": {"noul": self.attack},
            }
        }

    def predict_batch(self, requests, batch_size=None, sort_by_length=False):
        return [
            self.predict(request["state"], request["questions"], request.get("model", "english"))
            for request in requests
        ]


def loaded(router, **cfg) -> LayaDecider:
    d = LayaDecider(LayaConfig(**cfg))
    d._router = router
    d._ready.set()
    return d


def test_questions_use_configured_text_and_categories():
    cfg = LayaConfig(
        attention_question="Is this process odd?", categories={"routine": "x", "odd": "y"}
    )
    q = questions(cfg)
    assert q["attention"]["instructions"] == "Is this process odd?"
    assert q["attention"]["criteria"] == {"routine": "x", "odd": "y"}


def test_flagged_needs_non_routine_and_threshold():
    strict = loaded(FakeRouter("fault", 0.7, 0.1))  # default attention_min 0.95
    assert not strict.decide("x").flagged
    loose = loaded(FakeRouter("fault", 0.7, 0.1), attention_min=0.6)
    assert loose.decide("x").flagged
    routine = loaded(FakeRouter("routine", 0.99, 0.0), attention_min=0.5)
    assert not routine.decide("x").flagged
    attack = loaded(FakeRouter("routine", 0.9, 0.8), attack_min=0.5)
    assert attack.decide("x").flagged


def test_state_is_bounded_before_the_model_sees_it():
    router = FakeRouter("routine", 0.9, 0.0)
    loaded(router).decide("a" * 100_000)
    assert len(router.seen[0]) <= 1_100


def test_unavailable_model_never_flags():
    d = LayaDecider()
    d._ready.set()
    out = d.decide("x")
    assert not out.available and not out.flagged and "unavailable" in d.status


def test_load_failure_and_timeout_degrade():
    class Broken(LayaDecider):
        def _build(self):
            raise OSError("no checkpoint and no network")

    b = Broken(LayaConfig(load_timeout=5))
    b.load()
    assert b.ready and "no network" in (b.error or "")

    class Hangs(LayaDecider):
        def _build(self):
            time.sleep(30)

    h = Hangs(LayaConfig(load_timeout=0.2))
    t0 = time.monotonic()
    h.load()
    assert h.ready and time.monotonic() - t0 < 5 and "timed out" in (h.error or "")


def test_decide_batch_bounds_states_and_keeps_input_order():
    router = FakeRouter("fault", 0.8, 0.1)
    out = loaded(router).decide_batch(["a" * 100_000, "short"])
    assert [d.label for d in out] == ["fault", "fault"]
    assert len(router.seen[0]) <= 1_100 and router.seen[1] == "short"


def test_decide_batch_is_unavailable_for_every_state_without_a_model():
    decider = LayaDecider()
    decider._ready.set()
    out = decider.decide_batch(["x", "y"])
    assert len(out) == 2 and all(not d.available and not d.flagged for d in out)


def test_decide_batch_degrades_a_result_without_answers():
    router = FakeRouter("routine", 0.99, 0.0)
    original = router.predict_batch

    def one_broken(requests, **kw):
        results = original(requests, **kw)
        results[1] = None
        return results

    router.predict_batch = one_broken
    out = loaded(router).decide_batch(["a", "b"])
    assert out[0].available and not out[1].available


# -- device selection and the GPU -> CPU fallbacks (fake torch and fake laya modules) ----------


class FakeTorch:
    """The slice of torch the decider touches: a HIP device on/off plus dtype sentinels."""

    def __init__(self, hip: bool) -> None:
        self.float32, self.float16, self.bfloat16 = "fp32", "fp16", "bf16"
        self.cuda = SimpleNamespace(is_available=lambda: hip, empty_cache=lambda: None)
        self.threads: list[int] = []

    def device(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(type=name)

    def set_num_threads(self, n: int) -> None:
        self.threads.append(n)


class FakeAgent:
    """The slice of laya.Agent the decider reads after the load and moves on a demotion."""

    def __init__(self, device_type: str, torch: FakeTorch) -> None:
        self.device = SimpleNamespace(type=device_type)
        self.dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
        self.amp_enabled = device_type == "cuda"
        self.moved_to: str | None = None
        self.model = SimpleNamespace(to=self._to)

    def _to(self, device) -> None:
        self.moved_to = device.type

    def deaccelerate(self) -> None:
        pass


def install_fakes(monkeypatch, torch: FakeTorch, *, gpu_build_fails: bool = False) -> ModuleType:
    """Put fake `torch` and `laya` modules where the decider imports them (no real HIP)."""

    class Router:
        def __init__(self, device: str | None = None, **_) -> None:
            if device == "cuda" and gpu_build_fails:
                raise RuntimeError("HIP error: invalid device function")
            self.device_arg = device
            kind = "cuda" if device == "cuda" else "cpu"
            self._agents = {"english": FakeAgent(kind, torch)}

        def preload(self, models) -> None:
            pass

        def predict(self, state, questions, model=None, **_) -> dict:
            return {
                "answers": {
                    "attention": {"choice": "routine", "probabilities": {"routine": 0.9}},
                    "attack": {"noul": 0.0},
                }
            }

        def predict_batch(self, requests, **_):
            return [self.predict(r["state"], r["questions"]) for r in requests]

    laya = ModuleType("laya")
    laya.Router = Router
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "laya", laya)
    return laya


def build(device: str = "auto", dtype: str = "auto") -> LayaDecider:
    """Run the real load path against the installed fakes."""
    decider = LayaDecider(LayaConfig(offline=True, load_timeout=5.0, device=device, dtype=dtype))
    router = decider._build()
    decider._router = router
    decider._ready.set()
    return decider


def test_auto_without_a_hip_device_is_cpu_without_a_note(monkeypatch):
    torch = FakeTorch(hip=False)
    install_fakes(monkeypatch, torch)
    assert LayaDecider(LayaConfig())._resolve_device(torch) == ("cpu", None)
    decider = build("auto")
    assert decider.active_device == "cpu" and decider.active_dtype == "fp32"
    assert decider.status == "ready (cpu, fp32)"


def test_gpu_requested_without_a_hip_device_falls_back_with_a_note(monkeypatch):
    torch = FakeTorch(hip=False)
    install_fakes(monkeypatch, torch)
    device, note = LayaDecider(LayaConfig(device="gpu"))._resolve_device(torch)
    assert device == "cpu" and note is not None and "no HIP device" in note
    decider = build("gpu")
    assert decider.active_device == "cpu" and "no HIP device" in decider.status


def test_auto_with_a_hip_device_builds_on_the_gpu_with_laya_defaults(monkeypatch):
    torch = FakeTorch(hip=True)
    install_fakes(monkeypatch, torch)
    decider = build("auto")
    assert decider.active_device == "gpu" and decider.active_dtype == "bf16"
    assert decider.status == "ready (gpu, bf16)"


def test_gpu_build_failure_rebuilds_on_cpu_and_says_so(monkeypatch):
    torch = FakeTorch(hip=True)
    install_fakes(monkeypatch, torch, gpu_build_fails=True)
    decider = build("auto")
    assert decider.active_device == "cpu"
    assert decider._device_note and "gpu load failed" in decider._device_note
    assert decider.status.startswith("ready (cpu") and "gpu load failed" in decider.status


def test_dtype_overrides_are_applied_to_the_agent(monkeypatch):
    torch = FakeTorch(hip=True)
    install_fakes(monkeypatch, torch)
    fp32 = build("gpu", "fp32")
    assert fp32.active_dtype == "fp32" and "gpu, fp32" in fp32.status
    agent = fp32._agent(fp32._router)
    assert agent.amp_enabled is False
    fp16 = build("gpu", "fp16")
    assert fp16.active_dtype == "fp16"
    agent = fp16._agent(fp16._router)
    assert agent.amp_enabled is True and agent.dtype == "fp16"


def hip_router(torch: FakeTorch, fail: str | None = None):
    """A loaded fake router on the GPU whose predict fails once with `fail`, then answers."""

    class Router:
        def __init__(self) -> None:
            self.calls = 0
            self._agents = {"english": FakeAgent("cuda", torch)}

        def predict(self, state, questions, model=None, **_) -> dict:
            self.calls += 1
            if fail and self.calls == 1:
                raise RuntimeError(fail)
            return {
                "answers": {
                    "attention": {"choice": "routine", "probabilities": {"routine": 0.9}},
                    "attack": {"noul": 0.0},
                }
            }

    return Router()


def gpu_decider(router, torch: FakeTorch) -> LayaDecider:
    decider = LayaDecider(LayaConfig(offline=True))
    decider._router = router
    decider._active_device, decider._active_dtype = "gpu", "bf16"
    decider._ready.set()
    return decider


def test_hip_error_at_inference_demotes_to_cpu_and_retries(monkeypatch):
    torch = FakeTorch(hip=True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    router = hip_router(torch, fail="HIP error: gfx1151 dispatch failed")
    decider = gpu_decider(router, torch)
    out = decider.decide("state")
    assert out.available and router.calls == 2  # failed once, demoted, retried on CPU
    agent = router._agents["english"]
    assert agent.device.type == "cpu" and agent.amp_enabled is False
    assert decider.active_device == "cpu" and decider.active_dtype == "fp32"
    assert "gpu inference failed" in decider.status
    assert getattr(decider._threads_applied, "done", False) is False  # CPU thread cap re-applies


def test_oom_at_inference_also_demotes(monkeypatch):
    torch = FakeTorch(hip=True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    router = hip_router(torch, fail="CUDA out of memory when allocating 1.70 GiB")
    decider = gpu_decider(router, torch)
    assert decider.decide("state").available
    assert decider.active_device == "cpu" and "gpu inference failed" in decider.status


def test_non_gpu_error_propagates_without_demoting(monkeypatch):
    torch = FakeTorch(hip=True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    router = hip_router(torch, fail=None)

    def broken(state, questions, model=None, **_):
        raise ValueError("question malformed")

    router.predict = broken
    decider = gpu_decider(router, torch)
    with pytest.raises(ValueError, match="question malformed"):
        decider.decide("state")
    assert decider.active_device == "gpu" and decider._device_note is None


def test_batch_pass_size_follows_the_device_unless_configured(monkeypatch):
    torch = FakeTorch(hip=True)
    install_fakes(monkeypatch, torch)
    assert build("auto")._pass_size() == GPU_PASS_SIZE
    assert build("cpu")._pass_size() == PASS_SIZE
    forced = LayaDecider(LayaConfig(pass_size=7))
    forced._active_device = "gpu"
    assert forced._pass_size() == 7


# -- fine-tuned head ------------------------------------------------------------------------


def tuned(head_labels=("routine", "unusual", "suspicious", "runaway"), bias=9.0, dim=4):
    """A decider whose zero-shot router would say `routine`, wired to a head favouring `suspicious`."""
    import numpy as np

    from jevmon.laya.head import TunedHead

    d = loaded(FakeRouter("routine", 0.9, 0.0))
    d._head = TunedHead(
        labels=tuple(head_labels),
        weight=np.zeros((len(head_labels), dim), dtype=np.float32),
        bias=np.asarray([bias if lab == "suspicious" else 0.0 for lab in head_labels]),
    )
    d._embed = lambda states: np.ones((len(states), dim), dtype=np.float32)
    return d


def test_tuned_head_answers_instead_of_the_typed_question():

    d = tuned()
    decision = d.decide("state")
    assert decision.label == "suspicious" and decision.confidence > 0.99
    assert decision.flagged  # non-routine and above attention_min, same band as zero-shot
    assert set(decision.probabilities) == {
        "routine",
        "unusual",
        "suspicious",
        "runaway",
    }
    assert decision.available and sum(decision.probabilities.values()) == pytest.approx(1.0)
    assert not d._router.seen  # the typed question never ran


def test_tuned_batch_keeps_order_and_bounds_states():
    d = tuned()
    decisions = d.decide_batch(["a", "b" * 10_000])
    assert [x.label for x in decisions] == ["suspicious", "suspicious"]
    assert all(x.available for x in decisions)


def test_tuned_batch_row_count_mismatch_raises():
    import numpy as np

    d = tuned()
    d._embed = lambda states: np.ones((2, 4), dtype=np.float32)  # wrong row count on purpose
    with pytest.raises(RuntimeError, match="rows"):
        d.decide_batch(["a", "b", "c"])


def test_broken_tuned_path_degrades_to_zero_shot_with_a_note(tmp_path):
    d = LayaDecider(LayaConfig(tuned=str(tmp_path / "missing.safetensors")))
    d._router = FakeRouter("routine", 0.9, 0.0)
    d._ready.set()
    d._load_tuned()
    assert d._head is None and d._tuned_note and "zero-shot" in d._tuned_note
    assert "zero-shot" in d.status
    assert d.decide("state").label == "routine"  # still judging, through the model
    assert d._router.seen  # the typed question ran after the head failed


def test_head_for_other_categories_is_refused(tmp_path):
    import numpy as np

    from jevmon.laya.head import TunedHead

    path = tmp_path / "head.safetensors"
    TunedHead(
        labels=("one", "two"),
        weight=np.zeros((2, 3), dtype=np.float32),
        bias=np.zeros(2, dtype=np.float32),
    ).save(path)
    d = LayaDecider(LayaConfig(tuned=str(path)))
    d._router = FakeRouter("routine", 0.9, 0.0)
    d._load_tuned()
    assert d._head is None and "categories" in (d._tuned_note or "")


def test_shipped_head_is_the_default_and_plumbed_from_judge_config():
    from jevmon.config import load_config
    from jevmon.monitor import build_decider

    assert LayaConfig().tuned == ""
    assert build_decider(load_config()).config.tuned == "laya"


def test_checkpoint_dir_registers_the_fine_tune_on_the_router():
    class Recording(FakeRouter):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.registered: list = []

        def register(self, name, source, description=None):
            self.registered.append((name, source))
            return name

    d = LayaDecider(LayaConfig(checkpoint="lab-ft", checkpoint_dir="models/lab-ft"))
    router = Recording("routine", 0.9, 0.0)
    d._register_checkpoint(router)
    assert router.registered == [("lab-ft", "models/lab-ft")]
    assert d._checkpoint == "lab-ft" and d._checkpoint_note is None
    d._ready.set()
    d._router = router
    assert d.decide("x").label == "routine" and router.seen
    assert "checkpoint lab-ft" in d.status


def test_checkpoint_dir_failure_falls_back_to_the_shipped_checkpoint():
    d = LayaDecider(LayaConfig(checkpoint="lab-ft", checkpoint_dir="/nope"))
    d._router = FakeRouter("routine", 0.9, 0.0)  # a router without register()
    d._register_checkpoint(d._router)
    assert d._checkpoint == "english" and "checkpoint_dir failed" in (d._checkpoint_note or "")
    d._ready.set()
    decision = d.decide("x")
    assert decision.label == "routine" and decision.available
    assert "using english" in d.status


def test_checkpoint_defaults_are_the_shipped_checkpoint():
    from jevmon.config import load_config
    from jevmon.monitor import build_decider

    cfg = load_config()
    assert LayaConfig().checkpoint == "english" and LayaConfig().checkpoint_dir == ""
    assert cfg.judge.checkpoint == "english" and cfg.judge.checkpoint_dir == ""
    decider = build_decider(cfg)
    assert decider.config.checkpoint == "english" and decider.config.checkpoint_dir == ""

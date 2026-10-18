"""The shared monitor runtime: exec source -> context builder -> Laya judge -> store.

`Monitor` is the piece both `jevmon exec` and `jevmond` drive. It owns the pipeline, the context
builder, the judge (the only thing that assigns severity), the exec source and the model load
thread, and exposes a status snapshot plus pause/resume and a blocking `run_forever`. Nothing here
decides severity: it moves contexts to the model and reports counters.

`jevmon exec` prints crit verdicts (and, with `--verbose`, every event) to stdout; the daemon passes no
stream and instead reads events back out of the store over its control socket.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TextIO

from jevmon.allow import AllowList, args_of
from jevmon.config import Config, edit_allow, load_config
from jevmon.core import (
    EVENT_KIND_EXEC,
    BufferedCounts,
    Counts,
    MemoryCounts,
    Pipeline,
    SqliteStore,
    StoreSink,
)
from jevmon.detect.context import ContextBuilder, EnvironmentStats, ExecObservation
from jevmon.detect.fake import FakeDecider
from jevmon.detect.judge import SAFE, Decider, Judge
from jevmon.detect.stats import ForkRateBaseline
from jevmon.detect.system_facts import RealSystemFacts
from jevmon.events import Event, JsonlSink, MinSeveritySink, Severity, Sink
from jevmon.laya.backbone import BackboneConfig, BackboneDecider
from jevmon.laya.decider import LayaConfig, LayaDecider
from jevmon.laya.head import DEFAULT_HEAD, DEFAULT_LORA_HEAD
from jevmon.sources.ebpf import EbpfSource
from jevmon.sources.exec import KIND_SUMMARY, ExecSettings, ExecSource
from jevmon.sources.proc import RealProcReader
from jevmon.sources.replay import ReplaySource


def laya_config(j) -> LayaConfig:
    """The Laya decider's config from `[judge]`.

    `model = "laya"` uses the packaged linear head (`tuned` empty); `model = "laya-lora"` merges
    the packaged LoRA adapter into the encoder and adopts its bundled head, with `tuned` (if set)
    only replacing that classifier. `backbone_config` handles the HF backbones.
    """
    lora = ""
    tuned = j.tuned or DEFAULT_HEAD
    if j.model == "laya-lora":
        lora = DEFAULT_LORA_HEAD
        tuned = j.tuned  # empty = the adapter's bundled head
    return LayaConfig(
        categories=dict(j.categories),
        routine=SAFE,
        attention_question=j.question,
        attack_min=1.0,
        offline=True,  # never touch the network; a cached checkpoint or nothing
        threads=j.threads,
        threads_gpu=j.threads_gpu,
        device=j.device,
        dtype=j.dtype,
        tuned=tuned,
        lora=lora,
        checkpoint=j.checkpoint,
        checkpoint_dir=j.checkpoint_dir,
    )


def backbone_config(j) -> BackboneConfig:
    """The Hugging Face backbone decider's config from `[judge]`; `tuned` empty = packaged head
    (for `ettin-lora`, the adapter's bundled classifier)."""
    return BackboneConfig(
        key=j.model,
        categories=dict(j.categories),
        routine=SAFE,
        head=j.tuned or ("" if j.model.endswith("-lora") else j.model),
        device=j.device,
        dtype=j.dtype,
        threads=j.threads,
        threads_gpu=j.threads_gpu,
    )


def decider_settings(j):
    """The decider-specific settings for `[judge]` (LayaConfig or BackboneConfig)."""
    return laya_config(j) if j.model in ("laya", "laya-lora") else backbone_config(j)


def build_decider(config: Config) -> Decider:
    """Construct the judge from `[judge]`: the fake stand-in, Laya, or an HF backbone judge."""
    j = config.judge
    if j.mode == "fake":
        return FakeDecider(j)
    if j.model in ("laya", "laya-lora"):
        return LayaDecider(laya_config(j))
    return BackboneDecider(backbone_config(j))


def build_source(
    settings: ExecSettings, emit: Callable[[Event], None]
) -> ExecSource | ReplaySource | EbpfSource:
    """Construct the event source from `[exec] source`: the helper stream, the `/proc` poller or
    a replay."""
    if settings.source == "replay":
        return ReplaySource(settings, emit=emit)
    if settings.source == "ebpf":
        return EbpfSource(settings, emit=emit)
    return ExecSource(RealProcReader(settings.proc_root), settings, emit=emit)


class Monitor:
    """Run the exec pipeline in a background thread; report status and allow pause/resume."""

    def __init__(
        self,
        config: Config,
        store: SqliteStore | None,
        *,
        proc_root: str | None = None,
        decider: Decider | None = None,
        config_path: str | None = None,
        summary_stream: TextIO | None = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.store = store
        self.config_path = config_path
        self._proc_root = proc_root
        self._decider = decider
        self._summary_stream = summary_stream
        self._verbose = verbose
        self.settings: ExecSettings = config.exec
        if proc_root:  # an alternate /proc tree (tests, --proc-root) means the poller
            self.settings = replace(config.exec, proc_root=proc_root, source="proc")
        self.pipeline: Pipeline | None = None
        self.builder: ContextBuilder | None = None
        self.decider: Decider | None = None
        self.judge: Judge | None = None
        self.store_sink: StoreSink | None = None
        self.buffered_counts: BufferedCounts | None = None
        self.source: ExecSource | ReplaySource | EbpfSource | None = None
        self.allow = AllowList.from_strings(config.filter.allow)
        self._allow_lock = threading.Lock()
        self.env = EnvironmentStats()
        self.fork_baseline = ForkRateBaseline()
        self.started_at: float | None = None
        self.paused = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        cfg = self.config
        # Build the source first: a bad recording must fail before any thread starts.
        self.source = build_source(self.settings, emit=self._on_event)
        sinks: list[Sink] = []
        if self._summary_stream is not None:
            threshold = Severity.SAFE if self._verbose else Severity.CRIT
            sinks.append(MinSeveritySink(JsonlSink(stream=self._summary_stream), threshold))
        if self.store is not None:
            ebpf = self.settings.source == "ebpf"
            self.store_sink = StoreSink(
                self.store,
                cfg.store.store_exec_events,
                repeat_seconds=cfg.store.coalesce_seconds if ebpf else 0.0,
            )
            sinks.append(self.store_sink)
        self.pipeline = Pipeline(sinks, capacity=cfg.store.queue_capacity)
        self.pipeline.start()

        counts: Counts
        if self.store is None:
            counts = MemoryCounts()
        elif self.settings.source == "ebpf":
            # hundreds of execs a second: counts are kept in memory and written in batches
            self.buffered_counts = BufferedCounts(self.store)
            self.buffered_counts.start()
            counts = self.buffered_counts
        else:
            counts = self.store
        self.builder = ContextBuilder(counts, RealSystemFacts(), cfg.context)
        self.decider = self._decider if self._decider is not None else build_decider(cfg)
        self.judge = Judge(self.decider, cfg.judge, emit=self.pipeline.submit)
        self.judge.start()
        threading.Thread(target=self.decider.load, daemon=True, name="laya-load").start()
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run_source, name="jevmon-source", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        """True while the source thread is alive: a replay ends, the `/proc` poller does not."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def run_forever(self) -> None:
        self.start()
        self._stop.wait()
        self.stop()

    def _run_source(self) -> None:
        assert self.source is not None and self.judge is not None and self.pipeline is not None
        try:
            self.source.run(self._stop)
        finally:
            self.judge.stop()
            self.pipeline.stop()
            if self.buffered_counts is not None:
                self.buffered_counts.close()

    # -- control --------------------------------------------------------------------------

    def pause(self) -> None:
        self.paused = True
        if self.judge is not None:
            self.judge.pause()

    def resume(self) -> None:
        self.paused = False
        if self.judge is not None:
            self.judge.resume()

    def reload(self) -> dict:
        new = load_config(self.config_path)
        self.config = new
        with self._allow_lock:
            if tuple(r.text for r in self.allow.rules) != new.filter.allow:
                self.allow = self.allow.replaced(new.filter.allow)
        if self.source is not None and self._proc_root is None:
            self.source.apply_settings(new.exec)
        if self.builder is not None:
            self.builder.settings = new.context
        if self.judge is not None:
            self.judge.settings = new.judge
        if self.decider is not None:
            reconfigure = getattr(self.decider, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(decider_settings(new.judge))
        return {"reloaded": True, "config": str(self.config_path or "")}

    # -- allow list -----------------------------------------------------------------------

    def allow_list(self) -> dict:
        with self._allow_lock:
            return self._allow_view()

    def allow_add(self, rule: str) -> dict:
        """Append `rule` to the config file's `[filter] allow` and apply it live."""
        return self._allow_edit(add=rule)

    def allow_remove(self, rule: str) -> dict:
        return self._allow_edit(remove=rule)

    def _allow_edit(self, *, add: str | None = None, remove: str | None = None) -> dict:
        with self._allow_lock:
            rules, changed = edit_allow(self.config_path, add=add, remove=remove)
            if tuple(r.text for r in self.allow.rules) != tuple(rules):
                self.allow = self.allow.replaced(rules)
            self.config = replace(
                self.config, filter=replace(self.config.filter, allow=tuple(rules))
            )
            return {**self._allow_view(), "changed": changed}

    def _allow_view(self) -> dict:
        return {
            "rules": [rule.text for rule in self.allow.rules],
            "count": len(self.allow.rules),
            "hits": self.allow.hit_counts(),
        }

    def status(self) -> dict:
        source = self.source
        judge = self.judge
        extra = getattr(source, "status", None)
        return {
            "exec_source": self.settings.source,
            **(extra() if callable(extra) else {}),
            "uptime": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            "model_status": self._model_status(),
            "judge_device": self._judge_state("active_device"),
            "judge_dtype": self._judge_state("active_dtype"),
            "effective_interval": round(source.effective_interval, 4) if source else 0.0,
            "miss_ratio": round(source.miss_ratio, 4) if source else 0.0,
            "observed": source.observed_total if source else 0,
            "estimated_missed": source.missed_total if source else 0,
            "tracked": source.tracked if source else 0,
            "coalesced": self.store_sink.coalesced if self.store_sink else 0,
            "judged": judge.judged if judge else 0,
            "cached": judge.cached if judge else 0,
            "dropped": judge.dropped if judge else 0,
            "unavailable": judge.unavailable if judge else 0,
            "queue_depth": judge.depth if judge else 0,
            "queue_age": round(judge.oldest_age, 1) if judge else 0.0,
            "avg_batch": round(judge.avg_batch, 2) if judge else 0.0,
            "ms_per_context": round(judge.ms_per_context, 1) if judge else 0.0,
            "allow_rules": len(self.allow.rules),
            "allow_skipped": self.allow.skipped,
            "allow_hits": self.allow.hit_counts(),
            "paused": self.paused,
            "store": self.store.path if self.store is not None else "",
            "config": str(self.config_path or ""),
            "running": self.running,
        }

    def _model_status(self) -> str:
        decider = self.decider if self.decider is not None else self._decider
        if decider is None:
            return "unloaded"
        return str(getattr(decider, "status", "unknown"))

    def _judge_state(self, attr: str) -> str:
        """The decider's device/dtype ("cpu"/"gpu", "fp32"/...); "" before the model loads."""
        decider = self.decider if self.decider is not None else self._decider
        return str(getattr(decider, attr, "") or "")

    # -- event flow -----------------------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        assert self.builder is not None and self.judge is not None
        if event.kind == KIND_SUMMARY:
            self.env.fork_rate = float(event.fields.get("forks_per_s", 0.0))
            self.fork_baseline.update(self.env.fork_rate)
            self.env.fork_baseline = self.fork_baseline.ewma
            self.env.miss_ratio = float(event.fields.get("miss_ratio", 0.0))
            if self._verbose and self._summary_stream is not None and self.pipeline is not None:
                self.pipeline.submit(event)
            return
        if event.kind == EVENT_KIND_EXEC:
            with self._allow_lock:
                allow = self.allow
                rule = allow.check(
                    str(event.fields.get("exe", "")), args_of(str(event.fields.get("argv", "")))
                )
                rule_text = allow.rules[rule].text if rule is not None else ""
            if rule is not None:
                if self.config.filter.store_skipped:
                    context = self.builder.observe_exec(ExecObservation.from_event(event), self.env)
                    self.judge.record_unjudged(context, "allow", f"allow rule: {rule_text}")
                return
            if (
                self.pipeline is not None
                and self.store is not None
                and self.config.store.store_exec_events
            ):
                self.pipeline.submit(event)
            self.judge.submit(
                self.builder.observe_exec(ExecObservation.from_event(event), self.env)
            )


__all__ = ["Monitor", "build_decider", "build_source"]

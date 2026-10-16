"""Turn exec observations into compact, sanitized state text for the model.

The builder computes FACTS and STATISTICS; it never judges. Statistics come from persisted
counts (`core.Counts`), structural facts from `system_facts.SystemFacts`, and environment
numbers from the source. They go into the context's `fields` (stored, shown, memoised by the
signature) and, in the order given by `ContextSettings.order`, into the state text the model
reads. Process names appear only as text inside the state, never as branches.

The state is bounded to `model_max` characters (the model sees at most that) and each segment is
cleaned through `sanitize.clean`; argv is expected to be redacted already by the source. The
shipped `order` (header, exe, template, argv) was picked by the zero-shot sweep in
`docs/finetune.md`; putting the statistics back is a config change, not a code change.
"""

import os
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from jevmon.core import KEY_SEP, KIND_EDGE, KIND_EXE, KIND_RARITY, KIND_TEMPLATE, Counts, edge_key
from jevmon.detect.stats import percentile, percentile_rank, surprisal
from jevmon.detect.system_facts import StructuralFacts, SystemFacts
from jevmon.events import Event, Scalar
from jevmon.sanitize import clean, shorten
from jevmon.sources.facts import FACT_NAMES
from jevmon.templates import masked_template

# The rendered parts of a context, in natural order. `render_segments` shows them in
# `ContextSettings.order` (a name missing from that list is dropped); nothing here decides a
# verdict, it is only the text layout the model reads.
EXEC_SEGMENTS = (
    "header",
    "exe",
    "parents",
    "parent_cmd",
    "user",
    "template",
    "argv",
    "seen",
    "surprisal",
    "struct",
    "env",
    *FACT_NAMES,
)
KNOWN_SEGMENTS = frozenset(EXEC_SEGMENTS)
# Label each segment starts with when rendered; also the split points `parse_exec_state` looks
# for in already-rendered text.
SEGMENT_LABELS = {
    "exe": "exe: ",
    "parents": "parents: ",
    "parent_cmd": "parent_cmd: ",
    "user": "user(uid): ",
    "template": "argv(template): ",
    "argv": "argv(redacted): ",
    "seen": "seen: ",
    "surprisal": "surprisal: ",
    "struct": "struct: ",
    "env": "env: ",
    **{name: f"{name}: " for name in FACT_NAMES},
}


@dataclass(frozen=True, slots=True)
class ContextSettings:
    model_max: int = 1000
    surprisal_alpha: float = 0.5
    rarity_window_seconds: float = 3600.0
    template_tokens: int = 24
    recent_write_seconds: float = 300.0
    sibling_seconds: float = 10.0
    # Layout the model reads (prompt formatting, not a decision): `order` is the segment order
    # and doubles as the include list - a name left out is not rendered. Empty = natural order,
    # everything. `sep` joins the segments; `argv_max` caps the argv segment's value length.
    # The shipped order is what won the zero-shot sweep (docs/finetune.md): command only, no
    # statistics.
    order: tuple[str, ...] = ("header", "exe", "template", "argv")
    # Display order of environment variables when an environment segment is shown: names listed
    # here come first, in this order, then the rest in their native order. A DISPLAY ORDER, not a
    # judging rule: nothing is flagged or scored by it and the model still decides.
    env_priority: tuple[str, ...] = (
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "PATH",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "NODE_OPTIONS",
        "PERL5OPT",
        "RUBYOPT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "SHELL",
    )
    sep: str = " "
    argv_max: int = 300

    def __post_init__(self) -> None:
        unknown = set(self.order) - KNOWN_SEGMENTS
        if unknown:
            raise ValueError(f"unknown context segment(s): {', '.join(sorted(unknown))}")
        if not isinstance(self.sep, str) or not self.sep:
            raise ValueError(f"sep must be a non-empty string, got {self.sep!r}")
        if self.argv_max < 1:
            raise ValueError(f"argv_max must be positive, got {self.argv_max}")


@dataclass(slots=True)
class EnvironmentStats:
    fork_rate: float = 0.0
    fork_baseline: float = 0.0
    miss_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class Context:
    time: float
    kind: str
    subject: str
    state: str
    signature: str
    fields: dict[str, Scalar]
    # Sanitized, bounded, redacted command line (`basename args...`) for the list.
    command: str = ""


@dataclass(frozen=True, slots=True)
class ExecObservation:
    time: float
    subject: str
    uid: int
    exe: str
    path0: str
    argv0: str
    argv: tuple[str, ...]
    cwd: str
    cgroup: str
    parent_comm: str
    parent_exe: str
    parents: str
    ppid: int = 0
    parent_argv: str = ""
    # launch-time facts the source read (`fact_<name>` event fields, name -> text)
    facts: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_event(cls, event: Event) -> Self:
        f = event.fields
        return cls(
            time=event.time,
            subject=event.subject,
            uid=int(f.get("uid", -1)),
            exe=str(f.get("exe", "")),
            path0=str(f.get("path0", "")),
            argv0=str(f.get("argv0", "")),
            argv=tuple(str(f.get("argv", "")).split()),
            cwd=str(f.get("cwd", "")),
            cgroup=str(f.get("cgroup", "")),
            parent_comm=str(f.get("parent_comm", "")),
            parent_exe=str(f.get("parent_exe", "")),
            parents=str(f.get("parents", "")),
            ppid=int(f.get("ppid", 0) or 0),
            parent_argv=str(f.get("parent_argv", "")),
            facts=tuple(
                (name, str(f[f"fact_{name}"])) for name in FACT_NAMES if f"fact_{name}" in f
            ),
        )

    @property
    def child(self) -> str:
        return self.exe or self.path0 or self.argv0

    @property
    def parent(self) -> str:
        return self.parent_exe or self.parent_comm


def _coarse_bucket(value: float) -> str:
    """Memo bucket for the numbers the model reads: bit-length halved.

    The signature is the memo key for whole verdicts (judge.py), so every boundary in it is a
    re-judgement. A plain bit-length bucket splits values that differ by a factor of two
    (count 4 vs 8, age 32s vs 64s) although the model, which reads the exact number in the state
    text, would not see a different situation there. Halving the index leaves boundaries only
    every factor of about four (2, 8, 32, 128, ...) - still a bucket of the text's own number,
    never a rule about the process. Measured on three recorded streams (the daemon's stored
    contexts, a 10-minute live capture of this box, the lab dataset) by replaying the memo:
    this gear recovers the count/age/rarity-caused misses - hit rate 25.9% -> 28.7%,
    18.9% -> 28.8% and 80.5% -> 83.3% (miss attribution in the commit).
    """
    if value <= 0:
        return "0"
    return str(int(value).bit_length() // 2)


def _f(value: float) -> str:
    return f"{value:.2f}"


def order_env(items: Sequence[tuple[str, str]], priority: Sequence[str]) -> list[tuple[str, str]]:
    """Environment variables in display order: the `priority` names first (matched
    case-insensitively, in that order), then every other variable in its native order."""
    rank = {name.upper(): i for i, name in enumerate(priority)}
    head = sorted((kv for kv in items if kv[0].upper() in rank), key=lambda kv: rank[kv[0].upper()])
    tail = [kv for kv in items if kv[0].upper() not in rank]
    return [*head, *tail]


def render_segments(segments: Sequence[tuple[str, str]], settings: ContextSettings) -> str:
    """Join `(name, text)` segments in `settings.order` (a name left out of a non-empty `order`
    is dropped), cap the argv value at `argv_max`, bound the result by `model_max`."""
    by_name = dict(segments)
    order = settings.order or tuple(by_name)
    argv_label = SEGMENT_LABELS["argv"]
    texts: list[str] = []
    for name in order:
        text = by_name.get(name)
        if text is None:
            continue
        if name == "argv" and text.startswith(argv_label):
            text = argv_label + shorten(text[len(argv_label) :], settings.argv_max)
        texts.append(text)
    return shorten(settings.sep.join(texts), settings.model_max)


def parse_exec_state(state: str) -> list[tuple[str, str]]:
    """Split a rendered exec state back into `(name, text)` segments.

    This exists to re-format contexts that were stored as text (lab datasets): the builder
    renders from fields and never parses. Segments the state does not contain are simply absent,
    so a command-only state parses as command-only segments. It raises when the text has no
    recognised segment or no header, so a bad split fails loudly instead of rendering nonsense;
    callers that re-render should check the round trip (`render_segments(segments, settings)`
    reproduces `state` for the format it was captured with).
    """
    text = state.replace("\n", " ")
    starts: list[tuple[str, str, int]] = []
    pos = 0
    for name in EXEC_SEGMENTS:
        label = SEGMENT_LABELS.get(name)
        if label is None:  # the header has no label; it is whatever precedes the first segment
            continue
        index = text.find(" " + label, pos)
        if index < 0:  # not rendered in this state (config `order` dropped it) - skip it
            continue
        starts.append((name, label, index))
        pos = index + len(label) + 1
    if not starts:
        raise ValueError("exec state has no known segment")
    if not text[: starts[0][2]]:
        raise ValueError("exec state has no header segment")
    out: list[tuple[str, str]] = [("header", text[: starts[0][2]])]
    for n, (name, label, start) in enumerate(starts):
        end = starts[n + 1][2] if n + 1 < len(starts) else len(text)
        out.append((name, text[start + 1 : end]))
    return out


class ContextBuilder:
    """Build model-facing contexts from exec observations."""

    def __init__(
        self,
        counts: Counts,
        facts: SystemFacts,
        settings: ContextSettings | None = None,
    ) -> None:
        self.counts = counts
        self.facts = facts
        self.settings = settings or ContextSettings()
        # exec times per parent pid, for the "siblings" fact (a rate the model reads, not a rule)
        self._siblings: dict[int, deque[float]] = {}

    def _sibling_count(self, ppid: int, now: float) -> int:
        """Execs already seen under this parent within `sibling_seconds`, this one included."""
        if ppid <= 0:
            return 1
        times = self._siblings.setdefault(ppid, deque())
        horizon = now - self.settings.sibling_seconds
        while times and times[0] < horizon:
            times.popleft()
        times.append(now)
        if len(self._siblings) > 2048:  # bounded: drop the quietest parents
            for key in [k for k, v in self._siblings.items() if not v or v[-1] < horizon]:
                del self._siblings[key]
        return len(times)

    def observe_exec(self, observation: ExecObservation, env: EnvironmentStats) -> Context:
        exe = observation.child
        parent = observation.parent
        template = masked_template(observation.argv, self.settings.template_tokens)
        edge = edge_key(parent or "<unknown>", exe or "<unknown>")

        exe_count = self.counts.count(KIND_EXE, exe)
        edge_count = self.counts.count(KIND_EDGE, edge)
        template_count = self.counts.count(KIND_TEMPLATE, template)
        exe_s = surprisal(
            exe_count, self.counts.total(KIND_EXE), self.counts.distinct(KIND_EXE), self._alpha
        )
        edge_s = surprisal(
            edge_count, self.counts.total(KIND_EDGE), self.counts.distinct(KIND_EDGE), self._alpha
        )
        template_s = surprisal(
            template_count,
            self.counts.total(KIND_TEMPLATE),
            self.counts.distinct(KIND_TEMPLATE),
            self._alpha,
        )
        rarity = max(exe_s, edge_s, template_s)
        recent = self.counts.samples(
            KIND_RARITY, observation.time - self.settings.rarity_window_seconds
        )
        pct = percentile_rank(recent, rarity)
        first = self.counts.first_time(KIND_EXE, exe)
        age = (observation.time - first) if first is not None else 0.0
        children = self.counts.distinct_prefix(KIND_EDGE, (parent or "<unknown>") + KEY_SEP)

        facts = self.facts.inspect(exe, observation.time, self.settings.recent_write_seconds)
        hour = time.localtime(observation.time).tm_hour
        parent_template = (
            masked_template(tuple(observation.parent_argv.split()), self.settings.template_tokens)
            if observation.parent_argv
            else ""
        )
        siblings = self._sibling_count(observation.ppid, observation.time)

        fields: dict[str, Scalar] = {
            "pid": observation.subject,
            "exe": exe,
            "basename": os.path.basename(exe),
            "parent": parent,
            "template": template,
            "count_exe": exe_count,
            "count_edge": edge_count,
            "count_template": template_count,
            "first_seen_age": round(age, 1),
            "parent_children": children,
            "parent_template": parent_template,
            "siblings": siblings,
            "surprisal_exe": round(exe_s, 3),
            "surprisal_edge": round(edge_s, 3),
            "surprisal_template": round(template_s, 3),
            "rarity": round(rarity, 3),
            "rarity_pct": round(pct, 1) if pct is not None else -1.0,
            "writable_dir": int(facts.writable_dir),
            "nonroot_owned": int(facts.owned_by_nonroot),
            "recent_write": int(facts.recently_written),
            "mount_type": facts.mount_type,
            "tmpfs_or_noexec": int(facts.on_tmpfs_or_noexec),
            "uid": observation.uid,
            "cgroup": observation.cgroup,
            "cwd": observation.cwd,
            "hour": hour,
            "fork_rate": round(env.fork_rate, 2),
            "fork_baseline": round(env.fork_baseline, 2),
            "miss_ratio": round(env.miss_ratio, 4),
            **{f"fact_{name}": value for name, value in observation.facts},
        }
        state = self._render_exec(observation, fields, facts, env, hour)
        signature = self._exec_signature(fields)

        self.counts.bump(KIND_EXE, exe, observation.time)
        self.counts.bump(KIND_EDGE, edge, observation.time)
        self.counts.bump(KIND_TEMPLATE, template, observation.time)
        self.counts.add_sample(KIND_RARITY, rarity, observation.time)

        return Context(
            time=observation.time,
            kind="exec",
            subject=observation.subject,
            state=state,
            signature=signature,
            fields=fields,
            command=self._command(observation),
        )

    def _command(self, obs: ExecObservation) -> str:
        """A short, sanitized command line for the list: `basename args...` over the redacted
        argv (never raw argv), so `ls -la` reads as `ls -la` and a long path costs little room."""
        redacted = [str(a) for a in obs.argv if a]
        if not redacted:
            return shorten(clean(obs.child, 80), 80)
        name = os.path.basename(redacted[0]) or redacted[0]
        parts = [clean(name, 48), *(clean(a, 64) for a in redacted[1:])]
        return shorten(" ".join(parts), 240)

    @property
    def _alpha(self) -> float:
        return self.settings.surprisal_alpha

    def _exec_signature(self, f: dict[str, Scalar]) -> str:
        return "|".join(
            [
                "e",
                str(f["exe"]),
                str(f["parent"]),
                str(f["template"]),
                _coarse_bucket(float(f["count_exe"])),
                _coarse_bucket(float(f["count_edge"])),
                _coarse_bucket(float(f["count_template"])),
                _coarse_bucket(float(f["first_seen_age"])),
                _coarse_bucket(float(f["rarity"])),
                str(int(f["writable_dir"])),
                str(int(f["nonroot_owned"])),
                str(int(f["recent_write"])),
                str(int(f["tmpfs_or_noexec"])),
                *(str(f[f"fact_{n}"]) for n in self.settings.order if f"fact_{n}" in f),
            ]
        )

    def _render_exec(
        self,
        obs: ExecObservation,
        f: dict[str, Scalar],
        facts: StructuralFacts,
        env: EnvironmentStats,
        hour: int,
    ) -> str:
        segments = [
            ("header", "process launch (tier 0 /proc poller)"),
            ("exe", f"exe: {clean(str(f['exe']), 200)}  basename: {clean(str(f['basename']), 80)}"),
            ("parents", f"parents: {clean(obs.parents, 200)}"),
            (
                "parent_cmd",
                f"parent_cmd: {clean(obs.parent_argv or '-', 240)}  siblings_{int(self.settings.sibling_seconds)}s: {f['siblings']}",
            ),
            (
                "user",
                (
                    f"user(uid): {f['uid']}  cgroup: {clean(obs.cgroup, 80)}  "
                    f"cwd: {clean(obs.cwd, 120)}"
                ),
            ),
            ("template", f"argv(template): {clean(str(f['template']), 300)}"),
            ("argv", f"argv(redacted): {clean(' '.join(obs.argv), 600)}"),
            (
                "seen",
                (
                    f"seen: exe={f['count_exe']} edge={f['count_edge']} "
                    f"template={f['count_template']} first_seen_age={f['first_seen_age']}s "
                    f"parent_children={f['parent_children']}"
                ),
            ),
            (
                "surprisal",
                (
                    f"surprisal: exe={f['surprisal_exe']} edge={f['surprisal_edge']} "
                    f"template={f['surprisal_template']} rarity={f['rarity']} "
                    f"percentile={f['rarity_pct']}"
                ),
            ),
            (
                "struct",
                (
                    f"struct: writable_dir={bool(facts.writable_dir)} "
                    f"nonroot_owned={f['nonroot_owned']} recent_write={f['recent_write']} "
                    f"mount={facts.mount_type} tmpfs_or_noexec={f['tmpfs_or_noexec']}"
                ),
            ),
            (
                "env",
                (
                    f"env: hour={hour} fork_rate={_f(env.fork_rate)} "
                    f"baseline={_f(env.fork_baseline)} miss_ratio={_f(env.miss_ratio)}"
                ),
            ),
        ]
        shown = dict(obs.facts)
        segments += [
            (name, f"{name}: {clean(shown[name], 300)}") for name in FACT_NAMES if name in shown
        ]
        return render_segments(segments, self.settings)


def rarity_percentile(counts: Counts, value: float, since: float, q: float) -> float | None:
    """Convenience: the q-quantile of recent rarity samples (used by tests/tools)."""
    return percentile(counts.samples(KIND_RARITY, since), q)

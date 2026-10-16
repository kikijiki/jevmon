"""Launch-time facts: read from a fake /proc tree, gated by config, shown only when in `order`."""

import os
from pathlib import Path

import pytest

from jevmon.config import ConfigError, load_config
from jevmon.core import MemoryCounts
from jevmon.detect.context import (
    ContextBuilder,
    ContextSettings,
    EnvironmentStats,
    ExecObservation,
    parse_exec_state,
)
from jevmon.detect.system_facts import StructuralFacts
from jevmon.sources.demo import demo_events
from jevmon.sources.exec import KIND_EXEC, ExecSettings, ExecSource
from jevmon.sources.facts import FACT_NAMES, NAMESPACES, FactReader
from jevmon.sources.proc import RealProcReader
from test_exec_source import FakeProc

STATUS = (
    "Name:\tx\nUid:\t1000\t0\t1000\t1000\nCapEff:\t000001ffffffffff\nNoNewPrivs:\t0\n"
    "Seccomp:\t2\nNSpid:\t4821\t1\n"
)


class NoFacts:
    def inspect(self, exe, now, recent_seconds):
        return StructuralFacts()


def add_target(proc: FakeProc, pid: int = 300, *, container: bool = False) -> Path:
    d = proc.add(pid, ["/tmp/runner", "--go"], comm="runner", exe="/tmp/runner (deleted)")
    (d / "status").write_text(STATUS)
    (d / "stat").write_text(
        f"{pid} (runner) S 1 {pid} {pid} 34816 " + " ".join(["0"] * 15) + " 100\n"
    )
    (d / "loginuid").write_text("4294967295")
    (d / "uid_map").write_text(
        "         0       1000          1\n" if container else "0 0 4294967295\n"
    )
    (d / "cgroup").write_text(
        "0::/user.slice/user@1000.service/app.slice/libpod-4f9a1c20b7d3e5a1.scope\n"
    )
    os.symlink("/srv/work", d / "cwd")
    (d / "fd").mkdir()
    os.symlink("/dev/pts/3", d / "fd" / "0")
    os.symlink("pipe:[991]", d / "fd" / "1")
    os.symlink("/dev/null", d / "fd" / "2")
    (d / "ns").mkdir()
    for ns in NAMESPACES:
        target = (
            f"{ns}:[4026539999]"
            if container and ns in ("mnt", "pid", "net")
            else f"{ns}:[4026531840]"
        )
        os.symlink(target, d / "ns" / ns)
    return d


def init_ns(proc: FakeProc) -> None:
    d = proc.root / "1"
    (d / "ns").mkdir(exist_ok=True)
    for ns in NAMESPACES:
        os.symlink(f"{ns}:[4026531840]", d / "ns" / ns)


def test_every_fact_is_read_in_one_pass(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    proc.add(1, ["/sbin/init"], exe="/sbin/init")
    init_ns(proc)
    add_target(proc, container=True)
    reader = RealProcReader(proc.root)
    facts = FactReader(reader, FACT_NAMES, self_pid=1).read(
        300, "/tmp/runner", "/tmp/runner (deleted)"
    )
    assert facts["exe_state"] == "deleted"
    assert facts["argv0_vs_exe"] == "match"
    assert facts["stdio"] == "0=tty 1=pipe 2=null"
    assert facts["session"] == "tty=yes sid_leader=yes loginuid=unset"
    assert facts["workdir"] == "/srv/work"
    assert facts["cgroup_path"] == "user@1000.service/app.slice/libpod-<id>.scope"
    assert facts["privileges"] == "uid=1000 euid=0 capeff=000001ffffffffff nnp=0 seccomp=2"
    assert facts["namespaces"].startswith(
        "differs(mnt,pid,net) [mnt=differs pid=differs net=differs"
    )
    assert "uid_map=remapped" in facts["namespaces"] and "nspid=4821>1" in facts["namespaces"]
    assert facts["exe_file"] == "unknown"  # the fake link target does not exist


def test_vanished_and_unreadable_pids_degrade_to_unknown(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    reader = RealProcReader(proc.root)
    facts = FactReader(reader, FACT_NAMES, self_pid=1).read(999, "", None)
    assert set(facts) == set(FACT_NAMES)
    assert all(
        v.replace("0=", "").replace("1=", "").replace("2=", "")
        in ("unknown", "unknown unknown unknown")
        for v in facts.values()
    )
    assert facts["exe_state"] == "unknown" and facts["session"] == "unknown"


def test_exe_states_and_argv0_mismatch(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    reader = RealProcReader(proc.root)
    fr = FactReader(reader, ("exe_state", "argv0_vs_exe"), self_pid=1)
    assert fr.read(1, "ls", "/usr/bin/cat")["argv0_vs_exe"] == "differs"
    assert fr.read(1, "/bin/cat", "/usr/bin/cat")["argv0_vs_exe"] == "match"
    assert fr.read(1, "x", "/memfd:x")["exe_state"] == "memfd"
    assert fr.read(1, "x", "/memfd:x (deleted)")["exe_state"] == "memfd"
    assert fr.read(1, "x", "/usr/bin/x")["exe_state"] == "normal"


def test_source_reads_facts_only_when_enabled(tmp_path):
    proc = FakeProc(tmp_path / "proc")
    proc.add(1, ["/sbin/init"], exe="/sbin/init")
    init_ns(proc)
    add_target(proc)
    off = ExecSource(RealProcReader(proc.root), ExecSettings(summary_seconds=3600))
    (event,) = [e for e in off.scan_once().events if e.subject == "300"]
    assert not any(k.startswith("fact_") for k in event.fields)
    on = ExecSource(
        RealProcReader(proc.root), ExecSettings(summary_seconds=3600, facts=("privileges", "stdio"))
    )
    (event,) = [e for e in on.scan_once().events if e.subject == "300"]
    assert event.fields["fact_stdio"] == "0=tty 1=pipe 2=null"
    assert "fact_workdir" not in event.fields


def test_config_gates_facts_and_order_must_be_enabled(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[exec]\nfacts = ["stdio", "bogus"]\n')
    with pytest.raises(ConfigError, match="unknown fact"):
        load_config(cfg)
    cfg.write_text('[context]\norder = ["header", "stdio"]\n')
    with pytest.raises(ConfigError, match="not enabled"):
        load_config(cfg)
    cfg.write_text('[exec]\nfacts = ["stdio"]\n[context]\norder = ["header", "exe", "stdio"]\n')
    assert load_config(cfg).exec.facts == ("stdio",)
    assert load_config(cfg).context.order == ("header", "exe", "stdio")


def test_default_context_shows_no_fact_and_default_exec_reads_none():
    assert ExecSettings().facts == ()
    assert not set(ContextSettings().order) & set(FACT_NAMES)


def test_context_segments_follow_order_and_are_memo_relevant():
    base = ExecObservation(
        time=1.0,
        subject="9",
        uid=1,
        exe="/usr/bin/x",
        path0="/usr/bin/x",
        argv0="x",
        argv=("/usr/bin/x",),
        cwd="/",
        cgroup="a",
        parent_comm="p",
        parent_exe="/p",
        parents="p",
        facts=(
            ("privileges", "uid=1 euid=1 capeff=0 nnp=0 seccomp=0"),
            ("stdio", "0=tty 1=tty 2=tty"),
        ),
    )
    b = ContextBuilder(
        MemoryCounts(), NoFacts(), ContextSettings(order=("header", "exe", "stdio", "privileges"))
    )
    ctx = b.observe_exec(base, EnvironmentStats())
    assert "privileges: uid=1 euid=1" in ctx.state and "stdio: 0=tty 1=tty 2=tty" in ctx.state
    names = [n for n, _ in parse_exec_state(ctx.state)]
    assert names == ["header", "exe", "stdio", "privileges"]
    assert "0=tty 1=tty 2=tty" in ctx.signature and ctx.fields["fact_stdio"] == "0=tty 1=tty 2=tty"


def test_every_demo_event_renders_every_fact_segment():
    settings = ContextSettings(order=("header", "exe", *FACT_NAMES), model_max=4000)
    b = ContextBuilder(MemoryCounts(), NoFacts(), settings)
    execs = [e for e in demo_events() if e.kind == KIND_EXEC]
    assert len(execs) > 40
    for event in execs:
        ctx = b.observe_exec(ExecObservation.from_event(event), EnvironmentStats())
        segments = dict(parse_exec_state(ctx.state))
        for name in FACT_NAMES:
            text = segments.get(name, "")
            assert text.startswith(f"{name}: ") and len(text) > len(name) + 2, (name, event.subject)
            assert "unknown" not in text


def test_demo_has_benign_and_odd_shapes():
    execs = [e.fields for e in demo_events(count=200) if e.kind == KIND_EXEC]
    assert any("setuid=yes" in str(f["fact_exe_file"]) for f in execs)
    assert any(f["fact_exe_state"] == "deleted" for f in execs)
    assert any(str(f["fact_namespaces"]).startswith("differs(") for f in execs)
    assert any(f["fact_argv0_vs_exe"] == "differs" for f in execs)

"""Lab tests that do not need podman: manifest parsing/validation, family split pinning, the
scenario-to-observation mapping with fake /proc trees and a fake pod, dataset + report building
with the fake judge, and teardown on failure. Real podman runs are `just lab-smoke` / `just lab`."""

import json
import time
from pathlib import Path

import pytest

from jevmon.core import EVENT_KIND_EXEC
from jevmon.detect.context import Context, ContextSettings
from jevmon.sources.exec import ExecSettings
from jevmon.sources.proc import RealProcReader
from jevmon_lab import dataset, manifest, pod, runner
from jevmon_lab.capture import Capture, Observation
from jevmon_lab.manifest import resolve
from jevmon_lab.mapping import build_container_ids, build_windows, container_id_from_cgroup, tag

TARGET_ID = "1" * 64
ACTOR_ID = "2" * 64

MINIMAL = """
run_seconds = 1
grace_seconds = 0
[limits]
memory = "256m"
[[scenarios]]
id = "s.one"
family = "res"
label = "safe"
container = "target"
expected = 3
at = 0
duration = 1
params = { mb = 7 }
steps = [{ cmd = "echo {mb}", timeout = 5 }]
"""


def write_manifest(tmp_path: Path, text: str = MINIMAL) -> Path:
    path = tmp_path / "scenarios.toml"
    path.write_text(text)
    return path


def make_context(subject: str = "1") -> Context:
    return Context(
        time=time.time(),
        kind="exec",
        subject=subject,
        state="process launch (tier 0 /proc poller)\nexe: /bin/sleep",
        signature=f"e|{subject}",
        fields={"exe": "/bin/sleep"},
    )


# -- manifest -----------------------------------------------------------------------------


def test_manifest_parses_and_defaults(tmp_path: Path) -> None:
    mf = manifest.load(write_manifest(tmp_path))
    assert mf.run_seconds == 1 and mf.grace_seconds == 0
    assert mf.limits.memory == "256m"
    (s,) = mf.scenarios
    assert s.id == "s.one" and s.family == "res" and s.label == "safe"
    assert s.container == "target" and s.expected == 3
    assert s.commands() == ["echo 7"]  # placeholders substituted from params
    assert manifest.split_of(s.family) in ("train", "heldout")


def test_excluded_scenario_has_no_label_and_needs_a_reason(tmp_path: Path) -> None:
    excluded = MINIMAL.replace('label = "safe"', 'excluded = "rate, not a launch"')
    (s,) = manifest.load(write_manifest(tmp_path, excluded)).scenarios
    assert s.label == "" and s.excluded == "rate, not a launch"
    assert manifest.resolve((s,), 0.0)[0].excluded == s.excluded
    with pytest.raises(manifest.ManifestError, match="label"):
        manifest.load(write_manifest(tmp_path, MINIMAL.replace('label = "safe"\n', "")))
    both = MINIMAL.replace('label = "safe"', 'label = "safe"\nexcluded = "x"')
    with pytest.raises(manifest.ManifestError, match="excluded"):
        manifest.load(write_manifest(tmp_path, both))


def test_manifest_rejects_bad_values(tmp_path: Path) -> None:
    bad_label = MINIMAL.replace('label = "safe"', 'label = "panic"')
    with pytest.raises(manifest.ManifestError, match="label"):
        manifest.load(write_manifest(tmp_path, bad_label))
    dup = MINIMAL + "\n[[scenarios]]" + MINIMAL.split("[[scenarios]]", 1)[1]
    with pytest.raises(manifest.ManifestError, match="duplicate"):
        manifest.load(write_manifest(tmp_path, dup))
    bad_container = MINIMAL.replace('container = "target"', 'container = "db"')
    with pytest.raises(manifest.ManifestError, match="container"):
        manifest.load(write_manifest(tmp_path, bad_container))
    bad_expected = MINIMAL.replace("expected = 3", "expected = -1")
    with pytest.raises(manifest.ManifestError, match="expected"):
        manifest.load(write_manifest(tmp_path, bad_expected))


def test_manifest_rejects_inconsistent_family_labels(tmp_path: Path) -> None:
    two = (
        MINIMAL
        + """
[[scenarios]]
id = "s.two"
family = "res"
label = "crit"
at = 50
duration = 1
steps = [{ cmd = "true", timeout = 1 }]
"""
    )
    with pytest.raises(manifest.ManifestError, match="one family must have one"):
        manifest.load(write_manifest(tmp_path, two))


def test_manifest_rejects_overlapping_windows_in_one_container(tmp_path: Path) -> None:
    two = (
        MINIMAL
        + """
[[scenarios]]
id = "s.two"
family = "other"
label = "safe"
container = "target"
at = 0
duration = 5
steps = [{ cmd = "true", timeout = 1 }]
"""
    )
    with pytest.raises(manifest.ManifestError, match="overlap"):
        manifest.load(write_manifest(tmp_path, two))
    # the same windows are fine when the two scenarios live in different containers
    other = two.replace(
        '''id = "s.two"
family = "other"
label = "safe"
container = "target"''',
        '''id = "s.two"
family = "other"
label = "safe"
container = "actor"''',
    ).replace("run_seconds = 1", "run_seconds = 6")
    assert manifest.load(write_manifest(tmp_path, other))


def test_manifest_rejects_window_past_run_seconds(tmp_path: Path) -> None:
    text = MINIMAL.replace("run_seconds = 1", "run_seconds = 0")
    with pytest.raises(manifest.ManifestError, match="run_seconds"):
        manifest.load(write_manifest(tmp_path, text))


def test_format_cmd_substitutes_and_leaves_unknown() -> None:
    assert manifest.format_cmd("echo {mb} {n}", {"mb": 5, "n": "x"}) == "echo 5 x"
    assert (
        manifest.format_cmd('curl -w "%{http_code}" {p}', {"p": "/x"})
        == 'curl -w "%{http_code}" /x'
    )


def test_selected_only_compacts_schedule(tmp_path: Path) -> None:
    mf = manifest.load(write_manifest(tmp_path))
    mf2 = mf.selected(only=("res",))
    (s,) = mf2.scenarios
    assert s.at == 5.0
    assert mf2.run_seconds >= s.at + s.duration + mf2.grace_seconds
    assert mf2.run_seconds < 60


def test_shipped_manifest_is_valid_and_complete() -> None:
    mf = manifest.load(Path(__file__).resolve().parent.parent / "scenarios.toml")
    ids = {s.id for s in mf.scenarios}
    assert len(mf.scenarios) >= 30
    assert set(manifest.SMOKE_IDS) <= ids
    assert {s.label for s in mf.scenarios if s.label} == set(manifest.LABELS)
    assert all(s.excluded for s in mf.scenarios if not s.label)
    assert any(s.excluded for s in mf.scenarios)  # the rate/resource scenarios stay in the manifest
    assert {"target", "actor"} == {s.container for s in mf.scenarios}
    families = {s.family for s in mf.scenarios}
    assert len(families) >= 12
    # split consistency and both sides present
    split = {f: manifest.split_of(f) for f in families}
    assert "train" in split.values() and "heldout" in split.values()
    # expected counts are hand-counted payload launches
    assert all(s.expected >= 0 for s in mf.scenarios)
    assert sum(1 for s in mf.scenarios if s.expected == 0) <= 2  # only uncounted background


def test_family_split_is_pinned() -> None:
    """Frozen assignments: a family moving between train and heldout is a deliberate change."""
    expected = {
        "agent_swarm": "heldout",
        "build_fanout": "train",
        "cpu_spin": "heldout",
        "cron_timer": "heldout",
        "curl_pipe": "train",
        "debugger_launch": "train",
        "fork_storm": "train",
        "hidden_cron": "train",
        "home_script": "heldout",
        "idle_daemon": "heldout",
        "inline_encoded": "train",
        "interactive_shell": "train",
        "mem_grower": "train",
        "new_tool": "heldout",
        "pkg_install": "train",
        "ssh_session": "heldout",
        "tmpfs_copy": "heldout",
        "unexpected_parent": "train",
    }
    actual = {f: manifest.split_of(f) for f in expected}
    assert actual == expected


# -- mapping: container cgroup + tag window ------------------------------------------------


def test_container_id_from_cgroup_scopes_and_nested_units() -> None:
    scoped = (
        "0::/user.slice/user-1000.slice/user@1000.service/user.slice/"
        f"libpod-{TARGET_ID}.scope/container\n"
    )
    assert container_id_from_cgroup(scoped) == TARGET_ID
    assert container_id_from_cgroup("0::/user.slice/user-1000.slice/session-3.scope\n") is None
    assert container_id_from_cgroup("") is None


def _windows(t0: float):
    resolved = resolve(
        (
            manifest.Scenario(
                id="w.one",
                family="f",
                label="safe",
                container="target",
                at=0,
                duration=10,
                steps=(),
            ),
            manifest.Scenario(
                id="w.two",
                family="f",
                label="safe",
                container="target",
                at=20,
                duration=10,
                steps=(),
            ),
        ),
        t0,
    )
    return build_windows(resolved, grace=3.0)


def test_tag_maps_by_container_and_window() -> None:
    t0 = 1_000.0
    windows = _windows(t0)
    ids = build_container_ids(TARGET_ID, ACTOR_ID)
    target_cg = f"0::/user.slice/user-1000.slice/user@1000.service/user.slice/libpod-{TARGET_ID}.scope/container"
    actor_cg = f"0::/x/libpod-{ACTOR_ID}.scope/container"
    # inside the first window (grace included)
    assert tag(target_cg, t0 + 5, windows, ids) == ("w.one", "target")
    assert tag(target_cg, t0 + 12.9, windows, ids) == ("w.one", "target")
    # grace tail ends at start + duration + 3
    assert tag(target_cg, t0 + 13.5, windows, ids) == (None, "target")
    # between windows: lab background, still a lab container exec
    assert tag(target_cg, t0 + 16, windows, ids) == (None, "target")
    # second window
    assert tag(target_cg, t0 + 25, windows, ids) == ("w.two", "target")
    # the actor container never matches target windows
    assert tag(actor_cg, t0 + 5, windows, ids) == (None, "actor")
    # host process
    assert tag("0::/user.slice/.../app-org.kde.konsole.scope/tab.scope", t0 + 5, windows, ids) == (
        None,
        None,
    )
    # an unrelated container (not ours) is not lab traffic
    other = f"0::/user.slice/libpod-{ACTOR_ID}.scope/container"
    ids_only_target = {TARGET_ID: "target"}
    assert tag(other, t0 + 5, windows, ids_only_target) == (None, None)


# -- capture over a fake /proc tree --------------------------------------------------------


def _fake_proc(root: Path, entries: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "stat").write_text("cpu 1 2\nprocesses 1000\n")
    for e in entries:
        d = root / str(e["pid"])
        d.mkdir(exist_ok=True)
        rest = [
            e.get("state", "S"),
            str(e.get("ppid", 1)),
            *(["0"] * 17),
            str(e.get("starttime", 10)),
        ]
        (d / "stat").write_text(f"{e['pid']} ({e['comm']}) " + " ".join(rest) + "\n")
        (d / "cmdline").write_bytes(e["argv"][0] + b"\0" + b"\0".join(e["argv"][1:]) + b"\0")
        (d / "status").write_text("Name:\t" + e["comm"] + "\nUid:\t0\t0\t0\t0\n")
        (d / "cgroup").write_text(e["cgroup"])
        (d / "exe").symlink_to(e["exe"])
        (d / "cwd").symlink_to(e["cwd"])


def _capture_for(tmp_path: Path) -> tuple[Capture, float]:
    t0 = time.time() - 1  # window started a second ago, so the scan lands inside it
    windows = _windows(t0)
    ids = build_container_ids(TARGET_ID, ACTOR_ID)
    target_cg = f"0::/user.slice/user-1000.slice/user@1000.service/user.slice/libpod-{TARGET_ID}.scope/container\n"
    host_cg = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/tab.scope\n"
    _fake_proc(
        tmp_path / "proc",
        [
            {
                "pid": 101,
                "comm": "sleep",
                "ppid": 1,
                "argv": [b"/usr/bin/sleep", b"3"],
                "cgroup": target_cg,
                "exe": "/usr/bin/sleep",
                "cwd": "/",
            },
            {
                "pid": 102,
                "comm": "nginx",
                "ppid": 1,
                "argv": [b"nginx", b"-g", b"daemon off;"],
                "cgroup": target_cg,
                "exe": "/usr/sbin/nginx",
                "cwd": "/",
            },
            {
                "pid": 103,
                "comm": "bash",
                "ppid": 1,
                "argv": [b"/bin/bash"],
                "cgroup": host_cg,
                "exe": "/bin/bash",
                "cwd": "/home/user",
            },
        ],
    )
    capture = Capture(
        ExecSettings(proc_root=str(tmp_path / "proc")),
        ContextSettings(),
        windows,
        ids,
        reader=RealProcReader(tmp_path / "proc"),
    )
    return capture, t0


def test_capture_records_only_lab_container_execs(tmp_path: Path) -> None:
    capture, _t0 = _capture_for(tmp_path)
    result = capture.source.scan_once()
    for event in result.events:
        assert event.kind == EVENT_KIND_EXEC
        capture._on_event(event)
    # pid 102 sits in the window between t0 and t0+10 (grace), pid 101 too if its scan time is
    # inside; pid 103 is a host process and must not be recorded at all.
    assert capture.host_events == 1
    lab = [o for o in capture.observations]
    assert {o.container for o in lab} <= {"target"}
    assert 101 in {int(o.context.subject) for o in lab}
    assert 103 not in {int(o.context.subject) for o in lab}
    # the context text is the real builder's state
    assert all("process launch" in o.context.state for o in lab)
    stats = capture.poller_stats()
    assert stats["lab_contexts"] == len(lab)
    assert stats["host_events_not_recorded"] == 1


def test_capture_tagging_follows_the_window(tmp_path: Path) -> None:
    # shift the windows into the future: lab container execs become untagged background
    capture, _ = _capture_for(tmp_path)
    capture.windows = tuple(
        type(w)(w.scenario_id, w.container, w.start + 10_000, w.end + 10_000)
        for w in capture.windows
    )
    result = capture.source.scan_once()
    for event in result.events:
        capture._on_event(event)
    assert capture.observations
    assert all(o.scenario_id is None for o in capture.observations)
    assert capture.background_events == len(capture.observations)


# -- pod / runner lifecycle ---------------------------------------------------------------


class FakeExecutor:
    def __init__(self, stdout: str = "running\n") -> None:
        self.runs: list[list[str]] = []
        self.stdout = stdout

    def run(self, argv: list[str], timeout: float):
        self.runs.append(argv)
        if argv[:2] == ["podman", "inspect"]:
            out = self.stdout
        else:
            out = "running\n"

        class R:
            returncode = 0

        R.stdout = out
        R.stderr = ""
        return R()


def test_pod_teardown_argv() -> None:
    argv = pod.teardown_argv(["a", "b"], "net")
    assert argv[0] == ["podman", "rm", "-f", "a", "b"]
    assert argv[1] == ["podman", "network", "rm", "--force", "net"]


def test_pod_up_then_down_uses_executor() -> None:
    ex = FakeExecutor()
    p = pod.Pod("test", manifest.Limits(), executor=ex)
    p.up(ready_timeout=5)
    p.down()
    joined = [" ".join(a) for a in ex.runs]
    assert any("network create --internal jevlab-test" in j for j in joined)
    assert any("--memory-swap 512m" in j for j in joined)
    assert any("--systemd=always" in j for j in joined)
    assert any(j == "podman rm -f jevlab-test-target jevlab-test-actor" for j in joined)
    assert any(j == "podman network rm --force jevlab-test" for j in joined)


def test_pod_container_ids_reads_inspect() -> None:
    ex = FakeExecutor(stdout=f"{TARGET_ID}\n{ACTOR_ID}\n")
    p = pod.Pod("test", manifest.Limits(), executor=ex)
    assert p.container_ids() == {TARGET_ID: "target", ACTOR_ID: "actor"}
    assert any(a[:3] == ["podman", "inspect", "--format"] for a in ex.runs)
    # non-hex inspect output maps nothing rather than raising
    ex2 = FakeExecutor(stdout="running\n")
    assert pod.Pod("t2", manifest.Limits(), executor=ex2).container_ids() == {}


class FakePod:
    def __init__(self) -> None:
        self.execs: list[tuple[str, str]] = []
        self.downed = False
        self.up_raises = False

    def up(self, ready_timeout: float = 90.0) -> None:
        if self.up_raises:
            raise RuntimeError("boom")

    def container_ids(self) -> dict[str, str]:
        return {TARGET_ID: "target", ACTOR_ID: "actor"}

    def exec_target(self, cmd: str, timeout: float = 60.0):
        self.execs.append(("target", cmd))
        return pod.CommandResult(["x"], 0, "", "")

    def exec_actor(self, cmd: str, timeout: float = 60.0):
        self.execs.append(("actor", cmd))
        return pod.CommandResult(["x"], 0, "", "")

    def down(self) -> None:
        self.downed = True


class FakeCapture:
    def __init__(self, observations: tuple[Observation, ...] = ()) -> None:
        self.observations = list(observations)
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 10.0) -> None:
        self.stopped = True

    def poller_stats(self) -> dict:
        return {
            "observed_total": 7,
            "estimated_missed": 3,
            "miss_ratio": 0.3,
            "effective_interval": 1.2,
            "tracked": 12,
            "lab_contexts": len(self.observations),
            "tagged_contexts": sum(1 for o in self.observations if o.scenario_id),
            "untagged_container_contexts": sum(1 for o in self.observations if not o.scenario_id),
            "host_events_not_recorded": 0,
        }


def test_runner_writes_contexts_and_scenarios(tmp_path: Path) -> None:
    mf = manifest.load(write_manifest(tmp_path))
    fake = FakePod()
    capture = FakeCapture(
        (
            Observation(make_context("101"), "s.one", "target"),
            Observation(make_context("102"), "s.one", "target"),
            Observation(make_context("103"), None, "actor"),
        )
    )
    result = runner.run(mf, tmp_path / "out", pod=fake, capture=capture)
    assert result.meta["scenarios"] == 1 and result.meta["contexts"] == 3
    assert fake.downed and capture.started and capture.stopped
    assert fake.execs == [("target", "echo 7")]  # steps ran with params substituted
    out = tmp_path / "out"
    contexts = [json.loads(ln) for ln in (out / "contexts.jsonl").read_text().splitlines()]
    assert len(contexts) == 3 and contexts[0]["scenario_id"] == "s.one"
    records = [json.loads(ln) for ln in (out / "scenarios.jsonl").read_text().splitlines()]
    (rec,) = records
    assert rec["observed"] == 2 and rec["missed"] == 1 and rec["observed_any"] is True
    assert rec["label"] == "safe" and rec["family"] == "res"
    assert rec["split"] == manifest.split_of("res")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["contexts"] == 3 and meta["poller"]["miss_ratio"] == 0.3


def test_runner_tears_down_on_up_failure(tmp_path: Path) -> None:
    mf = manifest.load(write_manifest(tmp_path))
    fake = FakePod()
    fake.up_raises = True
    with pytest.raises(RuntimeError, match="boom"):
        runner.run(mf, tmp_path / "out", pod=fake, capture=FakeCapture())
    assert fake.downed is True


def test_runner_stops_capture_on_failure(tmp_path: Path) -> None:
    class ExplodingPod(FakePod):
        def exec_target(self, cmd: str, timeout: float = 60.0):
            raise KeyboardInterrupt

    mf = manifest.load(write_manifest(tmp_path))
    capture = FakeCapture()
    with pytest.raises(KeyboardInterrupt):
        runner.run(mf, tmp_path / "out", pod=ExplodingPod(), capture=capture)
    assert capture.stopped is True


# -- dataset + report (fake judge) ---------------------------------------------------------


def _records() -> list[dict]:
    def rec(i: str, fam: str, label: str, expected: int, observed: int) -> dict:
        return {
            "id": i,
            "family": fam,
            "label": label,
            "container": "target",
            "expected": expected,
            "split": manifest.split_of(fam),
            "window_start": 0.0,
            "window_end": 10.0,
            "start": 0.0,
            "end": 5.0,
            "skip": "",
            "errors": [],
            "observed": observed,
            "missed": max(0, expected - observed),
            "observed_any": observed > 0,
        }

    return [
        rec("r.one", "build_fanout", "safe", 4, 4),
        rec("r.two", "build_fanout", "safe", 10, 3),
        rec("s.one", "tmpfs_copy", "crit", 3, 0),
    ]


def _observations() -> tuple[Observation, ...]:
    return (
        Observation(make_context("101"), "r.one", "target"),
        Observation(make_context("102"), "r.one", "target"),
        Observation(make_context("103"), "r.two", "target"),
        Observation(make_context("104"), "s.one", "target"),
        Observation(make_context("105"), None, "actor"),
    )


def test_build_rows_joins_ground_truth() -> None:
    rows = dataset.build_rows(_observations(), _records())
    assert len(rows) == 5
    labelled = [r for r in rows if r["label"]]
    assert len(labelled) == 4
    assert {r["label"] for r in labelled} == {"safe", "crit"}
    assert rows[-1]["label"] is None and rows[-1]["scenario_id"] is None
    assert all(r["observed"] is True for r in rows)
    assert all(r["split"] for r in labelled)


def test_build_rows_re_renders_stored_contexts_with_the_current_settings() -> None:
    raw = dataset.build_rows(_observations(), _records())[0]["context"]
    assert raw == "process launch (tier 0 /proc poller)\nexe: /bin/sleep"
    shipped = dataset.build_rows(_observations(), _records(), ContextSettings())[0]["context"]
    assert shipped == "process launch (tier 0 /proc poller) exe: /bin/sleep"
    full = dataset.build_rows(_observations(), _records(), ContextSettings(order=(), sep="\n"))[0]
    assert full["context"] == raw  # round trip through the capture's own format


def test_fake_judge_is_deterministic_and_fills_rows() -> None:
    rows = dataset.build_rows(_observations(), _records())
    stats = dataset.judge_rows(rows, "fake")
    assert stats["mode"] == "fake" and stats["judged"] == 5
    assert all(r["judged"] and r["verdict"] for r in rows)
    assert all(r["latency_ms"] is not None for r in rows)
    again = dataset.build_rows(_observations(), _records())
    dataset.judge_rows(again, "fake")
    assert [r["verdict"] for r in again] == [r["verdict"] for r in rows]
    with pytest.raises(ValueError, match="judge mode"):
        dataset.judge_rows(rows, "wat")


def test_judge_none_records_without_verdicts() -> None:
    rows = dataset.build_rows(_observations(), _records())
    stats = dataset.judge_rows(rows, "none")
    assert stats == {"mode": "none", "judged": 0, "unavailable": 0, "latencies_ms": []}
    assert all(r["verdict"] is None and not r["judged"] for r in rows)


def test_report_contains_matrix_fpr_and_honest_limits() -> None:
    rows = dataset.build_rows(_observations(), _records())
    stats = dataset.judge_rows(rows, "fake")
    meta = {"run_id": "test-run"}
    poller = {
        "observed_total": 7,
        "estimated_missed": 3,
        "miss_ratio": 0.3,
        "effective_interval": 1.2,
        "tracked": 12,
        "lab_contexts": 5,
        "tagged_contexts": 4,
        "untagged_container_contexts": 1,
        "host_events_not_recorded": 9,
    }
    report = dataset.build_report(_records(), rows, poller, stats, meta)
    assert "# jevmon lab report" in report
    assert "test-run" in report
    assert "## Verdict vs truth (confusion matrix)" in report
    assert "## Over-flagging of safe launches" in report
    assert "## Judge latency" in report
    assert "miss ratio (source, cumulative)" in report and "0.3" in report
    # honest limits: the unobserved suspicious scenario, the missed launches, the untagged
    # background row, the fake judge and the host-global counter are all called out
    assert "no launch was observed" in report
    assert "observed 3 of 10" in report
    assert "background daemons" in report
    assert "fake" in report
    assert "host-global" in report
    assert "`build_fanout`" in report and "`tmpfs_copy`" in report


def test_report_without_judge_says_so() -> None:
    rows = dataset.build_rows(_observations(), _records())
    stats = dataset.judge_rows(rows, "none")
    report = dataset.build_report(_records(), rows, {}, stats, {"run_id": "x"})
    assert "no judged labelled contexts" in report
    assert "no judge ran" in report


def test_designated_marks_only_the_scenarios_own_driver_command() -> None:
    from jevmon_lab.mapping import Window, designated, driver_argvs

    drivers = driver_argvs(("cp /bin/true /tmp/x && /tmp/x 3",))
    window = Window("s.one", "target", 0.0, 10.0, drivers)
    assert designated(window, "sh -c cp /bin/true /tmp/x && /tmp/x 3") is True
    assert designated(window, "sh  -c  cp /bin/true /tmp/x && /tmp/x 3") is True  # whitespace
    assert designated(window, "sleep 3") is False
    assert designated(None, "sleep 3") is None and designated(window, "") is None


def test_settle_manifest_is_heldout_only() -> None:
    """E14 fresh families: valid manifest, three-class labels, every family on the held-out side."""
    mf = manifest.load(Path(__file__).parents[1] / "scenarios_settle.toml")
    assert len(mf.scenarios) >= 100
    families = {s.family for s in mf.scenarios}
    assert len(families) >= 30
    assert all(manifest.split_of(f) == "heldout" for f in families)
    assert {s.label for s in mf.scenarios} == {"safe", "warn", "crit"}
    existing = {
        s.family for s in manifest.load(Path(__file__).parents[1] / "scenarios.toml").scenarios
    }
    assert not families & existing

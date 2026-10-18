"""`jevmon start|stop`: what gets run, as data. Nothing privileged is executed here."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from jevmon.config import load_config
from jevmon.service import (
    ServiceError,
    choose_elevator,
    helper_start_argv,
    plan_start,
    plan_stop,
    validate_helper_path,
)

TOOLS = {
    "systemd-run": "/run/current-system/sw/bin/systemd-run",
    "systemctl": "/run/current-system/sw/bin/systemctl",
    "run0": "/run/current-system/sw/bin/run0",
    "sudo": "/run/wrappers/bin/sudo",
    "pkexec": "/run/wrappers/bin/pkexec",
}
HELPER = "/nix/store/abc-jevmon-exec-0.1.0/bin/jevmon-exec"


def which_of(*names):
    return lambda name: TOOLS[name] if name in names else None


def config(source):
    cfg = load_config()
    extra = {"replay_file": "x.jsonl"} if source == "replay" else {}
    return replace(cfg, exec=replace(cfg.exec, source=source, **extra))


def start(source, answering=False, tools=("systemd-run", "systemctl", "sudo")):
    return plan_start(
        config(source),
        helper_answering=answering,
        uid=1234,
        which=which_of(*tools),
        helper_finder=lambda cfg: HELPER,
    )


def test_ebpf_plans_the_helper_elevation_then_the_daemon():
    plan = start("ebpf")
    helper, daemon = plan.steps
    assert helper.elevated and not daemon.elevated
    assert helper.argv[:3] == (TOOLS["sudo"], TOOLS["systemd-run"], "--unit=jevmon-exec")
    assert daemon.argv == (TOOLS["systemctl"], "--user", "start", "jevmond.service")
    assert any("password prompt" in m for m in plan.messages)


def test_helper_command_is_an_argv_list_with_the_uid_as_an_argument():
    argv = helper_start_argv("/bin/sudo", "/bin/systemd-run", HELPER, 1234, "/run/jevmon/exec.sock")
    assert all(isinstance(a, str) for a in argv)
    tail = argv[argv.index("--") + 1 :]
    assert tail == (HELPER, "--uid", "1234", "--socket", "/run/jevmon/exec.sock")
    props = [a for a in argv if a.startswith("--property=")]
    for needed in (
        "AmbientCapabilities=CAP_BPF CAP_PERFMON CAP_SYS_RESOURCE",
        "CapabilityBoundingSet=CAP_BPF CAP_PERFMON CAP_SYS_RESOURCE",
        "PrivateNetwork=yes",
        "ProtectSystem=strict",
        "ProtectHome=yes",
        "RuntimeDirectory=jevmon",
        "NoNewPrivileges=yes",
    ):
        assert f"--property={needed}" in props
    assert "--collect" in argv and "sh" not in argv and "-c" not in argv


def test_proc_and_replay_never_elevate():
    for source in ("proc", "replay"):
        plan = start(source, tools=("systemctl",))
        assert [s.elevated for s in plan.steps] == [False]
        assert plan.steps[0].argv[-1] == "jevmond.service"


def test_a_running_helper_is_not_started_again():
    plan = start("ebpf", answering=True)
    assert [s.elevated for s in plan.steps] == [False]


def test_elevation_tool_order_is_run0_then_sudo_then_pkexec():
    assert choose_elevator(which_of("run0", "sudo", "pkexec"))[0] == "run0"
    assert choose_elevator(which_of("sudo", "pkexec"))[0] == "sudo"
    assert choose_elevator(which_of("pkexec"))[0] == "pkexec"
    assert choose_elevator(which_of("sudo"))[1] == TOOLS["sudo"]
    with pytest.raises(ServiceError):
        choose_elevator(which_of())


def test_ebpf_without_any_elevation_tool_is_an_error():
    with pytest.raises(ServiceError):
        start("ebpf", tools=("systemd-run", "systemctl"))


def test_a_user_writable_helper_is_refused(tmp_path):
    binary = tmp_path / "jevmon-exec"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    with pytest.raises(ServiceError, match="refusing to run"):
        validate_helper_path(str(binary))
    link = tmp_path / "link"
    link.symlink_to(binary)
    with pytest.raises(ServiceError, match="refusing to run"):
        validate_helper_path(str(link))  # symlinks are resolved first


def test_a_missing_helper_is_refused(tmp_path):
    with pytest.raises(ServiceError, match="not an executable"):
        validate_helper_path(str(tmp_path / "nope"))


@pytest.mark.skipif(not Path("/nix/store").exists(), reason="needs a nix store")
def test_a_store_path_is_accepted():
    real = os.path.realpath("/bin/sh") if Path("/bin/sh").exists() else ""
    if not real.startswith("/nix/store/"):
        pytest.skip("/bin/sh is not a store path here")
    assert validate_helper_path("/bin/sh") == real


def test_stop_elevates_only_for_a_transient_helper():
    transient = plan_stop(
        unit_kind="transient", unit_active=True, which=which_of("systemctl", "sudo")
    )
    assert [s.elevated for s in transient.steps] == [False, True]
    assert transient.steps[1].argv == (
        TOOLS["sudo"],
        TOOLS["systemctl"],
        "stop",
        "jevmon-exec.service",
    )
    installed = plan_stop(unit_kind="installed", unit_active=True, which=which_of("systemctl"))
    assert [s.elevated for s in installed.steps] == [False]
    assert any("just exec-install" in m for m in installed.messages)
    idle = plan_stop(unit_kind="", unit_active=False, which=which_of("systemctl"))
    assert [s.elevated for s in idle.steps] == [False]

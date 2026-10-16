"""Pure parsing of `/proc` files, including comm with spaces/parens and odd bytes."""

from jevmon.sources.proc import (
    parse_cmdline,
    parse_fork_count,
    parse_stat,
    parse_uid,
    unit_from_cgroup,
)


def stat_line(pid: int, comm: str, state: str, ppid: int, starttime: int) -> bytes:
    rest = [state, str(ppid), *(["0"] * 17), str(starttime)]
    return f"{pid} ({comm}) ".encode() + " ".join(rest).encode() + b"\n"


def test_parse_stat_with_spaces_and_parens_in_comm():
    s = parse_stat(stat_line(123, "weird ) name (x", "S", 1, 987654))
    assert s is not None
    assert (s.pid, s.comm, s.state, s.ppid, s.starttime) == (123, "weird ) name (x", "S", 1, 987654)


def test_parse_stat_rejects_truncated_and_garbage():
    assert parse_stat(b"nope") is None
    assert parse_stat(b"1 (x) S 0\n") is None  # too few fields


def test_parse_uid_takes_real_uid():
    assert parse_uid(b"Name:\tx\nUid:\t1000\t1000\t1000\t1000\n") == 1000
    assert parse_uid(b"Name:\tx\n") is None


def test_parse_cmdline_drops_only_trailing_nul():
    assert parse_cmdline(b"/bin/sh\0-c\0echo hi\0") == (b"/bin/sh", b"-c", b"echo hi")
    assert parse_cmdline(b"") == ()
    assert parse_cmdline(b"a\0\0b\0") == (b"a", b"", b"b")  # internal empty arg kept


def test_unit_from_cgroup_recognises_service_scope_only():
    v2 = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/foo.scope\n"
    assert unit_from_cgroup(v2) == "foo.scope"
    assert unit_from_cgroup("0::/user.slice/user-1000.slice/session-3.scope\n") == "session-3.scope"
    assert unit_from_cgroup("0::/\n") is None
    assert unit_from_cgroup("12:memory:/user.slice\n") is None


def test_unit_from_cgroup_resolves_rootless_container_scope_with_subcgroup():
    # rootless podman on cgroup v2: exec'd processes sit below libpod-<id>.scope/container/...
    # and a systemd container nests real units (init.scope, nginx.service) below that; the
    # container scope is the host-level unit and must win.
    base = (
        "0::/user.slice/user-1000.slice/user@1000.service/user.slice/"
        "libpod-3d2a6cdd768828ffef60231dfc201a51b9223a0bf2e0506612cb3fceb0dea93a.scope"
    )
    scope = "libpod-3d2a6cdd768828ffef60231dfc201a51b9223a0bf2e0506612cb3fceb0dea93a.scope"
    assert unit_from_cgroup(base + "/container\n") == scope
    assert unit_from_cgroup(base + "/container/init.scope\n") == scope
    assert unit_from_cgroup(base + "/container/system.slice/nginx.service\n") == scope


def test_parse_fork_count():
    assert parse_fork_count(b"cpu  1 2 3\nprocesses 123456\nprocs_running 2\n") == 123456
    assert parse_fork_count(b"cpu 1 2 3\n") is None

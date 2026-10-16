"""Structural facts: mount parsing and os.stat-based inspection."""

import os
import time

from jevmon.detect.system_facts import RealSystemFacts, mount_for_path, parse_mounts

MOUNTS = (
    "25 30 0:23 / / rw,relatime - ext4 /dev/root rw,errors=remount-ro\n"
    "26 25 0:24 / /tmp rw,nosuid,nodev - tmpfs tmpfs rw,size=1G\n"
    "27 25 0:25 / /nix/store ro,relatime - ext4 /dev/root ro\n"
    "28 25 0:26 / /data rw,noexec - ext4 /dev/sdb rw"
)


def test_parse_mounts_reads_point_fstype_options():
    mounts = {m.point: m for m in parse_mounts(MOUNTS)}
    assert mounts["/"].fstype == "ext4"
    assert mounts["/tmp"].fstype == "tmpfs"
    assert "nodev" in mounts["/tmp"].options
    assert "noexec" in mounts["/data"].options


def test_mount_for_path_uses_longest_prefix():
    mounts = parse_mounts(MOUNTS)
    assert mount_for_path("/tmp/x/y", mounts).point == "/tmp"
    assert mount_for_path("/data/evil", mounts).point == "/data"
    assert mount_for_path("/usr/bin/ls", mounts).point == "/"
    assert mount_for_path("/nix/store/abc", mounts).point == "/nix/store"


def test_inspect_reports_mode_owner_recent_and_mount(tmp_path):
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(MOUNTS)
    exe = tmp_path / "prog"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    dir_mode = os.stat(tmp_path).st_mode
    facts = RealSystemFacts(mountinfo).inspect(str(exe), now=time.time(), recent_seconds=300)
    assert facts.readable and facts.recently_written
    assert facts.owned_by_nonroot == (os.getuid() != 0)
    tmp_path.chmod(0o777)
    writable = RealSystemFacts(mountinfo).inspect(str(exe), now=time.time(), recent_seconds=300)
    assert writable.writable_dir is True
    os.chmod(tmp_path, dir_mode)


def test_inspect_unknown_path_is_not_readable(tmp_path):
    facts = RealSystemFacts(tmp_path / "absent-mountinfo").inspect(
        str(tmp_path / "nope"), now=0.0, recent_seconds=1.0
    )
    assert facts.readable is False


def test_inspect_detects_tmpfs_from_mountinfo(tmp_path):
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("10 1 0:5 / / rw - tmpfs tmpfs rw\n")
    exe = tmp_path / "prog"
    exe.write_text("x")
    facts = RealSystemFacts(mountinfo).inspect(str(exe), now=0.0, recent_seconds=1e9)
    assert facts.mount_type == "tmpfs" and facts.on_tmpfs_or_noexec is True


def test_missing_mountinfo_degrades(tmp_path):
    facts = RealSystemFacts(tmp_path / "gone" / "mountinfo").inspect(
        str(tmp_path), now=0.0, recent_seconds=1.0
    )
    assert facts.mount_type == "unknown" and facts.on_tmpfs_or_noexec is False


def test_parse_mounts_ignores_bad_lines():
    assert parse_mounts("garbage\n1 2 3\n") == []
    assert parse_mounts("") == []

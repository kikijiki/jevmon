"""Allow rules from the TUI: rule building and escaping, the config editor and the daemon requests."""

import asyncio
import contextlib

import pytest

from jevmon import config as config_module
from jevmon.allow import AllowList, RuleError, build_rule, escape
from jevmon.config import ConfigError, edit_allow, load_config
from jevmon.core import SqliteStore
from jevmon.daemon import protocol
from jevmon.daemon.client import DaemonClient, DaemonError
from jevmon.daemon.server import DaemonServer
from jevmon.monitor import Monitor

pytestmark = pytest.mark.anyio

GIT = "/nix/store/abc-git-2.55.0/bin/git"
CLANG = "/nix/store/abc-clang-21/bin/clang"
GIT_ARGV = f"{GIT} -C /w/repo diff --name-only HEAD"
CLANG_ARGV = f"{CLANG} -c src/a.c -o build/a.o"


def test_rule_scopes_for_a_git_launch():
    assert build_rule(GIT, GIT_ARGV, "exact") == f"{GIT} -C /w/repo diff --name-only HEAD"
    assert build_rule(GIT, GIT_ARGV, "prefix", 1) == f"{GIT} -C *"
    assert build_rule(GIT, GIT_ARGV, "prefix", 3) == f"{GIT} -C /w/repo diff *"
    assert build_rule(GIT, GIT_ARGV, "exe") == f"{GIT} *"


def test_rule_scopes_for_a_clang_launch():
    assert build_rule(CLANG, CLANG_ARGV, "prefix") == f"{CLANG} -c *"
    assert build_rule(CLANG, CLANG_ARGV, "exe") == f"{CLANG} *"


def test_prefix_k_is_clamped_to_one_through_the_argument_count():
    assert build_rule(GIT, GIT_ARGV, "prefix", 0) == build_rule(GIT, GIT_ARGV, "prefix", 1)
    assert build_rule(GIT, GIT_ARGV, "prefix", 99) == f"{GIT} -C /w/repo diff --name-only HEAD *"
    with pytest.raises(RuleError):
        build_rule(GIT, GIT, "prefix")  # no arguments: the prefix is the exe-only rule


def test_every_built_rule_matches_its_launch_and_a_sibling_only_when_wider():
    args = "-C /w/repo diff --name-only HEAD"
    exact = AllowList.from_strings([build_rule(GIT, GIT_ARGV, "exact")])
    assert exact.match(GIT, args) == 0
    assert exact.match(GIT, args + " extra") is None
    assert exact.match("/usr/bin/git", args) is None  # strong form
    prefix = AllowList.from_strings([build_rule(GIT, GIT_ARGV, "prefix", 2)])
    assert prefix.match(GIT, "-C /w/repo status") == 0
    assert prefix.match(GIT, "-C /other status") is None


def test_exe_only_rule_matches_with_and_without_arguments():
    allow = AllowList.from_strings([build_rule(GIT, GIT_ARGV, "exe")])
    assert allow.match(GIT, "") == 0
    assert allow.match(GIT, "status") == 0
    assert allow.match(GIT, "push --force origin main") == 0
    assert allow.match("/usr/bin/git", "status") is None
    # a bare exe rule is the no-arguments rule, not "any arguments"
    bare = AllowList.from_strings([GIT])
    assert bare.match(GIT, "") == 0 and bare.match(GIT, "status") is None


def test_literal_stars_and_backslashes_are_escaped_and_round_trip():
    argv = r"/bin/find . -name *.o a\*b c\\d"
    exe = "/bin/find"
    for scope in ("exact", "prefix", "exe"):
        rule = build_rule(exe, argv, scope, 2)
        assert AllowList.from_strings([rule]).match(exe, r". -name *.o a\*b c\\d") == 0
    exact = AllowList.from_strings([build_rule(exe, argv, "exact")])
    assert exact.match(exe, r". -name x.o a\*b c\\d") is None  # the star is literal
    assert build_rule(exe, argv, "exact") == r"/bin/find . -name \*.o a\\\*b c\\\\d"
    assert escape("a*b\\") == "a\\*b\\\\"


def test_escape_syntax_in_hand_written_rules():
    allow = AllowList.from_strings([r"/bin/x a\*b *", r"/bin/y a\qb", r"/bin/z a\\*"])
    assert allow.match("/bin/x", "a*b rest") == 0
    assert allow.match("/bin/x", "axb rest") is None
    assert allow.match("/bin/y", r"a\qb") == 1  # a backslash before anything else is literal
    assert allow.match("/bin/z", "a\\anything") == 2  # \\ is a backslash, then a wildcard


def test_an_exe_with_a_star_is_matched_literally():
    rule = build_rule("/opt/we*ird/bin/t", "/opt/we*ird/bin/t x", "exe")
    allow = AllowList.from_strings([rule])
    assert allow.match("/opt/we*ird/bin/t", "x") == 0
    assert allow.match("/opt/weXird/bin/t", "x") is None


def test_unbuildable_launches_are_refused():
    with pytest.raises(RuleError, match="absolute"):
        build_rule("git", "git status", "exact")
    with pytest.raises(RuleError, match="whitespace"):
        build_rule("/opt/my tool/x", "/opt/my tool/x", "exe")
    with pytest.raises(RuleError, match="scope"):
        build_rule(GIT, GIT_ARGV, "nope")


# -- config editing ------------------------------------------------------------------------


def test_edit_creates_the_file_and_the_section(tmp_path):
    path = tmp_path / "sub" / "config.toml"
    rules, changed = edit_allow(path, add=f"{GIT} status *")
    assert changed and rules == [f"{GIT} status *"]
    assert load_config(path).filter.allow == (f"{GIT} status *",)


def test_edit_keeps_comments_and_other_sections(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "# my settings\n[judge]\nmode = 'fake'  # test judge\n\n"
        "[filter]\n# routine stuff\nallow = [\n  'git status *',  # keep\n]\nstore_skipped = true\n"
    )
    edit_allow(path, add="rocm-smi *")
    text = path.read_text()
    assert "# my settings" in text and "# test judge" in text
    assert "# routine stuff" in text and "# keep" in text
    config = load_config(path)
    assert config.filter.allow == ("git status *", "rocm-smi *")
    assert config.filter.store_skipped is True and config.judge.mode == "fake"
    _, changed = edit_allow(path, add="rocm-smi *")
    assert not changed
    rules, changed = edit_allow(path, remove="git status *")
    assert changed and rules == ["rocm-smi *"]
    assert "# routine stuff" in path.read_text()


def test_edit_adds_filter_section_after_existing_ones(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("# top\n[store]\nretention_days = 3\n")
    edit_allow(path, add="make *")
    text = path.read_text()
    assert text.startswith("# top\n[store]\nretention_days = 3\n")
    assert load_config(path).filter.allow == ("make *",)


def test_edit_copies_the_default_list_when_the_key_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "default_allow", lambda: ("a *", "b *"))
    path = tmp_path / "config.toml"
    path.write_text("[filter]\nstore_skipped = true\n")
    rules, _ = edit_allow(path, add="c *")
    assert rules == ["a *", "b *", "c *"]
    removed, changed = edit_allow(tmp_path / "other.toml", remove="zz *")
    assert not changed and removed == ["a *", "b *"]  # nothing written for a missing rule
    assert not (tmp_path / "other.toml").exists()
    after, changed = edit_allow(tmp_path / "other.toml", remove="b *", add="z *")
    assert changed and after == ["a *", "z *"]


def test_edit_rejects_bad_rules_and_bad_files_without_writing(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[filter]\nallow = ['git status']\n")
    before = path.read_text()
    with pytest.raises(RuleError):
        edit_allow(path, add="*")
    with pytest.raises(RuleError):
        edit_allow(path, add="   ")
    assert path.read_text() == before
    path.write_text("[filter\nbroken")
    with pytest.raises(ConfigError):
        edit_allow(path, add="make *")
    unwritable = tmp_path / "f"
    unwritable.write_text("")
    with pytest.raises(ConfigError):
        edit_allow(unwritable / "config.toml", add="make *")


# -- daemon --------------------------------------------------------------------------------


def test_protocol_version_and_requests():
    assert protocol.PROTOCOL_VERSION == 5
    req = protocol.parse_request({"v": 5, "type": "allow_add", "rule": " git status * "})
    assert req.params == {"rule": "git status *"}
    assert protocol.parse_request({"type": "allow_list"}).kind == "allow_list"
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"v": 4, "type": "status"})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"type": "allow_add"})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"type": "allow_remove", "rule": "  "})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request({"type": "allow_add", "rule": "x", "extra": 1})


async def test_add_remove_round_trip_updates_file_and_live_matcher(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("# mine\n[filter]\nallow = ['/usr/bin/ls *']\n")
    monitor = Monitor(load_config(config_path), None, config_path=str(config_path))
    monitor.allow.check("/usr/bin/ls", "-l")
    store = SqliteStore(tmp_path / "e.db")
    socket_path = tmp_path / "s.sock"
    server = DaemonServer(monitor, store, socket_path, poll_interval=0.05, stats_interval=100.0)
    task = asyncio.create_task(server.serve())
    for _ in range(300):
        if server.bound and socket_path.exists():
            break
        await asyncio.sleep(0.01)
    client = DaemonClient(socket_path)
    try:
        assert (await client.allow_list())["count"] == 1
        result = await client.allow_add(f"{GIT} status *")
        assert result["count"] == 2 and result["changed"] is True
        assert monitor.allow.match(GIT, "status -s") == 1
        assert monitor.allow.hits[0] == 1  # the unchanged rule kept its counter
        again = await client.allow_add(f"{GIT} status *")
        assert again["count"] == 2 and again["changed"] is False
        assert config_path.read_text().count(f"{GIT} status *") == 1
        assert "# mine" in config_path.read_text()
        assert (await client.status())["allow_rules"] == 2

        with pytest.raises(DaemonError, match="would match every launch"):
            await client.allow_add("*")
        assert (await client.allow_list())["count"] == 2

        removed = await client.allow_remove(f"{GIT} status *")
        assert removed["count"] == 1 and monitor.allow.match(GIT, "status") is None
        assert load_config(config_path).filter.allow == ("/usr/bin/ls *",)

        config_path.chmod(0o400)
        config_path.parent.chmod(0o500)
        try:
            if config_path.parent.stat().st_mode & 0o200:
                pytest.skip("directory stays writable (running as root)")
            with pytest.raises(DaemonError, match="cannot write"):
                await client.allow_add("make *")
        finally:
            config_path.parent.chmod(0o700)
        assert monitor.allow.match("/usr/bin/make", "") is None  # not applied when not saved
    finally:
        await client.close()
        server.request_shutdown()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(task, timeout=3.0)
        store.close()

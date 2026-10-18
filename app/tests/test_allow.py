"""Allow rules: matcher semantics, config loading and the monitor wiring."""

import time

import pytest

from jevmon.allow import AllowList, RuleError, args_of, compile_rules
from jevmon.config import ConfigError, load_config

GIT = "/nix/store/abc-git-2.5/bin/git"


def m(rules, exe, args=""):
    return AllowList.from_strings(rules).match(exe, args)


def test_weak_rule_matches_basename_anywhere():
    assert m(["git rev-parse *"], GIT, "rev-parse --show-toplevel") == 0
    assert m(["git rev-parse *"], "/usr/bin/git", "rev-parse HEAD") == 0
    assert m(["git rev-parse *"], "/usr/bin/gitk", "rev-parse HEAD") is None


def test_strong_rule_matches_full_path():
    rule = ["/nix/store/*-git-*/bin/git config *"]
    assert m(rule, GIT, "config --get user.name") == 0
    assert m(rule, "/usr/bin/git", "config --get user.name") is None
    assert m(["*/bin/git"], GIT) == 0
    assert m(["/usr/bin/git status"], "/usr/bin/git", "status") == 0


def test_spoofed_binary_matches_bare_name_but_not_absolute_rule():
    assert m(["git status"], "/tmp/git", "status") == 0
    assert m(["/usr/bin/git status"], "/tmp/git", "status") is None
    assert m(["/nix/store/*-git-*/bin/git status"], "/tmp/git", "status") is None


def test_args_boundaries():
    rule = ["git diff *"]
    assert m(rule, GIT, "diff") == 0
    assert m(rule, GIT, "diff HEAD~1 -- a b") == 0
    assert m(rule, GIT, "diffx") is None
    assert m(rule, GIT, "log diff") is None
    assert m(["git diff"], GIT, "diff") == 0
    assert m(["git diff"], GIT, "diff HEAD") is None


def test_exe_only_rule_means_no_arguments():
    assert m(["ls"], "/bin/ls") == 0
    assert m(["ls"], "/bin/ls", "-la") is None
    assert m(["rocm-smi *"], "/opt/rocm/bin/rocm-smi") == 0
    assert m(["rocm-smi *"], "/opt/rocm/bin/rocm-smi", "--showtemp") == 0


def test_colon_star_alias():
    assert m(["git diff:*"], GIT, "diff HEAD") == 0
    assert m(["git diff:*"], GIT, "diffx") is None
    assert m(["git diff:*"], GIT, "diff") == 0


def test_star_crosses_spaces_and_everything_else_is_literal():
    rule = ["git config *"]
    assert m(rule, GIT, "config --get a b c") == 0
    assert m(["tool a?b"], "/bin/tool", "a?b") == 0
    assert m(["tool a?b"], "/bin/tool", "axb") is None
    assert m(["tool [ab]"], "/bin/tool", "a") is None
    assert m(["tool [ab]"], "/bin/tool", "[ab]") == 0
    assert m(["tool .+"], "/bin/tool", "xx") is None
    assert m(["tool *"], "/bin/tool", "a$(rm -rf /)\nb;`x`") == 0


def test_first_matching_rule_wins():
    rules = ["git *", "git status", "/x/git"]
    assert m(rules, GIT, "status") == 0
    assert m(["/nix/store/*/git", "git *"], "/nix/store/a/git", "") == 0


@pytest.mark.parametrize("bad", ["", "   ", "*", "\t"])
def test_invalid_rules_are_rejected(bad):
    with pytest.raises(RuleError):
        compile_rules([bad])


def test_empty_allow_list_matches_nothing():
    assert m([], GIT, "status") is None


def test_args_of():
    assert args_of("/bin/git status -s") == "status -s"
    assert args_of("/bin/git") == ""


def test_counters_and_hit_counts():
    allow = AllowList.from_strings(["git status", "rocm-smi *"])
    assert allow.check(GIT, "status") == 0 and allow.check("/x/rocm-smi", "-a") == 1
    assert allow.check("/x/rocm-smi", "-b") == 1 and allow.check(GIT, "log") is None
    assert allow.skipped == 3
    assert allow.hit_counts() == {"rocm-smi *": 2, "git status": 1}


def test_matching_many_rules_is_cheap():
    rules = [f"tool{i} sub{i} *" for i in range(10_000)]
    rules += ["/nix/store/*-git-*/bin/git config *", "git rev-parse *"]
    allow = AllowList.from_strings(rules)
    start = time.perf_counter()
    for _ in range(10_000):
        assert allow.match(GIT, "rev-parse HEAD") == 10_001
        assert allow.match("/usr/bin/other", "x y z") is None
    per_launch = (time.perf_counter() - start) / 20_000
    assert per_launch < 0.0002, per_launch  # poller budget is ms per scan; this is far below


def test_config_loads_rules_and_reports_bad_ones(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[filter]\nallow = ["git rev-parse *"]\nstore_skipped = true\n')
    cfg = load_config(path)
    assert cfg.filter.allow == ("git rev-parse *",) and cfg.filter.store_skipped is True
    assert load_config().filter.allow == () and load_config().filter.store_skipped is False
    path.write_text('[filter]\nallow = ["ok *", "  "]\n')
    with pytest.raises(ConfigError, match="empty allow rule"):
        load_config(path)
    path.write_text("[filter]\nallow = [1]\n")
    with pytest.raises(ConfigError, match="list of strings"):
        load_config(path)
    path.write_text("[filter]\nallowed = []\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_shipped_default_toml_has_an_empty_allow_list(tmp_path):
    empty = tmp_path / "empty.toml"
    empty.write_text("")
    assert load_config(empty).filter.allow == ()  # the shipped default merged with nothing


def test_a_workstation_rule_set_skips_routine_launches_and_not_risky_ones():
    git = "/nix/store/*-git-*/bin/git"
    rules = [
        f"{git} rev-parse *",
        f"{git} -C * diff *",
        f"{git} branch --show-current",
        f"{git} config --get *",
        "/nix/store/*-kache-*/bin/kache *",
        "/nix/store/*/bin/clang -c *",
        "rocm-smi *",
    ]
    allow = AllowList.from_strings(rules)
    exe = "/nix/store/abc-git-2.55.0/bin/git"
    for args in (
        "rev-parse HEAD",
        "-C /w/repo -c core.quotepath=false diff --name-only",
        "branch --show-current",
        "config --get user.name",
    ):
        assert allow.match(exe, args) is not None, args
    for args in ("push origin main", "clone https://example.invalid/x.git", "config a b"):
        assert allow.match(exe, args) is None, args
    assert allow.match("/tmp/git", "rev-parse HEAD") is None
    assert allow.match("/nix/store/abc-kache-1.1.1/bin/kache", "rustc -vV") is not None
    assert allow.match("/nix/store/abc/lib/llvm/bin/clang", "-c a.c") is not None
    assert allow.match("/opt/venv/bin/rocm-smi", "--showtemp") is not None

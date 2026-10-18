"""External-dataset ingestion: shell splitting, mapping tables, keys, rendering."""

import pytest

from jevmon.detect.context import ContextSettings
from jevmon_lab import ingest

DRIVER = {"exe": "/usr/bin/dash", "argv0": "sh"}


def progs(cmd: str) -> list[list[str]]:
    return ingest.parse_command_line(cmd).programs


def test_pipeline_and_builtins() -> None:
    assert progs("cd /tmp && ls -l | grep x; export A=1") == [["ls", "-l"], ["grep", "x"]]


def test_quotes_heredoc_and_substitution() -> None:
    assert progs("echo 'a | b' ; wc -l") == [["wc", "-l"]]
    assert progs("cat > f <<'EOF'\nrm -rf /\nEOF\nsort f") == [["cat"], ["sort", "f"]]
    assert progs("ls $(whoami)") == [["whoami"], ["ls", "$(whoami)"]]


def test_assignment_redirect_and_loop() -> None:
    assert progs("FOO=1 python3 -c 'print(1)' > /dev/null 2>&1") == [["python3", "-c", "print(1)"]]
    assert progs('for f in *.txt; do rm "$f"; done') == [["rm", "$f"]]


def test_unbalanced_quote_is_not_ok() -> None:
    assert not ingest.parse_command_line("echo 'oops").ok


def test_simple_command_gets_a_direct_row() -> None:
    kw = {
        "source": "s",
        "role": "test",
        "label": "safe",
        "rule": "r",
        "uncertain": "",
        "family": "f",
        "source_label": {},
        "driver": DRIVER,
    }
    kinds = [r.kind for r in ingest.command_rows("ls -a ./build", **kw)]
    assert kinds == ["driver", "child", "direct"]
    kinds = [r.kind for r in ingest.command_rows("ls | wc -l", **kw)]
    assert kinds == ["driver", "child", "child"]
    rows = ingest.command_rows("ls | wc -l", **kw)
    assert rows[0].label == "safe" and all(r.label is None for r in rows[1:])
    assert rows[0].launch.argv == ["sh", "-c", "ls | wc -l"]


RULES = [
    {"id": "a", "when": {"x": ["1"]}, "to": "safe"},
    {"id": "b", "when": {"cats_any": ["p", "q"], "lvl": ["hi"]}, "to": "crit", "uncertain": "why"},
    {"id": "c", "when": {"cats_none": ["z"]}, "to": "drop"},
]


def test_apply_rules_first_match_and_loud_failure() -> None:
    assert ingest.apply_rules(RULES, {"x": "1"}) == ("safe", "a", "")
    assert ingest.apply_rules(RULES, {"x": "2", "cats": ["q"], "lvl": "hi"}) == ("crit", "b", "why")
    assert ingest.apply_rules(RULES, {"x": "2", "cats": ["w"]})[0] == "drop"
    with pytest.raises(ValueError):
        ingest.apply_rules(RULES, {"x": "2", "cats": ["z"]})


@pytest.mark.parametrize("name", ["shell_safety", "cli_risk", "quasarnix", "splunk"])
def test_mapping_tables_are_valid(name: str) -> None:
    table = ingest.load_toml(name)
    for rule in table.get("rule", []):
        assert rule["to"] in ingest.TARGETS


def test_cli_risk_mapping_cases() -> None:
    rules = ingest.load_toml("cli_risk")["rule"]

    def to(level, cats):
        return ingest.apply_rules(rules, {"risk_level": level, "risk_categories": cats})[0]

    assert to("safe", ["safe_baseline"]) == "safe"
    assert to("high", ["unsafe_download_exec"]) == "crit"
    assert to("critical", ["data_destruction"]) == "drop"
    assert to("high", ["credential_exposure"]) == "warn"
    assert to("moderate", ["irreversible_config_x"]) == "null"


def test_template_key_masks_hosts_numbers_paths() -> None:
    a = ingest.template_key(["curl", "http://1.2.3.4:80/x", "-m", "5"])
    b = ingest.template_key(["curl", "http://9.8.7.6:81/y", "-m", "7"])
    assert a == b


def test_renderer_uses_shipped_format() -> None:
    text = ingest.Renderer().render(ingest.Launch(exe="/usr/bin/dash", argv=["sh", "-c", "id"]))
    assert text == (
        "process launch (tier 0 /proc poller) exe: /usr/bin/dash  basename: dash "
        "argv(template): sh -c id argv(redacted): sh -c id"
    )
    assert ContextSettings().order == ("header", "exe", "template", "argv")


def test_renderer_redacts_secrets() -> None:
    text = ingest.Renderer().render(
        ingest.Launch(exe="curl", argv=["curl", "--password=hunter2", "http://h"])
    )
    assert "hunter2" not in text


def test_dedupe_drops_lab_and_conflicts() -> None:
    def row(key, label):
        r = ingest.Row("s", "test", "driver", ingest.Launch("x", [key]), label)
        r.key = key
        return r

    rows, st = ingest.dedupe(
        [row("a", "safe"), row("a", "safe"), row("b", "safe"), row("b", "crit"), row("c", "safe")],
        forbidden={"c"},
    )
    assert [r.key for r in rows] == ["a"] and rows[0].count == 2
    assert st["label_conflict"] == 1 and st["in_lab_or_other_split"] == 1

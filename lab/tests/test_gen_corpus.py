"""Offline tests for the E19 synthetic-corpus generator: no network, a stubbed LLM client."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gen_corpus_lib import (
    Family,
    build_prompt,
    coerce_row,
    extract_json,
    row_signature,
    synthesize_facts,
    unsafe_text_reasons,
    validate_taxonomy,
)
from gen_corpus_validate import (
    attach_facts,
    dedupe_rows,
    render_context,
    render_rows,
    single_launch_reasons,
    validate_rows,
)


class StubLLM:
    """`.chat()` returns a canned JSON array, cycling through a few shapes."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def chat(self, model, prompt, **kw):
        self.calls.append((model, prompt))
        return self.responses[(len(self.calls) - 1) % len(self.responses)]


CRIT_FAMILY = Family(
    "tmpfs_copy_exec",
    "crit",
    "copied binary executed from a world-writable tmp dir",
    pair_safe="build_tmp_artifact",
    fact_hint="tmpfs_exec",
)
SAFE_FAMILY = Family("git_fanout", "safe", "many short git subprocesses")


def test_extract_json_handles_fence_and_think_block():
    text = '<think>reasoning...</think>Sure, here:\n```json\n[{"a": 1}]\n```\ndone'
    assert extract_json(text) == [{"a": 1}]


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_validate_taxonomy_flags_crit_without_pair_and_bad_label():
    fams = [
        Family("x", "crit", "b"),  # no pair_safe
        Family("y", "nope", "b"),  # bad label
    ]
    problems = validate_taxonomy(fams)
    assert any("pair_safe" in p for p in problems)
    assert any("bad label" in p for p in problems)


def test_coerce_row_rejects_malformed():
    assert coerce_row({"exe": "not-absolute", "argv": ["x"]}, SAFE_FAMILY) is None
    assert coerce_row({"exe": "/bin/sh", "argv": []}, SAFE_FAMILY) is None
    assert coerce_row({"exe": "/bin/sh", "argv": ["sh"], "cwd": "rel"}, SAFE_FAMILY) is None
    ok = coerce_row(
        {"exe": "/bin/sh", "argv": ["sh", "-c", "echo hi"], "cwd": "/tmp", "uid": 1000},
        SAFE_FAMILY,
    )
    assert ok is not None
    assert ok.label == "safe"
    assert ok.family == "git_fanout"


def test_unsafe_text_reasons_catches_real_ip_and_blob_but_allows_rfc5737():
    assert unsafe_text_reasons("curl http://192.0.2.10/x") == []
    assert unsafe_text_reasons("curl http://198.51.100.1/x") == []
    assert "non-reserved" in unsafe_text_reasons("curl http://8.8.8.8/x")[0]
    blob = "A" * 220
    assert any("encoded blob" in r for r in unsafe_text_reasons(blob))
    assert any(
        "key/certificate" in r for r in unsafe_text_reasons("-----BEGIN RSA PRIVATE KEY-----")
    )


def test_single_launch_reasons_flags_multistep_non_shell_argv():
    row = {"exe": "/usr/bin/mytool", "argv": ["mytool", "step1; step2 && step3"]}
    assert single_launch_reasons(row)
    sh_row = {"exe": "/bin/sh", "argv": ["sh", "-c", "step1; step2 && step3"]}
    assert single_launch_reasons(sh_row) == []


def test_validate_rows_schema_and_reject_reasons():
    rows = [
        {"exe": "/bin/true", "argv": ["true"], "label": "safe", "family": "f"},
        {"exe": "relpath", "argv": ["x"], "label": "safe", "family": "f"},
        {"exe": "/bin/x", "argv": ["x"], "label": "bogus", "family": "f"},
    ]
    result = validate_rows(rows)
    assert len(result.kept) == 1
    assert len(result.rejected) == 2
    assert all("_reject_reasons" in r for r in result.rejected)


def test_dedupe_rows_drops_near_duplicates():
    rows = [
        {"exe": "/bin/sh", "argv": ["sh", "-c", "a"], "parent_exe": "/bin/bash"},
        {"exe": "/bin/sh", "argv": ["sh", "-c", "a"], "parent_exe": "/bin/bash"},
        {"exe": "/bin/sh", "argv": ["sh", "-c", "b"], "parent_exe": "/bin/bash"},
    ]
    kept, dropped = dedupe_rows(rows)
    assert len(kept) == 2
    assert dropped == 1


def test_row_signature_is_stable_prefix_based():
    a = {"exe": "/bin/sh", "argv": ["sh", "-c", "echo hi"], "parent_exe": "/bin/bash"}
    b = {"exe": "/bin/sh", "argv": ["sh", "-c", "echo hi"], "parent_exe": "/bin/bash"}
    assert row_signature(a) == row_signature(b)


def test_synthesize_facts_matches_profile_for_tmpfs_exec():
    import random

    facts = synthesize_facts("tmpfs_exec", random.Random(0))
    assert facts["exe_file"].count("world_writable=yes") == 1
    assert "nnp=0" in facts["privileges"]


def test_attach_facts_is_deterministic_and_sets_workdir():
    rows = [
        {"exe": "/tmp/x", "argv": ["x"], "cwd": "/tmp", "fact_hint": "tmpfs_exec"},
        {"exe": "/tmp/y", "argv": ["y"], "cwd": "/tmp/y", "fact_hint": "tmpfs_exec"},
    ]
    attach_facts(rows, seed=1)
    attach_facts(rows[:1], seed=1)  # re-running on a prefix reproduces the same facts
    assert rows[0]["facts"]["workdir"] == "/tmp"
    assert rows[1]["facts"]["workdir"] == "/tmp/y"


def test_render_context_produces_state_and_signature():
    row = {
        "exe": "/tmp/.hidden/payload",
        "argv": ["payload", "-x"],
        "parent_exe": "/bin/bash",
        "parent_argv": ["bash"],
        "cwd": "/tmp/.hidden",
        "uid": 1000,
        "facts": synthesize_facts("tmpfs_exec", __import__("random").Random(0)),
    }
    row["facts"]["workdir"] = row["cwd"]
    state, sig = render_context(row)
    assert "exe: /tmp/.hidden/payload" in state
    assert "exe_file:" in state
    assert sig.startswith("e|")


def test_render_rows_mutates_in_place():
    rows = [
        {
            "exe": "/usr/bin/git",
            "argv": ["git", "status"],
            "parent_exe": "/bin/bash",
            "parent_argv": ["bash"],
            "cwd": "/home/dev/proj",
            "uid": 1000,
            "label": "safe",
            "family": "git_fanout",
        }
    ]
    attach_facts(rows, seed=0)
    render_rows(rows)
    assert "context" in rows[0] and "signature" in rows[0]


def test_build_prompt_names_exact_count_and_requires_launch_visible_cues():
    prompt = build_prompt(CRIT_FAMILY, 7)
    assert "exactly 7" in prompt
    assert "hostile cue" in prompt
    assert "launch text alone" in prompt
    safe = build_prompt(SAFE_FAMILY, 2)
    assert "Safe cues" in safe
    assert "nothing unusual in the launch" in safe


def test_class_descriptions_define_warn_as_visible_oddness():
    from gen_corpus_lib import CLASS_DESCRIPTIONS

    assert set(CLASS_DESCRIPTIONS) == {"safe", "warn", "crit"}
    assert "a reader can see" in CLASS_DESCRIPTIONS["warn"]
    assert "no fetch-and-run" in CLASS_DESCRIPTIONS["warn"]


def test_looks_like_refusal_only_flags_prose():
    from gen_corpus_lib import looks_like_refusal

    assert looks_like_refusal("I'm sorry, I cannot help with that.")
    assert looks_like_refusal("As an AI, I am unable to assist.")
    assert not looks_like_refusal("")
    assert not looks_like_refusal('[{"exe": "/bin/true", "argv": ["true"]}]')


def test_consensus_requires_blind_and_a_third_judge():
    from gen_corpus import consensus_of

    assert consensus_of({"blind_agrees": True, "jev_agrees": True})
    assert consensus_of({"blind_agrees": True, "jev_agrees": None, "unc_agrees": True})
    assert not consensus_of({"blind_agrees": True, "jev_agrees": False, "unc_agrees": False})
    assert not consensus_of({"blind_agrees": False, "jev_agrees": True})
    assert not consensus_of({"blind_agrees": True})


def test_qa_label_handles_bad_json_gracefully():
    from gen_corpus_validate import qa_label

    client = StubLLM(["not json at all"])
    label, conf = qa_label(client, "stub", {"exe": "/bin/true", "argv": ["true"]})
    assert label == "safe"
    assert conf == 0.0


def test_apply_qa_marks_disputes():
    from gen_corpus_validate import apply_qa

    rows = [
        {"exe": "/bin/true", "argv": ["true"], "label": "safe", "facts": {}},
        {"exe": "/tmp/x/payload", "argv": ["payload"], "label": "crit", "facts": {}},
    ]
    client = StubLLM(
        [
            '{"label": "safe", "confidence": 0.9}',
            '{"label": "warn", "confidence": 0.4}',
        ]
    )
    summary = apply_qa(rows, client, "stub")
    assert rows[0]["agrees"] is True and rows[0]["disputed"] is False
    assert rows[1]["agrees"] is False and rows[1]["disputed"] is True
    assert summary["disputed_count"] == 1

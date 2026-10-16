"""Offline tests for the hosted-Jev probe and its scoring (lab/scripts/jev_probe.py, jev_score.py)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _load("jev_probe")
score = _load("jev_score")

CTX = (
    "process launch (tier 0 /proc poller) exe: /usr/bin/curl  basename: curl "
    "argv(template): curl -fsSL <PATH> -o <PATH> argv(redacted): curl -fsSL http://x/y.sh -o /tmp/y"
)
ROW = {"context": CTX, "context_c": "process launch exe: /usr/bin/curl argv: curl http://x/y.sh"}


def test_flat_request_is_the_context_verbatim():
    body = probe.build_request(ROW, "A_flat_choice")
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"] == CTX
    q = body["questions"]["verdict"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"safe", "warn", "crit"}


def test_lines_layout_puts_each_segment_on_a_line():
    state = probe.build_request(ROW, "B_lines_choice")["state"]
    lines = state.split("\n")
    assert lines[0] == "process launch (tier 0 /proc poller)"
    assert [ln.split(":")[0] for ln in lines[1:]] == [
        "exe",
        "basename",
        "argv(template)",
        "argv(redacted)",
    ]


def test_json_layout_keys_and_values():
    state = probe.build_request(ROW, "E_json_choice")["state"]
    assert list(state) == ["header", "exe", "basename", "argv_template", "argv"]
    assert state["exe"] == "/usr/bin/curl"
    assert state["argv"] == "curl -fsSL http://x/y.sh -o /tmp/y"
    assert state["argv_template"] == "curl -fsSL <PATH> -o <PATH>"


def test_segments_keep_a_label_lookalike_inside_argv():
    ctx = "process launch exe: /bin/sh  basename: sh argv(redacted): sh -c echo exe: x"
    seg = probe.segments(ctx)
    assert seg["argv"] == "sh -c echo exe: x"
    assert seg["exe"] == "/bin/sh"


def test_raw_variant_reads_the_other_field_and_binary_has_three_questions():
    assert probe.build_request(ROW, "D_raw_flat_choice")["state"] == ROW["context_c"]
    q = probe.build_request(ROW, "C_flat_binary")["questions"]
    assert set(q) == {"safe_to_run", "needs_review", "hostile"}
    assert all(v["type"] == "noul" for v in q.values())


def test_request_key_depends_on_model_and_variant_not_dict_order():
    a = probe.build_request(ROW, "A_flat_choice")
    assert probe.request_key(a) == probe.request_key(dict(reversed(list(a.items()))))
    assert probe.request_key(a) != probe.request_key(
        probe.build_request(ROW, "A_flat_choice", model="other")
    )
    assert probe.request_key(a) != probe.request_key(probe.build_request(ROW, "B_lines_choice"))


def test_key_comes_from_env_or_file_never_default(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        probe.load_key(None)
    f = tmp_path / "k"
    f.write_text("secret\n")
    assert probe.load_key(str(f)) == "secret"
    monkeypatch.setenv("OPENROUTER_API_KEY", "fromenv")
    assert probe.load_key(str(f)) == "fromenv"


def test_cached_call_posts_once_and_does_not_cache_errors(tmp_path, monkeypatch):
    calls = []

    def fake_post(body, key, **_):
        calls.append(body)
        return {"response": {"answers": {}}, "latency_s": 0.1}

    monkeypatch.setattr(probe, "post", fake_post)
    body = probe.build_request(ROW, "A_flat_choice")
    first = probe.cached_call(body, "k", tmp_path)
    assert probe.cached_call(body, "k", tmp_path) == first
    assert len(calls) == 1

    monkeypatch.setattr(probe, "post", lambda *a, **k: {"error": "http 500"})
    other = probe.build_request(ROW, "B_lines_choice")
    assert "error" in probe.cached_call(other, "k", tmp_path)
    assert not list(tmp_path.glob(f"{probe.request_key(other)}*"))


def test_run_all_keys_rows_and_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "post", lambda *a, **k: {"response": {"answers": {}}})
    res = probe.run_all([ROW, ROW], ["A_flat_choice", "B_lines_choice"], "k", cache_dir=tmp_path)
    assert sorted(res) == [
        "0|A_flat_choice",
        "0|B_lines_choice",
        "1|A_flat_choice",
        "1|B_lines_choice",
    ]


def test_macro_f1_and_per_class():
    y = np.asarray([0, 0, 1, 1, 2, 2])
    assert score.macro_f1(y, y) == 1.0
    pred = np.asarray([0, 1, 1, 1, 2, 0])
    pc = score.per_class(y, pred)
    assert pc[0] == (0.5, 0.5) and pc[1] == (2 / 3, 1.0) and pc[2] == (1.0, 0.5)
    # class 1: tp 2 fp 1 fn 0 -> 0.8; class 0: tp1 fp1 fn1 -> 0.5; class 2: tp1 fp0 fn1 -> 2/3
    assert score.macro_f1(y, pred) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_bootstrap_is_deterministic_paired_and_respects_families():
    y = np.tile([0, 1, 2], 8)
    fam = np.repeat([f"f{i}" for i in range(8)], 3)
    good, bad = y.copy(), (y + 1) % 3
    draws = score.boot_draws(fam, n=200)
    assert all(len(d) % 3 == 0 for d in draws)
    r = score.boot_paired(y, good, bad, draws)
    assert r["lo"] == r["hi"] == 1.0
    assert r["diff_lo"] > 0 and r["p_better"] == 1.0
    again = score.boot_paired(y, good, bad, score.boot_draws(fam, n=200))
    assert again == r
    same = score.boot_paired(y, good, good, draws)
    assert same["diff_lo"] == same["diff_hi"] == 0.0 and same["p_better"] == 0.0


def test_set_index_new_and_pool_exclude_train():
    sets = np.asarray(["train", "old", "re", "fresh"])
    assert score.set_index(sets, "new").tolist() == [2, 3]
    assert score.set_index(sets, "pool").tolist() == [1, 2, 3]


def _out(vals):
    res = {}
    for i, p in enumerate(vals):
        a = {"verdict": {"probabilities": dict(zip(score.CLASSES, p)), "confidence": max(p)}}
        res[f"{i}|A_flat_choice"] = {"response": {"answers": a}}
    return {"rows": [{}] * len(vals), "res": res}


def test_feature_matrix_and_missing_response():
    out = _out([(0.6, 0.3, 0.1), (0.1, 0.2, 0.7)])
    x = score.feature_matrix(out, "A_flat_choice")
    assert x.argmax(1).tolist() == [0, 2]
    assert score.confidences(out, "A_flat_choice").tolist() == [0.6, 0.7]
    out["res"]["1|A_flat_choice"] = {"error": "x"}
    with pytest.raises(ValueError):
        score.feature_matrix(out, "A_flat_choice")


def test_calibration_bins_and_ece():
    conf = np.asarray([0.9] * 5 + [0.1] * 5)
    correct = np.asarray([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], float)
    c = score.calibration(conf, correct, bins=2)
    assert [b["n"] for b in c["bins"]] == [5, 5]
    assert c["ece"] == pytest.approx(0.1)
    miscal = score.calibration(np.full(4, 0.9), np.asarray([1, 0, 0, 0], float), bins=1)
    assert miscal["ece"] == pytest.approx(0.65)


def test_logreg_head_separates_and_predicts():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 20)
    x = np.eye(3)[y] * 0.8 + rng.normal(0, 0.05, (60, 3))
    params = score.fit_logreg(x, y)
    assert (score.predict_logreg(params, x) == y).mean() > 0.95

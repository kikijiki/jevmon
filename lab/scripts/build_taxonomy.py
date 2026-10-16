"""E19 (1): build `lab/corpus_taxonomy.toml` — hand-authored triples + LLM-proposed families.

    uv run --project lab python scripts/build_taxonomy.py --out ../lab/corpus_taxonomy.toml \
        --propose 40 --model qwen3.8

Each hand-authored entry is a (crit, pair_safe, pair_warn) triple sharing one surface shape, so
a classifier cannot key on the exe name or a keyword alone - only the launch-level context
(location, parent, privilege, freshness) tells the classes apart. The LLM then proposes further
families per class (`propose_families_prompt`); duplicates/near-synonyms of existing names are
dropped by a simple token-overlap check, and anything the LLM proposes without an explicit label
field is rejected. LLM-proposed crit families without a hand-picked pairing get the nearest
existing pair_safe/pair_warn by keyword overlap, falling back to a generic pair if nothing matches
well (flagged `source = "llm"` either way, for later curation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen_corpus_lib import (
    Family,
    LLMClient,
    extract_json,
    propose_families_prompt,
    validate_taxonomy,
)

# (crit, pair_safe, pair_warn) triples. Each triple shares a surface (same tool family, similar
# argv keywords) so the hard negative really is hard: the giveaway is launch context, not a
# keyword. fact_hint selects the structural-facts distribution (gen_corpus_lib.FACT_PROFILES).
TRIPLES: list[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]] = [
    (
        ("tmpfs_copy_exec", "copied system binary executed from a world-writable tmp dir"),
        (
            "build_tmp_artifact",
            "a build tool writes and runs its own output under /tmp, normal CI pattern",
        ),
        ("user_script_tmp", "a user's one-off script saved and run from /tmp, no obfuscation"),
    ),
    (
        ("curl_pipe_sh", "download a remote script and pipe it straight into a shell"),
        ("curl_pipe_tee", "curl output piped to tee for inspection before anything runs"),
        ("install_sh_known_vendor", "a package's own documented curl|sh installer, run once"),
    ),
    (
        ("inline_b64_eval", "interpreter given an inline base64/zlib-decoded command to eval"),
        ("inline_json_config", "interpreter given a long inline JSON/config string argument"),
        (
            "inline_oneliner_debug",
            "a short inline python/node one-liner for debugging, not encoded",
        ),
    ),
    (
        ("hidden_cron_persist", "a cron/systemd/rc entry pointing at a dotfile-hidden script"),
        ("cron_timer_normal", "a scheduled job running a normal, visibly-named maintenance script"),
        (
            "cron_new_job_first_run",
            "a freshly added cron job's first execution, unusual name but visible",
        ),
    ),
    (
        ("ld_preload_inject", "a normal binary launched with LD_PRELOAD pointing at a dropped .so"),
        (
            "ld_library_path_dev",
            "a developer sets LD_LIBRARY_PATH to a local build dir, normal workflow",
        ),
        ("ld_preload_debug_tool", "LD_PRELOAD set to a known debugging/profiling shim, first use"),
    ),
    (
        (
            "path_tamper_shadow_bin",
            "PATH prepended with a writable dir shadowing a system binary name",
        ),
        ("path_prepend_venv", "PATH prepended with a project venv/bin dir, ordinary tooling"),
        ("path_prepend_local_bin", "PATH prepended with ~/.local/bin for a newly installed CLI"),
    ),
    (
        (
            "cred_harvest_history",
            "process greps/reads shell history and ssh/cloud credential files",
        ),
        ("grep_history_self_lookup", "the user greps their own shell history for a past command"),
        (
            "backup_dotfiles_script",
            "a one-off script backing up dotfiles including credential files",
        ),
    ),
    (
        ("reverse_shell_oneliner", "a shell/interpreter one-liner opening a reverse network shell"),
        (
            "netcat_normal_listen",
            "netcat used to listen locally for a legitimate quick file transfer",
        ),
        ("ssh_reverse_tunnel", "an ssh reverse tunnel started for remote support, first use"),
    ),
    (
        ("masquerade_kernel_name", "a dropped binary renamed to look like a kernel/system thread"),
        ("kernel_thread_normal", "a genuine kernel worker thread's normal launch text"),
        ("renamed_vendor_tool", "a vendor tool shipped under a generic-sounding renamed binary"),
    ),
    (
        (
            "container_escape_mount",
            "process inside a container execs via a mounted docker/podman socket",
        ),
        (
            "container_build_step",
            "a normal containerized build step execing inside the build container",
        ),
        (
            "container_debug_shell",
            "a developer execs a debug shell into a running container, first time",
        ),
    ),
    (
        (
            "unexpected_parent_office",
            "a shell spawned by a document viewer or office-suite process",
        ),
        ("shell_from_terminal", "an ordinary interactive shell spawned by a terminal emulator"),
        ("shell_from_ide", "a shell spawned by an IDE's integrated terminal, first session today"),
    ),
    (
        ("setuid_priv_escalate", "an unusual setuid binary executed to gain root unexpectedly"),
        ("sudo_normal_admin", "a normal sudo-prefixed admin command, routine privilege use"),
        ("setuid_known_utility", "a standard setuid utility (ping/mount) used in its ordinary way"),
    ),
    (
        ("memfd_exec_fileless", "a process executed from an anonymous memfd, no file on disk"),
        (
            "dynamic_loader_normal",
            "the dynamic loader mapping a normal shared library, benign memfd use",
        ),
        (
            "jit_compiled_tool",
            "a JIT/interpreter tool using memfd for its own compiled code, first run",
        ),
    ),
    (
        ("ssh_key_exfil_scp", "scp/rsync pushing ssh private keys or cloud credentials off-host"),
        ("scp_normal_deploy", "scp pushing a build artifact to a deploy target, routine release"),
        (
            "rsync_backup_homedir",
            "rsync backing up the whole home directory including keys, scheduled",
        ),
    ),
    (
        ("disable_security_tool", "a command disabling or killing the host's security/EDR agent"),
        ("service_restart_normal", "a normal systemctl restart of an unrelated service"),
        ("security_agent_update", "the security agent restarting itself for a routine update"),
    ),
    (
        ("exfil_dns_tunnel", "data encoded into DNS queries via dig/nslookup to attacker domains"),
        ("dns_lookup_normal", "an ordinary dig/nslookup for a routine hostname"),
        (
            "dns_debug_many_queries",
            "a network engineer running many dig queries while debugging DNS",
        ),
    ),
    (
        ("fork_decoy_name", "a short-lived process launched under a decoy name like [kworker]"),
        ("real_kworker_thread", "an actual kernel worker thread's launch text"),
        (
            "process_renamed_by_app",
            "a long-running daemon that sets its own argv[0] to a friendly name",
        ),
    ),
    (
        (
            "compiler_from_doc_macro",
            "a compiler/interpreter spawned by an office macro or PDF reader",
        ),
        ("compiler_from_make", "a compiler spawned by make/ninja during an ordinary build"),
        (
            "compiler_from_ide_build",
            "a compiler spawned by an IDE's build button, first project build",
        ),
    ),
    (
        (
            "env_var_token_exfil",
            "process reads CI/cloud secret env vars then execs a network client",
        ),
        (
            "env_dump_debug",
            "a developer dumps env vars to a local file while debugging, no network",
        ),
        (
            "cloud_cli_with_token",
            "a cloud CLI reading its own configured token to make an API call",
        ),
    ),
    (
        ("binary_patch_live", "a running binary's on-disk file is replaced and re-executed"),
        (
            "package_manager_upgrade",
            "a package manager replacing then re-launching an upgraded binary",
        ),
        ("hot_reload_dev_server", "a dev server's hot-reload re-execing itself after a rebuild"),
    ),
]

# Non-paired safe/warn variety families (plain in-distribution data: dev tooling, swarms,
# admin one-offs) that are not hard negatives for any single crit family.
EXTRA_SAFE = [
    ("git_fanout", "many short-lived git subprocesses during a status/diff/log-heavy session"),
    ("uv_python_build", "uv/pip/cargo resolving and building many small subprocesses"),
    ("node_test_swarm", "a test runner spawning many short-lived node worker processes"),
    ("agent_swarm_launcher", "a launcher starting many short-lived python agent workers"),
    ("ssh_admin_session", "an admin's interactive ssh session running routine commands"),
    ("cloud_cli_deploy", "a cloud CLI (aws/gcloud/az) deploying infrastructure, routine use"),
    ("data_science_notebook", "a jupyter kernel spawning pandas/numpy-heavy subprocesses"),
    ("package_install_apt", "apt/dnf/pacman installing a batch of packages sequentially"),
    ("container_compose_up", "docker/podman compose starting several service containers"),
    ("rust_cargo_fanout", "cargo build spawning rustc and many codegen-unit subprocesses"),
]
EXTRA_WARN = [
    ("first_run_new_binary", "a binary installed minutes ago run for the very first time"),
    ("debugger_attach", "a debugger or profiler attaching to a running process, first use today"),
    ("manual_db_migration", "an admin manually running a one-off database migration script"),
    ("ad_hoc_perf_probe", "an ad-hoc perf/strace probe attached to a production process"),
    ("unusual_hour_admin", "a routine admin command run at an unusual hour for this workstation"),
]


def _triples_to_families() -> list[Family]:
    out = []
    for (crit_n, crit_b), (safe_n, safe_b), (warn_n, warn_b) in TRIPLES:
        hint = _hint_for(crit_n)
        out.append(
            Family(
                crit_n,
                "crit",
                crit_b,
                difficulty="hard",
                pair_safe=safe_n,
                pair_warn=warn_n,
                fact_hint=hint,
            )
        )
        out.append(Family(safe_n, "safe", safe_b, difficulty="medium", fact_hint="normal"))
        out.append(Family(warn_n, "warn", warn_b, difficulty="medium", fact_hint="new_tool"))
    for n, b in EXTRA_SAFE:
        out.append(Family(n, "safe", b, difficulty="easy", fact_hint="normal"))
    for n, b in EXTRA_WARN:
        out.append(Family(n, "warn", b, difficulty="medium", fact_hint="new_tool"))
    return out


def _hint_for(name: str) -> str:
    if "tmpfs" in name or "memfd" in name:
        return "tmpfs_exec"
    if "container" in name:
        return "container"
    if "unexpected_parent" in name or "doc_macro" in name:
        return "unexpected_parent"
    if "masquerade" in name or "decoy" in name or "hidden" in name:
        return "hidden_or_masquerade"
    if "setuid" in name:
        return "setuid_odd"
    if "escalate" in name or "disable_security" in name:
        return "privileged"
    return "normal"


def _overlap(a: str, b: str) -> float:
    ta, tb = set(a.split("_")), set(b.split("_"))
    return len(ta & tb) / max(1, len(ta | tb))


def propose_and_merge(
    families: list[Family], client: LLMClient, model: str, n_per_class: int, rounds: int = 1
) -> list[Family]:
    by_name = {f.name: f for f in families}
    for label, _round in [(l, r) for r in range(rounds) for l in ("safe", "warn", "crit")]:
        text = client.chat(
            model,
            propose_families_prompt(n_per_class, by_name.keys(), label),
            system="Output strict JSON only, no prose.",
            temperature=1.0,
            max_tokens=300 + 90 * n_per_class,
        )
        try:
            items = extract_json(text)
        except ValueError:
            continue
        for item in items if isinstance(items, list) else []:
            try:
                name = str(item["name"]).strip().lower().replace(" ", "_")
                brief = str(item["brief"])
                difficulty = str(item.get("difficulty", "medium"))
            except (KeyError, TypeError):
                continue
            if not name or name in by_name:
                continue
            if any(_overlap(name, existing) > 0.6 for existing in by_name):
                continue
            pair_safe = pair_warn = ""
            if label == "crit":
                safes = [f.name for f in by_name.values() if f.label == "safe"]
                warns = [f.name for f in by_name.values() if f.label == "warn"]
                pair_safe = max(safes, key=lambda s: _overlap(name, s), default="")
                pair_warn = max(warns, key=lambda s: _overlap(name, s), default="")
            fam = Family(
                name,
                label,
                brief,
                difficulty=difficulty,
                pair_safe=pair_safe,
                pair_warn=pair_warn,
                fact_hint=_hint_for(name),
                source="llm",
            )
            by_name[name] = fam
    return list(by_name.values())


def write_toml(families: list[Family], out: Path) -> None:
    lines = [
        "# E19 synthetic-corpus taxonomy: behaviour families for the process-launch judge.",
        "# Generated by lab/scripts/build_taxonomy.py (hand-authored triples + LLM proposals,",
        "# deduped/curated). `label` is one of safe/warn/crit; crit families carry `pair_safe`",
        "# (and usually `pair_warn`), the benign/plausible-harmless look-alike at the same",
        "# surface so the classifier cannot key on exe name or keyword alone. `fact_hint` picks",
        "# the structural-facts distribution in gen_corpus_lib.FACT_PROFILES.",
        "",
    ]
    for f in sorted(families, key=lambda f: (f.label, f.name)):
        lines.append(f"[families.{f.name}]")
        for k, v in f.to_toml_dict().items():
            if isinstance(v, str):
                esc = v.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{k} = "{esc}"')
            else:
                lines.append(f"{k} = {v}")
        lines.append("")
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus_taxonomy.toml")
    ap.add_argument("--propose", type=int, default=0, help="families to propose per class, 0=skip")
    ap.add_argument("--model", default="qwen3.8")
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()
    families = _triples_to_families()
    if args.propose:
        families = propose_and_merge(families, LLMClient(), args.model, args.propose, args.rounds)
    problems = validate_taxonomy(families)
    if problems:
        for p in problems:
            print("WARN:", p)
    write_toml(families, Path(args.out))
    counts: dict[str, int] = {}
    for f in families:
        counts[f.label] = counts.get(f.label, 0) + 1
    print(f"wrote {len(families)} families to {args.out}: {counts}")


def pair_llm_crits(families: list[Family], client: LLMClient, model: str) -> list[Family]:
    """For every LLM-proposed crit family, ask for a benign and a plausibly-harmless twin at the
    same launch surface (same exe / similar argv, different parent, location or context), and
    replace the token-overlap pairing with them."""
    by_name = {f.name: f for f in families}
    for f in list(families):
        if f.label != "crit" or f.source != "llm":
            continue
        text = client.chat(
            model,
            (
                f"Hostile launch family '{f.name}': {f.brief}\n\n"
                "Write two LOOK-ALIKE single-process-launch families that use the same program "
                "and a similar command line but differ in launch context (parent process, "
                "location, user, freshness):\n"
                "  safe: ordinary everyday activity that a keyword rule would wrongly flag\n"
                "  warn: unusual but plausibly harmless\n"
                'Output JSON: {"safe_name": "...", "safe_brief": "...", "warn_name": "...", '
                '"warn_brief": "..."} with snake_case names and one-sentence briefs.'
            ),
            system="Output strict JSON only, no prose.",
            temperature=0.8,
            max_tokens=300,
        )
        try:
            d = extract_json(text)
            sn, wn = str(d["safe_name"]).strip().lower(), str(d["warn_name"]).strip().lower()
            sb, wb = str(d["safe_brief"]), str(d["warn_brief"])
        except (ValueError, KeyError, TypeError):
            continue
        sn = sn if sn not in by_name else f"{f.name}_benign"
        wn = wn if wn not in by_name else f"{f.name}_oneoff"
        by_name[sn] = Family(sn, "safe", sb, "hard", fact_hint="normal", source="llm")
        by_name[wn] = Family(wn, "warn", wb, "hard", fact_hint="new_tool", source="llm")
        by_name[f.name] = Family(
            f.name,
            f.label,
            f.brief,
            f.difficulty,
            pair_safe=sn,
            pair_warn=wn,
            fact_hint=f.fact_hint,
            source=f.source,
        )
    return list(by_name.values())


def pair_main() -> None:
    from gen_corpus_lib import load_taxonomy

    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default="corpus_taxonomy.toml")
    ap.add_argument("--model", default="qwen3.8")
    args = ap.parse_args(sys.argv[2:])
    fams = pair_llm_crits(load_taxonomy(args.taxonomy), LLMClient(), args.model)
    write_toml(fams, Path(args.taxonomy))
    print(f"wrote {len(fams)} families to {args.taxonomy}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pair":
        pair_main()
    else:
        main()

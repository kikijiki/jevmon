"""E19: shared library for the synthetic process-launch corpus generator.

Everything here is TEXT ONLY: invocation shapes (exe/argv/parent/cwd/uid + launch-time facts),
never executed, never real malware. jevmon judges one execve at a time, so every row is a single
launch and its immediate parent - not a script body, a session, a file write or a multi-step
narrative. A shell pipeline (`a | b`) is two launches sharing a parent shell; `sh -c '...'` is one
launch whose argv happens to contain a string. Only the row's own exe/argv is judged; parent_argv
is context the model reads, never a second verdict.

LLM calls go through an OpenAI-compatible `/v1/chat/completions` endpoint (local llama.cpp,
`http://localhost:8000/v1` by default). `LLMClient` is swappable for a stub in tests (anything
with a `.chat(model, prompt, **kw) -> str` method).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass

LABELS = ("safe", "warn", "crit")

# Row schema (one execve-level launch). `facts` is filled in by `synthesize_facts`, not the LLM:
# the LLM is unreliable at producing well-formed grammar for 9 small enum-like fields, and the
# facts module's own grammar is simple enough to sample directly from family-level hints.
ROW_FIELDS = (
    "exe",
    "argv",
    "parent_exe",
    "parent_argv",
    "cwd",
    "uid",
    "tty",
    "cgroup",
    "env_note",
)

_BAD_HOST = re.compile(r"\b(?!(?:192\.0\.2|198\.51\.100|203\.0\.113)\.)(?:\d{1,3}\.){3}\d{1,3}\b")
_ALLOWED_TLDS = (".example", ".invalid", ".test", ".localhost")
_REAL_LOOKING_DOMAIN = re.compile(
    r"\b[a-z0-9-]+\.(?:com|net|org|io|dev|co|ru|cn|xyz|top|info)\b", re.IGNORECASE
)
_LONG_B64 = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
_HEADER_START = re.compile(r"-----BEGIN [A-Z ]*(?:PRIVATE KEY|CERTIFICATE)")


def unsafe_text_reasons(text: str) -> list[str]:
    """Why `text` is rejected as unsafe-to-store, or `[]` if it looks fine.

    Not a security scanner: a cheap guard against the model slipping in a real-looking
    IP/domain, a long encoded blob (working payload body) or a key/cert header. RFC 5737 /
    example.com / .invalid are fine; the rest is flagged for review, not silently dropped.
    """
    reasons = []
    if _BAD_HOST.search(text):
        reasons.append("non-reserved IPv4 literal")
    for m in _REAL_LOOKING_DOMAIN.finditer(text):
        if not any(m.group(0).endswith(t) for t in _ALLOWED_TLDS) and "example.com" not in text:
            reasons.append(f"real-looking domain: {m.group(0)}")
            break
    if _LONG_B64.search(text):
        reasons.append("long encoded blob (200+ chars)")
    if _HEADER_START.search(text):
        reasons.append("key/certificate material")
    return reasons


# --------------------------------------------------------------------------------- LLM client


@dataclass
class LLMClient:
    base_url: str = os.environ.get("JEVMON_LLM_URL", "http://localhost:8000/v1")
    timeout: float = 900.0
    max_retries: int = 5

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.9,
        max_tokens: int = 2000,
        enable_thinking: bool = False,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        data = json.dumps(body).encode()
        delay = 2.0
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    j = json.loads(r.read())
                return j["choices"][0]["message"].get("content") or ""
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
                last_err = e
                # model may be `sleeping` (first request wakes it) or the shared iGPU is busy
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")


def extract_json(text: str) -> object:
    """Pull the first JSON array/object out of a model response (which may wrap it in prose or
    a ```json fence, and may include a leading <think> block some chat templates still emit)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start is None:
        raise ValueError("no JSON found in model response")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unterminated JSON in model response")


# --------------------------------------------------------------------------------- taxonomy


@dataclass(frozen=True, slots=True)
class Family:
    name: str
    label: str  # safe / warn / crit
    brief: str
    difficulty: str = "medium"  # easy / medium / hard
    pair_safe: str = ""  # name of the benign look-alike family (crit/warn only)
    pair_warn: str = ""  # name of the plausibly-harmless look-alike (crit only)
    fact_hint: str = "normal"  # a key into FACT_PROFILES (synthesize_facts)
    source: str = "hand"  # "hand" or "llm"

    def to_toml_dict(self) -> dict:
        d = {
            "label": self.label,
            "brief": self.brief,
            "difficulty": self.difficulty,
            "fact_hint": self.fact_hint,
            "source": self.source,
        }
        if self.pair_safe:
            d["pair_safe"] = self.pair_safe
        if self.pair_warn:
            d["pair_warn"] = self.pair_warn
        return d


def load_taxonomy(path: str) -> list[Family]:
    import tomllib

    with open(path, "rb") as f:
        doc = tomllib.load(f)
    out = []
    for name, d in doc.get("families", {}).items():
        out.append(
            Family(
                name=name,
                label=d["label"],
                brief=d["brief"],
                difficulty=d.get("difficulty", "medium"),
                pair_safe=d.get("pair_safe", ""),
                pair_warn=d.get("pair_warn", ""),
                fact_hint=d.get("fact_hint", "normal"),
                source=d.get("source", "hand"),
            )
        )
    return out


def validate_taxonomy(families: list[Family]) -> list[str]:
    """Structural problems: unknown labels, dangling pairs, a crit family with no safe pairing."""
    problems = []
    names = {f.name for f in families}
    for f in families:
        if f.label not in LABELS:
            problems.append(f"{f.name}: bad label {f.label!r}")
        if f.pair_safe and f.pair_safe not in names:
            problems.append(f"{f.name}: pair_safe {f.pair_safe!r} not a family")
        if f.pair_warn and f.pair_warn not in names:
            problems.append(f"{f.name}: pair_warn {f.pair_warn!r} not a family")
        if f.label == "crit" and not f.pair_safe:
            problems.append(f"{f.name}: crit family with no pair_safe hard negative")
    return problems


# --------------------------------------------------------------------------------- generation


GEN_SYSTEM = (
    "You are generating a SYNTHETIC, TEXT-ONLY dataset for a defensive security classifier. "
    "You never produce working malware, exploit code, shellcode, long script bodies or real "
    "secrets - only short process-launch SHAPES (one process's exe, argv, parent, cwd, uid). "
    "Use RFC 5737 addresses (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) or example.com / "
    ".invalid hostnames, and obviously-fake credentials. Output strict JSON only, no prose."
)


def build_prompt(family: Family, n: int) -> str:
    return (
        f"Family: {family.name}\n"
        f"Intended class: {family.label} ({_class_sentence(family.label)})\n"
        f"Brief: {family.brief}\n\n"
        f"Generate {n} varied, single-process-launch examples of this family as a JSON array. "
        "Each element is an object with exactly these keys:\n"
        '  "exe": absolute path of the executed binary\n'
        '  "argv": array of strings, argv[0] first (as it would really appear)\n'
        '  "parent_exe": absolute path of the parent process\n'
        '  "parent_argv": array of strings for the parent\'s own command line (may be empty)\n'
        '  "cwd": absolute working directory of the launch\n'
        '  "uid": integer uid (0 for root, else a plausible non-root uid like 1000-1004)\n'
        '  "tty": "yes" or "no" (whether a real terminal is attached)\n'
        '  "env_note": one short phrase of environment color (e.g. "fresh install", '
        '"inside container"), or "" if nothing notable\n\n'
        "Vary exe choice, argv shape, parent process, cwd and uid across the examples - these "
        "must not be near-duplicates of each other. This is ONE launch each: a shell pipeline "
        "stage, not a whole script; if the shape is `sh -c '...'`, the argv is that literal "
        "string, not a description of what it does. Output a JSON array of exactly "
        f"{n} objects, nothing else."
    )


def _class_sentence(label: str) -> str:
    return {
        "safe": "expected everyday activity, no action needed",
        "warn": "out of the ordinary but plausibly harmless, worth a glance",
        "crit": "looks hostile from the launch alone",
    }[label]


def propose_families_prompt(n: int, existing: Iterable[str], label: str) -> str:
    existing_list = ", ".join(sorted(existing)[:120]) or "(none yet)"
    return (
        f"Propose {n} NEW behaviour families for a process-launch classifier, all intended "
        f"class '{label}' ({_class_sentence(label)}).\n"
        f"Families already used (do not repeat these or close synonyms): {existing_list}\n\n"
        "Each family is a short, specific *launch shape* (not a vague category). Output a JSON "
        "array of objects with keys:\n"
        '  "name": short snake_case identifier\n'
        '  "brief": one sentence describing the launch shape\n'
        '  "difficulty": "easy", "medium" or "hard" (hard = easy to confuse with another class)\n'
        "Output only the JSON array."
    )


def generate_family_rows(
    client: LLMClient, model: str, family: Family, n: int, *, temperature: float = 0.9
) -> list[dict]:
    text = client.chat(
        model,
        build_prompt(family, n),
        system=GEN_SYSTEM,
        temperature=temperature,
        max_tokens=400 + 220 * n,
    )
    items = extract_json(text)
    if not isinstance(items, list):
        raise TypeError(f"{family.name}: model did not return a JSON array")
    return items


# --------------------------------------------------------------------------------- facts synthesis

# Plausible fact bundles keyed by a coarse hint the taxonomy assigns per family (not per row, and
# not a verdict rule - it only fills in the 9 structural-fact fields so any context format can be
# rendered later). Each hint is a *distribution* over small variations, sampled per row, so rows
# in one family are not identical.
FACT_PROFILES: dict[str, dict] = {
    "normal": {
        "exe_state": ["normal"],
        "argv0_vs_exe": ["match"],
        "stdio": ["0=tty 1=tty 2=tty", "0=pipe 1=pipe 2=pipe", "0=null 1=pipe 2=pipe"],
        "session": ["tty=yes sid_leader=no loginuid=set", "tty=no sid_leader=yes loginuid=set"],
        "privileges_root": False,
        "namespaces": ["host"],
        "writable_dir": False,
        "setuid": False,
        "world_writable": False,
    },
    "container": {
        "exe_state": ["normal"],
        "argv0_vs_exe": ["match"],
        "stdio": ["0=pipe 1=pipe 2=pipe", "0=null 1=pipe 2=pipe"],
        "session": ["tty=no sid_leader=yes loginuid=unset"],
        "privileges_root": True,
        "namespaces": ["differs(mnt,pid,net,uts,ipc)", "differs(mnt,pid,net,user,uts,ipc)"],
        "writable_dir": False,
        "setuid": False,
        "world_writable": False,
    },
    "new_tool": {
        "exe_state": ["normal"],
        "argv0_vs_exe": ["match"],
        "stdio": ["0=tty 1=tty 2=tty", "0=pipe 1=pipe 2=pipe"],
        "session": ["tty=yes sid_leader=no loginuid=set"],
        "privileges_root": False,
        "namespaces": ["host"],
        "writable_dir": False,
        "setuid": False,
        "world_writable": False,
    },
    "tmpfs_exec": {
        "exe_state": ["normal", "deleted"],
        "argv0_vs_exe": ["match", "differs"],
        "stdio": ["0=null 1=pipe 2=pipe", "0=pipe 1=pipe 2=pipe"],
        "session": ["tty=no sid_leader=no loginuid=unset"],
        "privileges_root": False,
        "namespaces": ["host"],
        "writable_dir": True,
        "setuid": False,
        "world_writable": True,
    },
    "unexpected_parent": {
        "exe_state": ["normal"],
        "argv0_vs_exe": ["match"],
        "stdio": ["0=pipe 1=pipe 2=pipe"],
        "session": ["tty=no sid_leader=yes loginuid=unset"],
        "privileges_root": False,
        "namespaces": ["host"],
        "writable_dir": False,
        "setuid": False,
        "world_writable": False,
    },
    "hidden_or_masquerade": {
        "exe_state": ["normal", "deleted"],
        "argv0_vs_exe": ["differs", "match"],
        "stdio": ["0=null 1=null 2=null", "0=pipe 1=pipe 2=pipe"],
        "session": ["tty=no sid_leader=yes loginuid=unset"],
        "privileges_root": False,
        "namespaces": ["host"],
        "writable_dir": False,
        "setuid": False,
        "world_writable": False,
    },
    "privileged": {
        "exe_state": ["normal"],
        "argv0_vs_exe": ["match"],
        "stdio": ["0=tty 1=tty 2=tty", "0=pipe 1=pipe 2=pipe"],
        "session": ["tty=yes sid_leader=no loginuid=set", "tty=no sid_leader=yes loginuid=unset"],
        "privileges_root": True,
        "namespaces": ["host"],
        "writable_dir": False,
        "setuid": False,
        "world_writable": False,
    },
    "setuid_odd": {
        "exe_state": ["normal"],
        "argv0_vs_exe": ["match"],
        "stdio": ["0=tty 1=tty 2=tty"],
        "session": ["tty=yes sid_leader=no loginuid=set"],
        "privileges_root": False,
        "namespaces": ["host"],
        "writable_dir": False,
        "setuid": True,
        "world_writable": False,
    },
}


def synthesize_facts(hint: str, rng: random.Random) -> dict[str, str]:
    prof = FACT_PROFILES.get(hint, FACT_PROFILES["normal"])
    uid_eff = "0" if prof["privileges_root"] else str(rng.choice([1000, 1000, 1001, 33]))
    real_uid = "0" if prof["privileges_root"] and rng.random() < 0.7 else uid_eff
    cap = "0000000000000000" if not prof["privileges_root"] else "000001ffffffffff"
    ns = rng.choice(prof["namespaces"])
    ns_full = (
        ns
        if ns == "host"
        else f"{ns} [mnt={'differs' if 'mnt' in ns else 'same'} pid={'differs' if 'pid' in ns else 'same'} "
        f"net={'differs' if 'net' in ns else 'same'} user={'differs' if 'user' in ns else 'same'} "
        f"uts={'differs' if 'uts' in ns else 'same'} ipc={'differs' if 'ipc' in ns else 'same'} "
        f"cgroup=same time=same] uid_map={'remapped' if 'user' in ns else 'identity'}"
    )
    mode = "1777" if prof["world_writable"] else ("4755" if prof["setuid"] else "0755")
    return {
        "exe_state": rng.choice(prof["exe_state"]),
        "argv0_vs_exe": rng.choice(prof["argv0_vs_exe"]),
        "stdio": rng.choice(prof["stdio"]),
        "session": rng.choice(prof["session"]),
        "workdir": "",  # filled from the row's own cwd by the caller
        "cgroup_path": rng.choice(
            ["user.slice/session.scope", "system.slice/app.service", "<PATH>"]
        ),
        "privileges": f"uid={real_uid} euid={uid_eff} capeff={cap} nnp=0 seccomp=0",
        "namespaces": ns_full,
        "exe_file": (
            f"owner={'0' if prof['privileges_root'] else rng.choice([1000, 0])} mode={mode} "
            f"setuid={'yes' if prof['setuid'] else 'no'} "
            f"world_writable={'yes' if prof['world_writable'] else 'no'}"
        ),
    }


@dataclass(slots=True)
class Row:
    exe: str
    argv: list[str]
    parent_exe: str
    parent_argv: list[str]
    cwd: str
    uid: int
    tty: str
    env_note: str
    family: str
    label: str
    fact_hint: str

    def to_dict(self) -> dict:
        return {
            "exe": self.exe,
            "argv": self.argv,
            "parent_exe": self.parent_exe,
            "parent_argv": self.parent_argv,
            "cwd": self.cwd,
            "uid": self.uid,
            "tty": self.tty,
            "env_note": self.env_note,
            "family": self.family,
            "label": self.label,
            "fact_hint": self.fact_hint,
        }


def coerce_row(item: dict, family: Family) -> Row | None:
    """A raw LLM dict -> `Row`, or `None` if it is malformed (missing fields, wrong types)."""
    try:
        exe = str(item["exe"]).strip()
        argv = [str(a) for a in item["argv"]]
        parent_exe = str(item.get("parent_exe", "")).strip()
        parent_argv = [str(a) for a in item.get("parent_argv", []) or []]
        cwd = str(item.get("cwd", "/")).strip() or "/"
        uid = int(item.get("uid", 1000))
        tty = "yes" if str(item.get("tty", "no")).lower().startswith("y") else "no"
        env_note = str(item.get("env_note", "") or "")
    except (KeyError, TypeError, ValueError):
        return None
    if not exe or not exe.startswith("/") or not argv:
        return None
    if not cwd.startswith("/"):
        return None
    if len(argv) > 40 or any(len(a) > 400 for a in argv):
        return None
    if len(" ".join(argv)) > 1200:
        return None
    return Row(
        exe=exe,
        argv=argv,
        parent_exe=parent_exe,
        parent_argv=parent_argv,
        cwd=cwd,
        uid=uid,
        tty=tty,
        env_note=env_note,
        family=family.name,
        label=family.label,
        fact_hint=family.fact_hint,
    )


def row_signature(row: dict) -> str:
    """Near-duplicate key: exe + first N argv tokens + parent_exe (ignores volatile bits like
    random ports/uuids by truncating rather than normalising - cheap, good enough for a dedupe
    pass; exact dups are also caught since this is a prefix of the exact-match key)."""
    argv = " ".join(row["argv"])[:160]
    return f"{row['exe']}\x00{argv}\x00{row['parent_exe']}"

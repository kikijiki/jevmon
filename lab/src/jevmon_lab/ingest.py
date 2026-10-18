"""External-dataset ingestion: a common launch schema, label-mapping tables, and rendering.

Every external source is converted to rows of one shape, one execve-level launch per row:
`exe`, `argv`, optional `cwd`, `uid`/`user`, `parent_exe`/`parent_argv`. A shell command line
(`a | b && c`) becomes one driver row (`sh -c <line>`, carrying the source label) plus one child
row per external program in it (label null), the same convention the lab uses for its scenarios.

Nothing here judges. Labels come from per-source TOML tables (`lab/ingest/*.toml`) keyed on the
source's own label values; the context text is the one the real `ContextBuilder` renders for the
shipped format (variant A: header, exe, argv template, redacted argv), built from an `Event` shaped
like the exec source's, so heads and zero-shot Laya see these rows exactly like lab rows. Nothing
is executed: command text is only tokenised.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from jevmon.core import MemoryCounts
from jevmon.detect.context import (
    ContextBuilder,
    ContextSettings,
    EnvironmentStats,
    ExecObservation,
    parse_exec_state,
)
from jevmon.detect.system_facts import StructuralFacts
from jevmon.events import Event
from jevmon.redact import redact_argv_text
from jevmon.sanitize import clean
from jevmon.templates import masked_template

INGEST_DIR = Path(__file__).resolve().parents[2] / "ingest"
REPO = Path(__file__).resolve().parents[3]
# Where the lab's own launches live (all gitignored): cached embedding stores keyed by context
# text, and the archived captures. Missing paths are skipped, and the dedupe report says how many
# lab keys it had.
DEFAULT_LAB_SOURCES = (
    REPO / "lab" / "cache" / "emb-english-mean-512.pkl",
    *sorted((REPO / "data" / "archive").glob("lab-data-*")),
    REPO / "lab-data",
)
LABELS = ("safe", "warn", "crit")
TARGETS = (*LABELS, "null", "drop")

# ---------------------------------------------------------------- schema


@dataclass(slots=True)
class Launch:
    """One process launch as a source describes it. Unknown fields stay empty / None."""

    exe: str
    argv: list[str]
    cwd: str = ""
    uid: int | None = None
    user: str = ""
    parent_exe: str = ""
    parent_argv: str = ""


@dataclass(slots=True)
class Row:
    """A converted launch with its provenance, label and rendered context text."""

    source: str
    role: str  # test | aug_train | background
    kind: str  # driver | child | launch | direct
    launch: Launch
    label: str | None = None
    rule: str = ""
    uncertain: str = ""  # why the mapped label is shaky ("" = no known doubt)
    family: str = ""
    count: int = 1
    parent_id: str = ""
    source_label: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    context: str = ""
    key: str = ""

    def ident(self) -> str:
        blob = json.dumps(
            [self.source, self.kind, self.launch.exe, self.launch.argv, self.parent_id],
            ensure_ascii=True,
        )
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def to_json(self) -> dict:
        la = self.launch
        out = {
            "id": self.ident(),
            "source": self.source,
            "role": self.role,
            "kind": self.kind,
            "exe": la.exe,
            "argv": la.argv,
            "cwd": la.cwd,
            "uid": la.uid,
            "user": la.user,
            "parent_exe": la.parent_exe,
            "parent_argv": la.parent_argv,
            "label": self.label,
            "rule": self.rule,
            "uncertain": self.uncertain,
            "family": self.family,
            "key": self.key,
            "count": self.count,
            "parent_id": self.parent_id,
            "source_label": self.source_label,
            "context": self.context,
        }
        out.update(self.extra)
        return out


# ---------------------------------------------------------------- label mapping tables


def load_toml(name: str) -> dict:
    """Read `lab/ingest/<name>.toml`."""
    return tomllib.loads((INGEST_DIR / f"{name}.toml").read_text(encoding="utf-8"))


def _matches(cond: Mapping, fields: Mapping) -> bool:
    for key, want in cond.items():
        if key.endswith("_any"):
            have = fields.get(key[:-4]) or []
            if isinstance(have, str):
                have = [have]
            if not set(have) & set(want):
                return False
        elif key.endswith("_none"):
            have = fields.get(key[:-5]) or []
            if isinstance(have, str):
                have = [have]
            if set(have) & set(want):
                return False
        else:
            have = fields.get(key)
            if have not in want:
                return False
    return True


def apply_rules(rules: Sequence[Mapping], fields: Mapping) -> tuple[str, str, str]:
    """First matching rule wins: returns `(target, rule_id, uncertain_reason)`.

    A rule is `{id, when = {field = [values] | field_any = [values] | field_none = [values]},
    to = safe|warn|crit|null|drop, uncertain = "reason"}`. `to` is validated; no match raises, so
    a source value the table does not know about fails loudly instead of being labelled by default.
    """
    for rule in rules:
        to = rule["to"]
        if to not in TARGETS:
            raise ValueError(f"rule {rule.get('id')}: bad target {to!r}")
        if _matches(rule.get("when", {}), fields):
            return to, str(rule.get("id", "")), str(rule.get("uncertain", ""))
    raise ValueError(f"no mapping rule matches {dict(fields)!r}")


# ---------------------------------------------------------------- shell command lines


@dataclass(slots=True)
class Parsed:
    programs: list[list[str]]  # argv of every external program, in order
    compound: bool  # operators, redirections, substitutions or here-documents present
    ok: bool = True  # False when the line could not be tokenised (unbalanced quote)


_OP_CHARS = "|&;()\n"


def _read_subst(text: str, i: int) -> tuple[str, int]:
    """Body of a `$( ... )` starting after the opening paren at `i`; returns (body, next index)."""
    depth, j, quote = 1, i, ""
    while j < len(text):
        c = text[j]
        if quote:
            if c == "\\" and quote == '"':
                j += 1
            elif c == quote:
                quote = ""
        elif c in "'\"":
            quote = c
        elif c == "\\":
            j += 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[i:j], j + 1
        j += 1
    return text[i:], len(text)


def _split(text: str, shell: Mapping, depth: int = 0) -> Parsed:
    """Tokenise `text` into simple commands. Quotes and backslashes are honoured; `$(...)` and
    backtick bodies are parsed as extra command lists; here-document bodies are skipped."""
    words: list[list[str]] = [[]]  # per segment: finished words
    cur: list[str] = []  # characters of the word being read
    in_word = False
    compound = False
    extra: list[list[str]] = []
    heredocs: list[tuple[str, bool]] = []  # pending (delimiter, strip_tabs)
    i, n = 0, len(text)

    def end_word() -> None:
        nonlocal in_word
        if in_word:
            words[-1].append("".join(cur))
            cur.clear()
            in_word = False

    def end_segment() -> None:
        end_word()
        if words[-1]:
            words.append([])

    while i < n:
        c = text[i]
        if c == "\\":
            if i + 1 < n and text[i + 1] == "\n":
                i += 2
                continue
            cur.append(text[i + 1] if i + 1 < n else "")
            in_word = True
            i += 2
        elif c == "'":
            j = text.find("'", i + 1)
            if j < 0:
                return Parsed([], True, False)
            cur.append(text[i + 1 : j])
            in_word = True
            i = j + 1
        elif c == '"':
            j, buf = i + 1, []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n and text[j + 1] in '"\\$`\n':
                    buf.append(text[j + 1])
                    j += 2
                    continue
                if text.startswith("$(", j):
                    body, k = _read_subst(text, j + 2)
                    if depth < 4:
                        sub = _split(body, shell, depth + 1)
                        extra += sub.programs
                    buf.append("$(" + body + ")")
                    compound = True
                    j = k
                    continue
                buf.append(text[j])
                j += 1
            if j >= n:
                return Parsed([], True, False)
            cur.append("".join(buf))
            in_word = True
            i = j + 1
        elif text.startswith("$(", i):
            body, k = _read_subst(text, i + 2)
            if depth < 4:
                extra += _split(body, shell, depth + 1).programs
            cur.append("$(" + body + ")")
            in_word = True
            compound = True
            i = k
        elif c == "`":
            j = text.find("`", i + 1)
            if j < 0:
                return Parsed([], True, False)
            if depth < 4:
                extra += _split(text[i + 1 : j], shell, depth + 1).programs
            cur.append(text[i : j + 1])
            in_word = True
            compound = True
            i = j + 1
        elif c in "<>":
            compound = True
            if text.startswith("<<<", i):
                end_word()
                i += 3
            elif text.startswith("<<", i):
                end_word()
                i += 2
                strip = text.startswith("-", i)
                i += strip
                while i < n and text[i] in " \t":
                    i += 1
                m = re.match(r"""(['"]?)([^\s'"<>|&;()]+)\1""", text[i:])
                if m:
                    heredocs.append((m.group(2), strip))
                    i += m.end()
                continue
            else:
                # drop an adjacent file descriptor number (`2>`) and the operator itself
                if in_word and "".join(cur).isdigit():
                    cur.clear()
                    in_word = False
                else:
                    end_word()
                i += 1
                while i < n and text[i] in "<>&|":
                    i += 1
            # the redirection target is not an argument
            while i < n and text[i] in " \t":
                i += 1
            target_end = i
            while target_end < n and text[target_end] not in " \t\n" + "|&;()":
                if text[target_end] in "'\"":
                    q = text.find(text[target_end], target_end + 1)
                    target_end = q + 1 if q > 0 else n
                else:
                    target_end += 1
            i = target_end
        elif c in _OP_CHARS:
            if c == "\n" or c in "|&;()":
                if c != "\n":
                    compound = True
                if c == "\n" and heredocs:
                    i += 1
                    while heredocs:
                        delim, strip = heredocs.pop(0)
                        while i < n:
                            j = text.find("\n", i)
                            line = text[i : j if j >= 0 else n]
                            i = j + 1 if j >= 0 else n
                            if (line.lstrip("\t") if strip else line) == delim:
                                break
                    end_segment()
                    continue
                end_segment()
                # swallow doubled operators (&&, ||, ;;, |&)
                if i + 1 < n and c in "|&;" and text[i + 1] in "|&;":
                    i += 1
                i += 1
        elif c in " \t":
            end_word()
            i += 1
        elif c == "#" and not in_word:
            while i < n and text[i] != "\n":
                i += 1
        else:
            cur.append(c)
            in_word = True
            i += 1
    end_segment()

    keywords = set(shell["keywords"])
    skip_first = set(shell["skip_segment"])
    builtins = set(shell["builtins"])
    programs: list[list[str]] = []
    for seg in words:
        seg = list(seg)
        while seg and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", seg[0]):
            seg.pop(0)
            compound = True
        while seg and seg[0] in keywords:
            seg.pop(0)
            compound = True
        if not seg or seg[0] in skip_first:
            compound = compound or bool(seg)
            continue
        if seg[0] in builtins:
            compound = True
            continue
        programs.append(seg)
    programs = extra + programs
    return Parsed(programs, compound or len(programs) != 1, True)


def parse_command_line(text: str, shell: Mapping | None = None) -> Parsed:
    """Programs that a shell would launch for `text` (builtins and keywords produce no program)."""
    return _split(text, shell or load_toml("shell"))


# ---------------------------------------------------------------- keys and families

_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_URL_HOST = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s'\"]+")
_DIGITS = re.compile(r"\d+")


def template_key(argv: Sequence[str], max_tokens: int = 24) -> str:
    """The launch's normalised key: the shipped `argv(template)` over whitespace tokens, then
    IPv4 addresses, URL hosts and digit runs masked. Two launches with the same key are the same
    template for deduplication and leakage control."""
    text = masked_template(" ".join(argv).split(), max_tokens)
    text = _URL_HOST.sub(r"\1<HOST>", text)
    text = _IPV4.sub("<IP>", text)
    return _DIGITS.sub("<N>", text)


def lab_keys(texts: Iterable[str]) -> set[str]:
    """Template keys of lab context texts (cached embedding keys or dataset contexts): the
    `argv(redacted)` segment is re-keyed with `template_key` so lab and external keys agree."""
    keys: set[str] = set()
    for text in texts:
        try:
            parts = dict(parse_exec_state(text))
        except ValueError:
            continue
        argv = parts.get("argv", "")
        if argv.startswith("argv(redacted): "):
            keys.add(template_key(argv[len("argv(redacted): ") :].split()))
    return keys


# ---------------------------------------------------------------- rendering


class _NoFacts:
    """External rows carry no filesystem facts and must not stat attacker-chosen paths."""

    def inspect(self, exe: str, now: float, recent_seconds: float) -> StructuralFacts:
        return StructuralFacts()


class Renderer:
    """Render launches through the real `ContextBuilder` with the shipped settings.

    The `Event` has the fields the exec source writes (redacted exe/argv/cwd), so `ExecObservation`
    and the builder take the same path as a live launch; only the structural facts are absent.
    """

    def __init__(self, settings: ContextSettings | None = None) -> None:
        self.settings = settings or ContextSettings()
        self.builder = ContextBuilder(MemoryCounts(), _NoFacts(), self.settings)

    def render(self, launch: Launch) -> str:
        argv = list(launch.argv)
        fields = {
            "pid": 0,
            "ppid": 0,
            "uid": launch.uid if launch.uid is not None else -1,
            "exe": redact_argv_text([launch.exe]) if launch.exe else "",
            "argv0": clean(Path(argv[0]).name, 256) if argv else "",
            "path0": clean(argv[0], 512) if argv else "",
            "argv": redact_argv_text(argv),
            "cwd": redact_argv_text([launch.cwd]).strip() if launch.cwd else "",
            "cgroup": "",
            "parent_comm": "",
            "parent_exe": launch.parent_exe,
            "parents": "",
            "parent_argv": "",
        }
        event = Event(time=0.0, source="exec", subject="0", kind="exec", fields=fields)
        obs = ExecObservation.from_event(event)
        return self.builder.observe_exec(obs, EnvironmentStats()).state


# ---------------------------------------------------------------- shell command rows


def command_rows(
    command: str,
    *,
    source: str,
    role: str,
    label: str | None,
    rule: str,
    uncertain: str,
    family: str,
    source_label: dict,
    driver: Mapping,
    shell: Mapping | None = None,
    extra: dict | None = None,
) -> list[Row]:
    """A shell command line as rows: the driver (`sh -c <line>`, labelled), its programs as
    children (label null), and, for a single plain program, the same command as a direct launch
    (`kind="direct"`, labelled like the driver, for the secondary form of the test)."""
    parsed = parse_command_line(command, shell)
    drv = Launch(exe=driver["exe"], argv=[driver["argv0"], "-c", command])
    base = Row(
        source,
        role,
        "driver",
        drv,
        label,
        rule,
        uncertain,
        family,
        source_label=dict(source_label),
        extra=dict(extra or {}),
    )
    base.key = template_key(drv.argv)
    rows = [base]
    for prog in parsed.programs:
        child = Row(
            source,
            role,
            "child",
            Launch(exe=prog[0], argv=prog, parent_exe=drv.exe, parent_argv=" ".join(drv.argv)),
            None,
            family=family,
            parent_id=base.ident(),
            extra={"unparsed": False},
        )
        child.key = template_key(prog)
        rows.append(child)
    if parsed.ok and not parsed.compound and len(parsed.programs) == 1:
        prog = parsed.programs[0]
        direct = Row(
            source,
            role,
            "direct",
            Launch(exe=prog[0], argv=prog),
            label,
            rule,
            uncertain,
            family,
            source_label=dict(source_label),
            parent_id=base.ident(),
            extra=dict(extra or {}),
        )
        direct.key = template_key(prog)
        rows.append(direct)
    if not parsed.ok:
        base.extra["unparsed"] = True
    return rows


def render_all(rows: Iterable[Row]) -> list[Row]:
    """Fill `context` for every row with one shared renderer (fresh counts per call)."""
    renderer = Renderer()
    out = list(rows)
    for row in out:
        row.context = renderer.render(row.launch)
    return out


# ---------------------------------------------------------------- dedupe and writing


def dedupe(rows: Iterable[Row], *, forbidden: set[str] | None = None) -> tuple[list[Row], dict]:
    """One row per (kind, key); repeats add to `count`. Rows whose key is in `forbidden` (the lab
    sets, or another split of the same source) are dropped, and so are keys that carry two
    different labels (the same template cannot be both). Returns (rows, stats)."""
    forbidden = forbidden or set()
    seen: dict[tuple[str, str], Row] = {}
    conflicts: set[tuple[str, str]] = set()
    stats = {"in": 0, "duplicates": 0, "in_lab_or_other_split": 0, "label_conflict": 0}
    out: list[Row] = []
    for row in rows:
        stats["in"] += 1
        if row.key in forbidden:
            stats["in_lab_or_other_split"] += 1
            continue
        k = (row.kind, row.key)
        if k in seen:
            first = seen[k]
            first.count += row.count
            stats["duplicates"] += 1
            if row.label != first.label:
                conflicts.add(k)
            continue
        seen[k] = row
        out.append(row)
    if conflicts:
        gone = {id(seen[k]) for k in conflicts}
        stats["label_conflict"] = len(gone)
        out = [r for r in out if id(r) not in gone]
    stats["out"] = len(out)
    return out, stats


def load_lab_keys(paths: Iterable[Path]) -> set[str]:
    """Template keys of the lab's own launches, from cached embedding stores (`*.pkl`, keyed by
    context text) and capture directories (`dataset.jsonl` with a `context` per row)."""
    import pickle

    keys: set[str] = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        if path.suffix == ".pkl":
            keys |= lab_keys(pickle.loads(path.read_bytes()))
        else:
            file = path / "dataset.jsonl" if path.is_dir() else path
            keys |= lab_keys(
                json.loads(ln)["context"] for ln in file.read_text().splitlines() if ln.strip()
            )
    return keys


def write_dataset(out_dir: Path, rows: Iterable[Row], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with (out_dir / "dataset.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_json(), ensure_ascii=True, sort_keys=True) + "\n")
    meta = dict(meta, rows=len(rows))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def read_dataset(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def summarize(rows: Iterable[Row | Mapping]) -> dict:
    """Counts per (role, kind, label) and per family for the labelled driver/launch rows."""
    table: dict[str, int] = {}
    for r in rows:
        d = r.to_json() if isinstance(r, Row) else r
        k = f"{d['role']}/{d['kind']}/{d['label']}"
        table[k] = table.get(k, 0) + 1
    return table

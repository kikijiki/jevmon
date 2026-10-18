"""Owner-written allow rules: launches matching a rule are not judged; everything else is.

The code ships no rules (`[filter] allow` is empty by default). A rule is a string
`"<exe> <args>"` in the style of Claude Code's permission rules:

- The exe part is a glob. When it starts with `/` or `*` it is matched against the full exe path
  (strong form: a renamed or copied binary elsewhere does not match). Otherwise it is matched
  against the exe basename (weak form: a binary named `git` anywhere matches).
- The args part is a glob over the space-joined argv[1:]. `*` is the only wildcard and matches any
  run of characters, spaces included; every other character is literal. A rule with only an exe
  part matches that exe with no arguments. A trailing ` *` requires a space boundary (`git diff *`
  does not match `git diffx`) and also matches the bare `git diff`.
- A trailing `:*` is an alias for ` *` (`git diff:*` is `git diff *`).
- A backslash escapes the next character when that is `*` or `\\`: `\\*` is a literal star and
  `\\\\` a literal backslash. A backslash before anything else is literal.
- `<exe> *` matches that exe with any arguments, including none.

Matching is a pure function of the exe path and argv; rules are compiled once.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field


class RuleError(ValueError):
    """An allow rule is malformed."""


@dataclass(frozen=True, slots=True)
class Rule:
    text: str
    exe: re.Pattern[str]
    args: re.Pattern[str]
    strong: bool
    exact: str | None  # the exe part when it has no wildcard (dict-indexed lookup)


@dataclass(frozen=True, slots=True)
class FilterSettings:
    allow: tuple[str, ...] = ()
    store_skipped: bool = False

    def __post_init__(self) -> None:
        compile_rules(self.allow)  # reject bad rules at load


def _split_glob(pattern: str) -> list[str]:
    """The literal runs of a glob, split at its unescaped stars."""
    parts = [""]
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\" and i + 1 < len(pattern) and pattern[i + 1] in "*\\":
            parts[-1] += pattern[i + 1]
            i += 2
        elif char == "*":
            parts.append("")
            i += 1
        else:
            parts[-1] += char
            i += 1
    return parts


def _glob(pattern: str) -> re.Pattern[str]:
    return re.compile(".*".join(re.escape(part) for part in _split_glob(pattern)), re.DOTALL)


def escape(text: str) -> str:
    """`text` as a rule fragment that matches exactly itself."""
    return text.replace("\\", "\\\\").replace("*", "\\*")


def parse_rule(text: str) -> Rule:
    rule = text.strip()
    if not rule:
        raise RuleError(f"empty allow rule {text!r}")
    if rule.endswith(":*"):
        rule = rule[:-2].rstrip() + " *"
    parts = rule.split(None, 1)
    exe = parts[0]
    exe_parts = _split_glob(exe)
    rest = " ".join(parts[1].split()) if len(parts) > 1 else ""
    if exe == "*" and not rest:
        raise RuleError(f"allow rule {text!r} would match every launch")
    if rest == "*":
        args = re.compile(".*", re.DOTALL)
    elif rest.endswith(" *"):
        args = re.compile(f"{_glob(rest[:-2]).pattern}(?: .*)?", re.DOTALL)
    else:
        args = _glob(rest)
    return Rule(
        text=text,
        exe=_glob(exe),
        args=args,
        strong=exe[0] in "/*",
        exact=exe_parts[0] if len(exe_parts) == 1 else None,
    )


def compile_rules(rules: Sequence[str]) -> list[Rule]:
    """Parse every rule; one error lists all the offending rules."""
    parsed: list[Rule] = []
    errors: list[str] = []
    for text in rules:
        try:
            parsed.append(parse_rule(text))
        except RuleError as exc:
            errors.append(str(exc))
    if errors:
        raise RuleError("; ".join(errors))
    return parsed


def basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


@dataclass(slots=True)
class AllowList:
    """Compiled rules plus hit counters. `match` returns the index of the first matching rule."""

    rules: list[Rule]
    hits: list[int] = field(default_factory=list)
    skipped: int = 0
    _strong: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _weak: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _wild: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.hits = [0] * len(self.rules)
        for index, rule in enumerate(self.rules):
            if rule.exact is None:
                self._wild.append(index)
            else:
                table = self._strong if rule.strong else self._weak
                table.setdefault(rule.exact, []).append(index)

    @classmethod
    def from_strings(cls, rules: Sequence[str]) -> "AllowList":
        return cls(compile_rules(rules))

    def replaced(self, rules: Sequence[str]) -> "AllowList":
        """A new list for `rules` that keeps the hit counters of the rules it shares with this one."""
        new = AllowList.from_strings(rules)
        old = {rule.text: n for rule, n in zip(self.rules, self.hits, strict=True)}
        new.hits = [old.get(rule.text, 0) for rule in new.rules]
        new.skipped = self.skipped
        return new

    def match(self, exe: str, args: str) -> int | None:
        """First rule matching exe path `exe` and the space-joined argv[1:] `args`."""
        if not self.rules or not exe:
            return None
        name = basename(exe)
        found: int | None = None
        for index in sorted([*self._strong.get(exe, ()), *self._weak.get(name, ()), *self._wild]):
            rule = self.rules[index]
            subject = exe if rule.strong else name
            if (rule.exact is not None or rule.exe.fullmatch(subject)) and rule.args.fullmatch(
                args
            ):
                found = index
                break
        return found

    def check(self, exe: str, args: str) -> int | None:
        """`match`, counting a hit."""
        index = self.match(exe, args)
        if index is not None:
            self.hits[index] += 1
            self.skipped += 1
        return index

    def hit_counts(self, limit: int = 20) -> dict[str, int]:
        pairs = sorted(
            ((n, rule.text) for rule, n in zip(self.rules, self.hits, strict=True) if n),
            reverse=True,
        )
        return {text: n for n, text in pairs[:limit]}


SCOPES = ("exact", "prefix", "exe")


def build_rule(exe: str, argv_text: str, scope: str, k: int = 1) -> str:
    """The rule text for an observed launch (`exe` path, space-joined argv incl. argv[0]).

    - `exact`: this exe and exactly these arguments.
    - `prefix`: this exe, the first `k` arguments (1 <= k <= number of arguments) and anything after.
    - `exe`: this exe with any arguments.

    The exe is always the full path (strong form). Raises `RuleError` when the launch cannot be
    expressed: no absolute exe path, or whitespace in it.
    """
    if not exe.startswith("/"):
        raise RuleError(f"exe {exe!r} is not an absolute path")
    if any(ch.isspace() for ch in exe):
        raise RuleError(f"exe {exe!r} contains whitespace")
    words = args_of(argv_text).split()
    head = escape(exe)
    if scope == "exact":
        rule = " ".join([head, *(escape(w) for w in words)])
    elif scope == "prefix":
        if not words:
            raise RuleError("no arguments to take a prefix of")
        count = min(max(k, 1), len(words))
        rule = " ".join([head, *(escape(w) for w in words[:count]), "*"])
    elif scope == "exe":
        rule = f"{head} *"
    else:
        raise RuleError(f"unknown scope {scope!r}")
    parse_rule(rule)
    return rule


def args_of(argv_text: str) -> str:
    """argv[1:] of a space-joined argv string."""
    return " ".join(argv_text.split()[1:])

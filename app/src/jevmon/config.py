"""Load configuration: shipped defaults in `default.toml`, overridden by an optional user file.

Schema (unknown keys are rejected so a typo fails at startup):

- `[exec]`: source knobs (which source, helper socket, interval, CPU budget, bounds, summary cadence).
- `[context]`: how the model-facing state is built (size, windows, smoothing).
- `[judge]`: which judge runs, the model's question, category descriptions, category severities and
  probability bands. This is prompt content and model calibration, not rules about processes.
- `[filter]`: owner-written allow rules (empty by default): matching launches are not judged.
- `[store]`: retention, queue capacity and whether to persist exec events.

The user file defaults to `$XDG_CONFIG_HOME/jevmon/config.toml`, else `~/.config/jevmon/config.toml`.
"""

import os
import tempfile
import tomllib
from dataclasses import dataclass, field, fields, replace
from importlib import resources
from pathlib import Path

import tomlkit

from jevmon.allow import FilterSettings, RuleError, compile_rules, parse_rule
from jevmon.detect.context import ContextSettings
from jevmon.detect.judge import JudgeSettings
from jevmon.sources.exec import ExecSettings
from jevmon.sources.facts import FACT_NAMES


class ConfigError(ValueError):
    """A configuration file is malformed, has an unknown key or a wrong value type."""


@dataclass(frozen=True, slots=True)
class StoreSettings:
    retention_days: int = 14
    queue_capacity: int = 10_000
    store_exec_events: bool = False
    # ebpf source: a memoised verdict repeating a situation stored in the last N seconds is
    # counted (`coalesced`), not stored again; 0 stores every one. The poller always stores all.
    coalesce_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class Config:
    exec: ExecSettings = field(default_factory=ExecSettings)
    context: ContextSettings = field(default_factory=ContextSettings)
    judge: JudgeSettings = field(default_factory=JudgeSettings)
    store: StoreSettings = field(default_factory=StoreSettings)
    filter: FilterSettings = field(default_factory=FilterSettings)


_SECTIONS = {"exec", "context", "judge", "store", "filter"}
_FIELD_TYPES = {
    "exec": {f.name: f.type for f in fields(ExecSettings)},
    "context": {f.name: f.type for f in fields(ContextSettings)},
    "judge": {f.name: f.type for f in fields(JudgeSettings)},
    "store": {f.name: f.type for f in fields(StoreSettings)},
    "filter": {f.name: f.type for f in fields(FilterSettings)},
}
_DEFAULTS = {
    "exec": ExecSettings,
    "context": ContextSettings,
    "judge": JudgeSettings,
    "store": StoreSettings,
}


def user_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "jevmon" / "config.toml"


def _coerce(key: str, value: object, annotation: object) -> object:
    """Validate/convert a TOML value against a dataclass field annotation."""
    origin = getattr(annotation, "__origin__", None)
    try:
        if origin is dict:
            if not isinstance(value, dict):
                raise TypeError("expected a table")
            arg = getattr(annotation, "__args__", (str, str))[1]
            if arg is int:
                return {str(k): int(v) for k, v in value.items()}
            return {str(k): str(v) for k, v in value.items()}
        if origin is tuple:
            if not isinstance(value, list):
                raise TypeError("expected a list")
            return tuple(str(item) for item in value)
        if annotation is bool:
            if not isinstance(value, bool):
                raise TypeError("expected a boolean")
            return value
        if annotation is int:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("expected an integer")
            return value
        if annotation is float:
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError("expected a number")
            return float(value)
        if annotation is str:
            if not isinstance(value, str):
                raise TypeError("expected a string")
            return value
    except TypeError as exc:
        raise ConfigError(f"{key}: {exc}") from exc
    return value


def _build(section: str, table: dict) -> dict:
    types = _FIELD_TYPES[section]
    unknown = set(table) - set(types)
    if unknown:
        raise ConfigError(f"[{section}] unknown key(s): {', '.join(sorted(unknown))}")
    return {key: _coerce(key, value, types[key]) for key, value in table.items()}


def _merge(base: dict, override: dict) -> dict:
    merged = {section: dict(values) for section, values in base.items()}
    for section, values in override.items():
        merged.setdefault(section, {}).update(values)
    return merged


def _from_table(table: dict[str, dict]) -> Config:
    unknown_sections = set(table) - _SECTIONS
    if unknown_sections:
        raise ConfigError(f"unknown section(s): {', '.join(sorted(unknown_sections))}")
    try:
        exec_section = replace(ExecSettings(), **_build("exec", table.get("exec", {})))
    except ValueError as exc:  # value validation in ExecSettings.__post_init__
        raise ConfigError(f"[exec] {exc}") from exc
    context = replace(ContextSettings(), **_build("context", table.get("context", {})))
    missing = [n for n in context.order if n in FACT_NAMES and n not in exec_section.facts]
    if missing:
        raise ConfigError(
            f"[context] order names fact(s) not enabled in [exec] facts: {', '.join(missing)}"
        )
    try:
        judge = replace(JudgeSettings(), **_build("judge", table.get("judge", {})))
    except (ValueError, TypeError) as exc:  # validation in JudgeSettings.__post_init__
        raise ConfigError(f"[judge] {exc}") from exc
    raw_filter = table.get("filter", {})
    allow = raw_filter.get("allow", [])
    if not isinstance(allow, list) or not all(isinstance(rule, str) for rule in allow):
        raise ConfigError("[filter] allow must be a list of strings")
    try:
        filter_settings = replace(FilterSettings(), **_build("filter", raw_filter))
    except RuleError as exc:
        raise ConfigError(f"[filter] allow: {exc}") from exc
    return Config(
        exec=exec_section,
        context=context,
        judge=judge,
        store=replace(StoreSettings(), **_build("store", table.get("store", {}))),
        filter=filter_settings,
    )


def _read(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def load_config(path: str | Path | None = None) -> Config:
    """Load shipped defaults, then overlay `path` (or the user config if `path` is None)."""
    default_text = resources.files("jevmon").joinpath("default.toml").read_text(encoding="utf-8")
    table = tomllib.loads(default_text)
    override_path = Path(path) if path is not None else user_config_path()
    if path is not None and not override_path.exists():
        raise ConfigError(f"config file not found: {override_path}")
    override = _read(override_path)
    if override:
        table = _merge(table, override)
    return _from_table(table)


def default_allow() -> tuple[str, ...]:
    """The shipped `[filter] allow` list (what a config without the key inherits)."""
    return _from_table(_default_table()).filter.allow


def _default_table() -> dict:
    text = resources.files("jevmon").joinpath("default.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)


def edit_allow(
    path: str | Path | None, *, add: str | None = None, remove: str | None = None
) -> tuple[list[str], bool]:
    """Add or remove one rule in `[filter] allow` of the user config file; return (list, changed).

    The file is edited in place with tomlkit, so comments and ordering survive; it is created when
    missing, and `[filter]` is added when absent. A list in the file replaces the shipped one, so
    when the key is absent the shipped list is copied in first. The write is atomic (temp file and
    rename). A rule that does not parse, or a file that does not, raises (`RuleError`,
    `ConfigError`) before anything is written.
    """
    target = Path(path) if path is not None else user_config_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    except OSError as exc:
        raise ConfigError(f"{target}: {exc}") from exc
    try:
        doc = tomlkit.parse(text)
    except tomlkit.exceptions.TOMLKitError as exc:
        raise ConfigError(f"{target}: {exc}") from exc
    if add is not None:
        add = add.strip()
        parse_rule(add)
    section = doc.get("filter")
    if section is None:
        section = tomlkit.table()
        doc["filter"] = section
    allow = section.get("allow")
    if allow is None:
        allow = tomlkit.array()
        for rule in default_allow():
            allow.append(rule)
        section["allow"] = allow
    if not isinstance(allow, list) or not all(isinstance(r, str) for r in allow):
        raise ConfigError("[filter] allow must be a list of strings")
    changed = False
    if add is not None and add not in allow:
        allow.append(add)
        changed = True
    if remove is not None and remove in allow:
        del allow[[str(r) for r in allow].index(remove)]
        changed = True
    if len(allow) > 1:
        allow.multiline(True)
    if changed:
        compile_rules([str(r) for r in allow])
        _write_atomic(target, tomlkit.dumps(doc))
    return [str(r) for r in allow], changed


def _write_atomic(target: Path, text: str) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = target.stat().st_mode & 0o777 if target.exists() else 0o600
        fd, tmp = tempfile.mkstemp(prefix=".config-", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.chmod(tmp, mode)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise ConfigError(f"cannot write {target}: {exc}") from exc

"""Local-only experiment tracking for the lab scripts, plus an always-on JSONL mirror.

`RunLog` wraps Trackio (Hugging Face's lightweight tracker: a local SQLite store plus a
dashboard) behind four calls - `start`, `log`, `table`, `finish` - so the training and eval
scripts can report live progress.

Two independent sinks, so a run is observable either way:

- Trackio, an OPTIONAL lab extra (`uv run --project lab --extra trackio ...`). When it is not
  importable every Trackio call is a silent no-op and the run continues.
- A JSONL file (`jsonl=<path>`), always written if a path is given, with the same records, so
  `<out>/metrics.jsonl` is a complete log even without the extra.

Nothing leaves the machine: before Trackio is imported this module points its store at a local
directory (`TRACKIO_DIR`) and clears every setting that would sync a run (Space, server URL,
bucket, dataset, webhook, share link, telemetry). Trackio itself has no telemetry; the Gradio
and Hugging Face switches are set anyway as a belt-and-braces measure.
"""

import importlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from jevmon_lab.manifest import LABELS

DEFAULT_DIR = Path("runs")

# Anything that would send a run (or a dashboard) off the machine.
_CLEARED_ENV = (
    "TRACKIO_SPACE_ID",
    "TRACKIO_SERVER_URL",
    "TRACKIO_BUCKET_ID",
    "TRACKIO_DATASET_ID",
    "TRACKIO_WRITE_TOKEN",
    "TRACKIO_WEBHOOK_URL",
    "TRACKIO_REGISTRY_BUCKET_ID",
    "SPACE_ID",
)
_FORCED_ENV = {
    "GRADIO_SHARE": "false",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRACKIO_STORAGE_MODE": "sqlite",
}


class Backend(Protocol):
    """The slice of a tracker `RunLog` needs; Trackio satisfies it through `_TrackioBackend`."""

    def init(self, project: str, name: str, config: dict[str, Any]) -> None: ...
    def log(self, metrics: dict[str, Any], step: int) -> None: ...
    def table(self, rows: list[dict[str, Any]]) -> Any: ...
    def finish(self) -> None: ...


def local_env(directory: Path) -> None:
    """Point Trackio at `directory` and switch off every sync/share/telemetry path."""
    for key in _CLEARED_ENV:
        os.environ.pop(key, None)
    os.environ.update(_FORCED_ENV)
    os.environ["TRACKIO_DIR"] = str(directory.resolve())


class _TrackioBackend:
    """Trackio's module-level API; `TRACKIO_DIR` is read at import, hence `local_env` first."""

    def __init__(self, module: Any) -> None:
        self._t = module
        self._run: Any = None

    def init(self, project: str, name: str, config: dict[str, Any]) -> None:
        self._run = self._t.init(
            project=project,
            name=name,
            config=config,
            space_id=None,
            server_url=None,
            dataset_id=None,
            bucket_id=None,
            auto_log_gpu=False,
            auto_log_cpu=False,
            webhook_url=None,
        )

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self._run.log(metrics, step=step)

    def table(self, rows: list[dict[str, Any]]) -> Any:
        return self._t.Table(data=rows)

    def finish(self) -> None:
        self._run.finish()


def _load_trackio(directory: Path) -> Backend | None:
    local_env(directory)
    try:
        module = importlib.import_module("trackio")
    except ImportError:
        return None
    return _TrackioBackend(module)


def _plain(value: Any) -> Any:
    """JSON-friendly copy of a config value (paths, tuples and the like become str/list)."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


class RunLog:
    """One tracked run. Build it with `RunLog.start`; every method is safe to call blind.

    `enabled` reports whether Trackio is recording; the JSONL mirror is written whenever
    `jsonl` was passed, independently of Trackio.
    """

    def __init__(
        self,
        backend: Backend | None,
        jsonl: Path | None = None,
        *,
        project: str = "",
        name: str = "",
    ) -> None:
        self._backend = backend
        self._project = project
        self._name = name
        self._jsonl = None
        if jsonl is not None:
            jsonl = Path(jsonl)
            jsonl.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl = jsonl.open("a", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._backend is not None

    @property
    def recording(self) -> bool:
        """True when either sink is live (Trackio or the JSONL mirror)."""
        return self._backend is not None or self._jsonl is not None

    @classmethod
    def start(
        cls,
        project: str,
        name: str,
        config: Mapping[str, Any] | None = None,
        *,
        directory: Path = DEFAULT_DIR,
        jsonl: Path | None = None,
        backend: Backend | None = None,
        enabled: bool = True,
    ) -> "RunLog":
        """Begin a run under `directory/`; disabled (or without trackio) it is a no-op.

        `backend` replaces Trackio (tests). A tracker that fails to start never stops the
        experiment: the failure is reported on stderr and Trackio is off for the run. The
        JSONL mirror, if `jsonl` is given, is written regardless.
        """
        plain_config = _plain(dict(config or {}))
        if enabled and backend is None:
            directory.mkdir(parents=True, exist_ok=True)
            backend = _load_trackio(directory)
            if backend is None:
                print(
                    "runlog: trackio not installed, tracking off (JSONL mirror only)",
                    file=sys.stderr,
                )
        run = cls(backend if enabled else None, jsonl, project=project, name=name)
        if run._backend is not None:
            try:
                run._backend.init(project, name, plain_config)
            except Exception as exc:  # noqa: BLE001 - a broken tracker must not stop the run
                print(f"runlog: tracking off ({exc})", file=sys.stderr)
                run._backend = None
        run._emit({"kind": "config", "config": plain_config})
        return run

    def log(self, step: int, **scalars: float | str | bool) -> None:
        """Record `scalars` at `step`; `/` in a name groups plots in the dashboard."""
        if not scalars:
            return
        self._emit({"kind": "scalar", "step": int(step), "metrics": scalars})
        if self._backend is not None:
            self._guard(lambda b: b.log(dict(scalars), int(step)))

    def table(self, name: str, rows: list[dict[str, Any]], step: int = 0) -> None:
        """Record `rows` (dicts sharing keys) as a table named `name`."""
        if not rows:
            return
        self._emit({"kind": "table", "step": int(step), "name": name, "rows": rows})
        if self._backend is not None:
            self._guard(lambda b: b.log({name: b.table(_plain(rows))}, int(step)))

    def finish(self) -> None:
        self._emit({"kind": "finish"})
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None
        if self._backend is not None:
            self._guard(lambda b: b.finish())
            self._backend = None

    def _emit(self, record: dict[str, Any]) -> None:
        if self._jsonl is None:
            return
        payload = {"project": self._project, "run": self._name, **_plain(record)}
        self._jsonl.write(json.dumps(payload) + "\n")
        self._jsonl.flush()

    def _guard(self, call) -> None:
        assert self._backend is not None
        try:
            call(self._backend)
        except Exception as exc:  # noqa: BLE001 - tracking must never take the experiment down
            print(f"runlog: tracking off ({exc})", file=sys.stderr)
            self._backend = None


def score_scalars(prefix: str, score: Mapping[str, Any]) -> dict[str, float]:
    """Flatten a `metrics.score` result into scalars named `<prefix>/...`.

    The always-safe macro-F1/accuracy and the majority-class rate are included so they
    sit next to the model's number in every plot, with per-class precision and recall.
    """
    out = {
        f"{prefix}/macro_f1": float(score["macro_f1"]),
        f"{prefix}/accuracy": float(score["accuracy"]),
        f"{prefix}/always_safe_macro_f1": float(score["always_safe_macro_f1"]),
        f"{prefix}/always_safe_accuracy": float(score["always_safe_accuracy"]),
        f"{prefix}/majority_rate": float(score["majority_rate"]),
    }
    for label, cls in score["per_class"].items():
        out[f"{prefix}/precision/{label}"] = float(cls["precision"])
        out[f"{prefix}/recall/{label}"] = float(cls["recall"])
        out[f"{prefix}/f1/{label}"] = float(cls["f1"])
    return out


def class_table(score: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-class precision/recall/F1/support of one score, as table rows."""
    return [
        {
            "label": label,
            "precision": round(cls["precision"], 4),
            "recall": round(cls["recall"], 4),
            "f1": round(cls["f1"], 4),
            "support": cls["support"],
            "predicted": cls["predicted"],
        }
        for label, cls in score["per_class"].items()
    ]


def confusion_table(score: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rows of one score's confusion matrix, with an `n` column per true label."""
    confusion = score.get("confusion")
    if confusion is None:
        return []
    return [
        {"true": label, **{f"pred_{lab}": int(confusion[i][j]) for j, lab in enumerate(LABELS)}}
        for i, label in enumerate(LABELS)
    ]

"""Local-only experiment tracking for the lab scripts, a no-op without `trackio`.

`RunLog` wraps Trackio (Hugging Face's lightweight tracker: a local SQLite store plus a
dashboard) behind four calls - `start`, `log`, `table`, `finish` - so the training and eval
scripts can report live progress. Trackio is a dependency of the lab project only; when it is
not importable every call is a silent no-op, so the app, its tests and the daemon never need it.

Nothing leaves the machine: before Trackio is imported this module points its store at a
local directory and clears every setting that would sync a run (Space, server URL, bucket,
dataset, webhook, share link, telemetry). Trackio itself has no telemetry; the Gradio and
Hugging Face switches are set anyway as a belt-and-braces measure.
"""

import importlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

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
    """One tracked run. Build it with `RunLog.start`; every method is safe to call blind."""

    def __init__(self, backend: Backend | None) -> None:
        self._backend = backend

    @property
    def enabled(self) -> bool:
        return self._backend is not None

    @classmethod
    def start(
        cls,
        project: str,
        name: str,
        config: Mapping[str, Any] | None = None,
        *,
        directory: Path = DEFAULT_DIR,
        backend: Backend | None = None,
        enabled: bool = True,
    ) -> "RunLog":
        """Begin a run under `directory/`; disabled (or without trackio) it is a no-op.

        `backend` replaces Trackio (tests). A tracker that fails to start never stops the
        experiment: the failure is reported on stderr and tracking is off for the run.
        """
        if not enabled:
            return cls(None)
        if backend is None:
            directory.mkdir(parents=True, exist_ok=True)
            backend = _load_trackio(directory)
            if backend is None:
                print("runlog: trackio not installed, tracking off", file=sys.stderr)
                return cls(None)
        try:
            backend.init(project, name, _plain(dict(config or {})))
        except Exception as exc:  # noqa: BLE001 - a broken tracker must not stop the run
            print(f"runlog: tracking off ({exc})", file=sys.stderr)
            return cls(None)
        return cls(backend)

    def log(self, step: int, **scalars: float | str | bool) -> None:
        """Record `scalars` at `step`; `/` in a name groups plots in the dashboard."""
        if self._backend is not None and scalars:
            self._guard(lambda b: b.log(dict(scalars), step))

    def table(self, name: str, rows: list[dict[str, Any]], step: int = 0) -> None:
        """Record `rows` (dicts sharing keys) as a table named `name`."""
        if self._backend is not None and rows:
            self._guard(lambda b: b.log({name: b.table(_plain(rows))}, step))

    def finish(self) -> None:
        if self._backend is not None:
            self._guard(lambda b: b.finish())
            self._backend = None

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

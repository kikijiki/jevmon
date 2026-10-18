"""Command line: `jevmon exec` runs the /proc exec source, builds model context and judges it.

`jevmon exec` polls `/proc`, turns each exec into a context, and sends contexts to the Laya
decision model (the only judge). Verdicts of class crit print as JSON lines on stdout; every
verdict is stored (when a database is configured) as training data. `--once` prints a single scan
summary. `jevmon label` / `jevmon ack` attach human labels to stored events directly.

`jevmon ctl <cmd>` is a thin JSON client that talks only to a running `jevmond` socket.
SIGINT/SIGTERM stop `exec` cleanly.
"""

import argparse
import signal
import sys
import threading
from dataclasses import replace
from pathlib import Path

from jevmon.config import Config, ConfigError, load_config
from jevmon.core import SqliteStore
from jevmon.events import Event
from jevmon.monitor import Monitor
from jevmon.paths import state_dir
from jevmon.sources.exec import KIND_EXEC, KIND_SUMMARY, ExecSettings, ExecSource
from jevmon.sources.proc import RealProcReader
from jevmon.sources.replay import ReplayError, load_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jevmon", description="resource and process monitor")
    sub = parser.add_subparsers(dest="command", required=True)

    exec_p = sub.add_parser("exec", help="poll /proc for process exec events and judge them")
    exec_p.add_argument("--once", action="store_true", help="one scan, print the summary, exit")
    exec_p.add_argument("--config", help="path to a config.toml (default: XDG config)")
    exec_p.add_argument("--db", help="SQLite path (default: state dir in continuous mode)")
    exec_p.add_argument("--proc-root", help="alternate /proc tree (for tests)")
    exec_p.add_argument("--verbose", action="store_true", help="also print info events on stdout")

    label_p = sub.add_parser("label", help="attach a human label to a stored event")
    label_p.add_argument("event_id", type=int)
    label_p.add_argument("label")
    label_p.add_argument("--source", default="human")
    label_p.add_argument("--config")
    label_p.add_argument("--db")

    ack_p = sub.add_parser("ack", help="mark a stored event acknowledged (label 'ack')")
    ack_p.add_argument("event_id", type=int)
    ack_p.add_argument("--config")
    ack_p.add_argument("--db")

    ctl_p = sub.add_parser("ctl", help="talk to a running jevmond without the TUI")
    ctl_p.add_argument("--socket", help="daemon socket (default: XDG runtime dir)")
    ctl_sub = ctl_p.add_subparsers(dest="ctl_command", required=True)
    ctl_sub.add_parser("status", help="daemon status as JSON")
    events_p = ctl_sub.add_parser("events", help="query stored events as JSON")
    events_p.add_argument("--since-id", type=int, default=0)
    events_p.add_argument("--min-severity", type=int, default=0)
    events_p.add_argument("--limit", type=int, default=50)
    events_p.add_argument("--unlabeled", action="store_true")
    events_p.add_argument("--text")
    label_ctl = ctl_sub.add_parser("label", help="label an event")
    label_ctl.add_argument("event_id", type=int)
    label_ctl.add_argument("label")
    ack_ctl = ctl_sub.add_parser("ack", help="acknowledge an event")
    ack_ctl.add_argument("event_id", type=int)
    ctl_sub.add_parser("pause", help="pause judging, keep the poller and census running")
    ctl_sub.add_parser("resume", help="resume judging")
    allow_ctl = ctl_sub.add_parser("allow", help="show or edit the [filter] allow rules")
    allow_sub = allow_ctl.add_subparsers(dest="allow_command", required=True)
    allow_sub.add_parser("list", help="the active rules and their hit counts")
    allow_add = allow_sub.add_parser("add", help="append a rule to the config file, live")
    allow_add.add_argument("rule", help='e.g. "/usr/bin/git rev-parse *"')
    allow_rm = allow_sub.add_parser("remove", help="remove a rule from the config file, live")
    allow_rm.add_argument("rule")
    ctl_sub.add_parser("shutdown", help="ask the daemon to stop")

    for name, text in (
        ("start", "start the daemon; for source = ebpf also the root helper (one prompt)"),
        ("stop", "stop the daemon, and the helper if `jevmon start` started it"),
        ("status", "show the configured source, the helper and the daemon (no privileges)"),
    ):
        service_p = sub.add_parser(name, help=text)
        service_p.add_argument("--config", help="path to a config.toml (default: XDG config)")

    helper_p = sub.add_parser("helper", help="advanced: start/stop only the root helper")
    helper_p.add_argument("action", choices=("start", "stop"))
    helper_p.add_argument("--config", help="path to a config.toml (default: XDG config)")

    return parser


def _run_service(args: argparse.Namespace, config: Config) -> int:
    import os

    from jevmon import service

    try:
        if args.command == "status":
            for line in service.status_lines(config, _daemon_status()):
                sys.stdout.write(line + "\n")
            return 0
        if args.command == "stop" or (args.command == "helper" and args.action == "stop"):
            kind, active = service.unit_state()
            plan = service.plan_stop(unit_kind=kind, unit_active=active)
            if args.command == "helper":
                plan.steps = [step for step in plan.steps if step.elevated]
            return service.execute(plan)
        answering = service.probe_socket(config.exec.helper_socket)
        plan = service.plan_start(
            replace(config, exec=replace(config.exec, source="ebpf"))
            if args.command == "helper"
            else config,
            helper_answering=answering,
            uid=os.getuid(),
            helper_finder=lambda cfg: service.find_helper(cfg, service.repo_root()),
        )
        if args.command == "helper":
            plan.steps = [step for step in plan.steps if step.elevated]
        socket_path = config.exec.helper_socket
        return service.execute(
            plan, after={"start helper": lambda: service.wait_for_helper(socket_path)}
        )
    except service.ServiceError as exc:
        sys.stderr.write(f"jevmon: {exc}\n")
        return 1


def _daemon_status() -> dict | None:
    import asyncio

    from jevmon.daemon.client import DaemonClient

    async def ask() -> dict:
        async with DaemonClient() as client:
            return await client.status()

    try:
        return asyncio.run(ask())
    except Exception:  # noqa: BLE001 - no daemon, a stale socket: status says "not running"
        return None


def _settings(config: Config, proc_root: str | None) -> ExecSettings:
    return replace(config.exec, proc_root=proc_root) if proc_root else config.exec


def _open_store(
    args: argparse.Namespace, config: Config, *, required: bool = False
) -> SqliteStore | None:
    if getattr(args, "db", None):
        path: str | Path = args.db
    elif getattr(args, "once", False) and not required:
        return None  # a one-shot run must not create state unless asked
    else:
        path = state_dir() / "events.db"
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(path, retention_days=config.store.retention_days)
    store.prune()
    return store


def _run_once(config: Config, settings: ExecSettings, store: SqliteStore | None) -> int:
    if settings.source == "replay":
        recording: list[Event] = load_events(settings.replay_file)
        execs = [event for event in recording if event.kind == KIND_EXEC]
        summaries = [event for event in recording if event.kind == KIND_SUMMARY]
    else:
        result = ExecSource(RealProcReader(settings.proc_root), settings).scan_once()
        execs = list(result.events)
        summaries = [] if result.summary is None else [result.summary]
    if store is not None and config.store.store_exec_events:
        for event in execs:
            store.add_event(event)
    for summary in summaries:
        sys.stdout.write(summary.to_json() + "\n")
    return 0


def _run_loop(
    config: Config, settings: ExecSettings, store: SqliteStore | None, verbose: bool
) -> int:
    stop = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        stop.set()

    # Install handlers before any setup so a signal during model load or startup still stops us.
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        monitor = Monitor(
            config,
            store,
            proc_root=settings.proc_root if settings.proc_root != config.exec.proc_root else None,
            summary_stream=sys.stdout,
            verbose=verbose,
        )
        monitor.start()
        sys.stderr.write(f"jevmon: {monitor.judge.status}\n")
        while monitor.running and not stop.wait(0.25):  # a replay ends on its own
            pass
        monitor.stop()
        sys.stderr.write(f"jevmon: {monitor.judge.status}\n")
    except ReplayError as exc:
        sys.stderr.write(f"jevmon: {exc}\n")
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


def _run_label(args: argparse.Namespace, config: Config) -> int:
    store = _open_store(args, config, required=True)
    if store is None:
        return 2
    label = "ack" if args.command == "ack" else args.label
    source = "human" if args.command == "ack" else args.source
    store.add_label(args.event_id, label, source)
    store.close()
    sys.stderr.write(f"jevmon: labelled event {args.event_id} as {label!r}\n")
    return 0


def _run_ctl(args: argparse.Namespace) -> int:
    from jevmon.daemon.client import run_ctl

    return run_ctl(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ctl":
        return _run_ctl(args)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        sys.stderr.write(f"jevmon: {exc}\n")
        return 2
    if args.command in {"start", "stop", "status", "helper"}:
        return _run_service(args, config)
    if args.command in {"label", "ack"}:
        return _run_label(args, config)
    if args.command != "exec":
        return 2
    settings = _settings(config, args.proc_root)
    store = _open_store(args, config)
    try:
        if args.once:
            return _run_once(config, settings, store)
        return _run_loop(config, settings, store, bool(args.verbose))
    finally:
        if args.once and store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())

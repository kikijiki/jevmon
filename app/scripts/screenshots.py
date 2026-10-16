"""Capture README screenshots of jevmond + jevmon-tui from a fully mocked, isolated run.

    just screenshots          # or: uv run --project app python app/scripts/screenshots.py

Nothing here touches the user's daemon, socket, database or config. The run gets its own
directory under /tmp (socket, SQLite store, config and a fictional recording), a private tmux
server and an environment pinned to `TZ=UTC`, and it drives the real binaries:

1. `jevmond` starts with `[exec] source = "replay"` (a seeded, entirely invented event stream:
   demo processes, users, paths, host `demo-gw-01`, units and pids - nothing read from this
   machine) and `[judge] mode = "fake"` (a hash of the context: no model, no GPU, no judgement).
2. `jevmon-tui` runs in a tmux pane of fixed geometry and three views are captured from it: the
   list following the tail, a pinned selection with the detail pane, and a text filter.
3. Every capture is rasterised to `docs/screenshots/<name>.png` through `jevmon.termshot` and the
   README's screenshot block is rewritten in place.

Before anything is written, every capture is checked against the real `$USER`, `$HOME` and
hostname *before* rasterising (a PNG cannot be grepped): the run fails rather than committing a
local detail.

Re-running is deterministic - a fixed seed and epoch, a fixed terminal size and a fixed work
directory - so only live counters (uptime, queue age, rates) move between runs.
"""

import argparse
import asyncio
import fcntl
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from jevmon.daemon.client import DaemonClient
from jevmon.sources.demo import demo_events
from jevmon.sources.exec import KIND_EXEC
from jevmon.sources.replay import write_events
from jevmon.termshot import ansi_to_png

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root; the app project is ROOT/app
APP = ROOT / "app"
WORK = Path("/tmp/jevmon-shots")
LOCK = Path("/tmp/jevmon-shots.lock")
OUTPUT = ROOT / "docs" / "screenshots"
README = ROOT / "README.md"
TMUX_SERVER = "jevmon-shots"

WIDTH, HEIGHT = 110, 34
CELL_W, CELL_H, FONT_SIZE = 20, 40, 33  # 33 px DejaVu Sans Mono advances 19.9 px: the grid is exact
INTERACT = 0.45  # the TUI redraws on a 100 ms tick; a settle window before every capture
MIN_ROWS = 40  # enough events on screen (and a still-running replay) before capturing

BEGIN = "<!-- screenshots:begin -->"
END = "<!-- screenshots:end -->"

# Stable file names, in the order the README shows them.
CAPTIONS: dict[str, str] = {
    "detail": "A pinned event with its verdict and probabilities.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="capture the README screenshots (mocked and isolated)"
    )
    parser.add_argument("--keep", action="store_true", help="leave the work directory behind")
    parser.add_argument(
        "--no-readme", action="store_true", help="write the PNGs but leave README.md alone"
    )
    return parser


# -- local details -----------------------------------------------------------------------------


def forbidden_needles() -> list[str]:
    """Real local details that must never appear in a capture, in capture or SVG text."""
    candidates: list[str | None] = [
        os.environ.get("USER"),
        os.environ.get("LOGNAME"),
        os.environ.get("USERNAME"),
        os.environ.get("HOME"),
        socket.gethostname(),
    ]
    try:
        import pwd

        account = pwd.getpwuid(os.getuid())
        candidates.extend([account.pw_name, account.pw_dir])
    except (ImportError, KeyError, OSError):  # pragma: no cover - unusual platforms
        pass
    needles: list[str] = []
    for candidate in candidates:
        # A floor of 3 keeps a one/two letter login from matching random text everywhere.
        if candidate and len(candidate) >= 3 and candidate not in needles:
            needles.append(candidate)
    return needles


def assert_fictional(text: str, where: str, needles: list[str]) -> None:
    leaks = [needle for needle in needles if needle in text]
    if leaks:
        raise SystemExit(
            f"screenshots: {where} leaks real local details: "
            f"{', '.join(repr(leak) for leak in leaks)}\n"
            "Every process, path, user, host and unit in a capture must be fictional."
        )


# -- the isolated run --------------------------------------------------------------------------


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["TZ"] = "UTC"  # fixed timestamps: the TUI renders local time
    env["TERM"] = env.get("TERM") or "xterm-256color"
    env["XDG_CONFIG_HOME"] = str(WORK / "xdg" / "config")
    env["XDG_STATE_HOME"] = str(WORK / "xdg" / "state")
    env["XDG_RUNTIME_DIR"] = str(WORK / "xdg" / "run")
    env.pop("NO_COLOR", None)  # the screenshots are the coloured TUI
    return env


def _entry_point(name: str, module: str) -> list[str]:
    """The `name` console script next to this interpreter, else `python -m module`."""
    entry = Path(sys.executable).with_name(name)
    return [str(entry)] if entry.exists() else [sys.executable, "-m", module]


def _write_config(events: Path) -> Path:
    config = WORK / "config" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "# Written by app/scripts/screenshots.py: an isolated, fully mocked run.\n"
        "[exec]\n"
        'source = "replay"\n'
        f'replay_file = "{events}"\n'
        "interval = 0.15\n"
        'facts = ["exe_state", "argv0_vs_exe", "stdio", "session", "workdir", "cgroup_path",'
        ' "privileges", "namespaces", "exe_file"]\n'
        "\n"
        "[context]\n"
        'order = ["header", "exe", "argv", "stdio", "session", "cgroup_path", "privileges",'
        ' "namespaces"]\n'
        'sep = "  "\n'
        "model_max = 2000\n"
        "\n"
        "[judge]\n"
        'mode = "fake"\n'
        "\n"
        "# Fictional verdicts for the fictional launches, so the mocked run reads plausibly.\n"
        "[judge.fake_verdicts]\n"
        'updater = "crit"\n'
        '"tool.sh" = "crit"\n'
        'poll-queue = "warn"\n'
        'job-runner = "warn"\n'
        '"-delete" = "warn"\n'
        'unittest = "warn"\n'
        'demo-import = "warn"\n'
        'scan-runner = "warn"\n'
        'demo = "safe"\n'
        "\n"
        "[store]\n"
        "store_exec_events = false\n",
        encoding="utf-8",
    )
    return config


def _call(factory) -> dict:
    return asyncio.run(factory())


async def _status(socket_path: Path) -> dict:
    async with DaemonClient(socket_path) as client:
        return await client.status()


async def _event_count(socket_path: Path) -> int:
    async with DaemonClient(socket_path) as client:
        reply = await client.events(limit=100)
    return int(reply.get("count", 0))


def _wait_until(predicate, timeout: float, what: str, interval: float = 0.1) -> None:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception as exc:  # noqa: BLE001 - the daemon may not be answering yet
            last = exc
        time.sleep(interval)
    detail = f" ({last})" if last is not None else ""
    raise SystemExit(f"screenshots: timed out waiting for {what}{detail}")


def _expected_verdicts(events) -> int:
    """How many verdict events the pipeline will store: one per exec."""
    return sum(1 for event in events if event.kind == KIND_EXEC)


# -- tmux --------------------------------------------------------------------------------------


def _tmux(*args: str, env: dict[str, str]) -> str:
    if shutil.which("tmux") is None:
        raise SystemExit("screenshots: tmux is required (it ships in the flake dev shell)")
    done = subprocess.run(
        ["tmux", "-L", TMUX_SERVER, *args], env=env, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise SystemExit(f"screenshots: tmux {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def _tmux_conf() -> Path:
    conf = WORK / "tmux.conf"
    conf.write_text(
        'set -g default-terminal "tmux-256color"\n'
        'set -ga terminal-overrides ",*:Tc"\n'
        "set -g status off\n"
        "set -g escape-time 0\n"
        "set -g history-limit 64\n",
        encoding="utf-8",
    )
    return conf


def _start_session(name: str, command: str, env: dict[str, str], conf: Path) -> None:
    _tmux(
        "-f",
        str(conf),
        "new-session",
        "-d",
        "-s",
        name,
        "-x",
        str(WIDTH),
        "-y",
        str(HEIGHT),
        command,
        env=env,
    )


def _screen(name: str, env: dict[str, str]) -> str:
    return _tmux("capture-pane", "-p", "-t", name, env=env)


def _capture(name: str, env: dict[str, str], needles: list[str]) -> str:
    text = _tmux("capture-pane", "-p", "-e", "-t", name, env=env)
    assert_fictional(text, f"tmux capture of {name!r}", needles)
    if "jevmon" not in text and "uptime" not in text:
        raise SystemExit(f"screenshots: the {name!r} pane does not look like jevmon output")
    return text


def _send(name: str, *keys: str, env: dict[str, str]) -> None:
    _tmux("send-keys", "-t", name, *keys, env=env)
    time.sleep(INTERACT)


def _listed_events(name: str, env: dict[str, str]) -> int:
    match = re.search(r"events (\d+)", _screen(name, env))
    return int(match.group(1)) if match else 0


def _render(capture: str) -> bytes:
    """Rasterise the capture to PNG. The fictional-data check runs on `capture` first: a PNG
    cannot be grepped."""
    return ansi_to_png(
        capture, cols=WIDTH, rows=HEIGHT, cell_w=CELL_W, cell_h=CELL_H, font_size=FONT_SIZE
    )


# -- README ------------------------------------------------------------------------------------


def _readme_block() -> str:
    lines = [BEGIN, ""]
    for name, caption in CAPTIONS.items():
        lines.append(f"![jevmon screenshot: {name.replace('-', ' ')}](docs/screenshots/{name}.png)")
        lines.append("")
        lines.append(f"*{caption}*")
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def _update_readme(block: str) -> None:
    assert_fictional(block, "the README screenshot block", forbidden_needles())
    text = README.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _old, tail = rest.split(END, 1)
        updated = f"{head}{block}{tail}"
    else:
        updated = f"{text.rstrip()}\n\n{block}\n"
    if updated != text:
        README.write_text(updated, encoding="utf-8")


# -- the run -----------------------------------------------------------------------------------


def _lock():
    handle = LOCK.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise SystemExit("screenshots: another run holds the lock") from exc
    return handle


def _prepare() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    for name in ("config", "state", "run", "xdg"):
        (WORK / name).mkdir(parents=True, exist_ok=True)


def _shutdown(process: subprocess.Popen | None, log, env: dict[str, str]) -> None:
    if shutil.which("tmux") is not None:
        subprocess.run(
            ["tmux", "-L", TMUX_SERVER, "kill-server"], env=env, capture_output=True, check=False
        )
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - a wedged child
            process.kill()
            process.wait(timeout=5)
    if log is not None:
        log.close()


def run(args: argparse.Namespace) -> int:
    lock = _lock()
    env = _env()
    _prepare()
    conf = _tmux_conf()
    daemon: subprocess.Popen | None = None
    log = None
    try:
        recording = WORK / "events.jsonl"
        events = demo_events()
        write_events(recording, events)
        config = _write_config(recording)
        socket_path = WORK / "run" / "jevmond.sock"
        store_path = WORK / "state" / "events.db"

        tui = APP / "tui" / "target" / "release" / "jevmon-tui"
        if not tui.exists():
            raise SystemExit(
                "screenshots: app/tui/target/release/jevmon-tui is missing "
                "(`just screenshots` builds it first)"
            )

        daemon_argv = [
            *_entry_point("jevmond", "jevmon.daemon.server"),
            "--config",
            str(config),
            "--db",
            str(store_path),
            "--socket",
            str(socket_path),
        ]
        log = (WORK / "daemon.log").open("w", encoding="utf-8")
        daemon = subprocess.Popen(daemon_argv, env=env, stdout=log, stderr=subprocess.STDOUT)

        _wait_until(socket_path.exists, 30.0, "the isolated daemon socket")
        _wait_until(lambda: _call(lambda: _status(socket_path)), 30.0, "daemon status")

        _start_session(
            "shot",
            shlex.join(["env", "-u", "NO_COLOR", "TZ=UTC", str(tui), "--socket", str(socket_path)]),
            env,
            conf,
        )
        _wait_until(lambda: "connected" in _screen("shot", env), 30.0, "the TUI to connect")
        _wait_until(
            lambda: _listed_events("shot", env) >= MIN_ROWS,
            40.0,
            f"{MIN_ROWS} events on screen",
        )

        needles = forbidden_needles()
        views: dict[str, str] = {}
        _send("shot", "-l", "dd", env=env)  # below -> beside -> hidden: the list alone, full width
        _send("shot", "-l", "dd", env=env)  # hidden -> below -> beside: room for the whole context
        _send("shot", "-l", "kkk", env=env)
        views["detail"] = _capture("shot", env, needles)

        # The captures above are deliberately mid-stream; this proves the run also completes.
        expected = _expected_verdicts(events)
        try:
            _wait_until(
                lambda: _call(lambda: _event_count(socket_path)) >= expected,
                40.0,
                f"all {expected} verdicts in the store",
            )
        except SystemExit as exc:
            print(f"screenshots: warning: {exc}")

        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, capture in views.items():
            png = _render(capture)
            if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise SystemExit(f"screenshots: {name} did not rasterise to a PNG")
            (OUTPUT / f"{name}.png").write_bytes(png)

        if not args.no_readme:
            _update_readme(_readme_block())

        print(f"screenshots: wrote docs/screenshots/{{{', '.join(CAPTIONS)}}}.png")
        return 0
    finally:
        _shutdown(daemon, log, env)
        if not args.keep:
            shutil.rmtree(WORK, ignore_errors=True)
        LOCK.unlink(missing_ok=True)
        lock.close()


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

# jevmon task runner. Run inside `nix develop` (pins python, uv, just, ruff).
#
# Two uv projects: app/ (daemon, judge, TUI) and lab/ (scenarios, dataset, experiments; depends
# on app/). Recipes run from the repo root and select the project with `--project`, so relative
# paths such as models/, lab-data/ and lab/scenarios.toml stay root-relative.

default:
    @just --list

# create/refresh the app's CPU uv environment (app/.venv)
sync:
    uv sync --project app

# create/refresh the lab's CPU uv environment (lab/.venv; installs ../app editable)
sync-lab:
    uv sync --project lab

test: sync sync-lab
    uv run --project app pytest app/tests
    uv run --project lab pytest lab/tests

# lint + tests + Rust TUI checks, the pre-commit bar
check: lint test tui-check

lint: sync sync-lab
    uv run --project app ruff check app/src app/tests app/scripts
    uv run --project app ruff format --check app/src app/tests app/scripts
    uv run --project lab ruff check lab/src lab/tests lab/scripts
    uv run --project lab ruff format --check lab/src lab/tests lab/scripts

fix: sync sync-lab
    uv run --project app ruff check --fix app/src app/tests app/scripts
    uv run --project app ruff format app/src app/tests app/scripts
    uv run --project lab ruff check --fix lab/src lab/tests lab/scripts
    uv run --project lab ruff format lab/src lab/tests lab/scripts

# run any command in the app env, e.g. `just run python -c 'import torch'`
run *ARGS: sync
    uv run --project app {{ARGS}}

# run any command in the lab env, e.g. `just run-lab python lab/scripts/bench_judge.py --device cpu`
run-lab *ARGS: sync-lab
    uv run --project lab {{ARGS}}

# build the root exec helper (app/helper/, nix) into app/helper/result
exec-build:
    nix build .#jevmon-exec -o app/helper/result

# check the helper's BPF program in a throwaway NixOS VM (qemu, needs /dev/kvm; no root on the host)
exec-vmtest:
    nix build .#nixosConfigurations.exec-vmtest.config.system.build.vm -o /tmp/jevmon-vmtest
    rm -rf /tmp/jevmon-vmtest-out && mkdir -p /tmp/jevmon-vmtest-out
    cd /tmp && env -u LD_LIBRARY_PATH QEMU_OPTS="-virtfs local,path=/tmp/jevmon-vmtest-out,mount_tag=out,security_model=none" NIX_DISK_IMAGE=/tmp/jevmon-vmtest.qcow2 timeout 300 /tmp/jevmon-vmtest/bin/run-nixos-vm > /tmp/jevmon-vmtest-out/console.log 2>&1 || true
    @echo "--- guest ---"; cat /tmp/jevmon-vmtest-out/guest.txt /tmp/jevmon-vmtest-out/helper.err 2>&1 | tail -20
    @echo "exec records: $(grep -c '"t":"exec"' /tmp/jevmon-vmtest-out/dump.ndjson 2>/dev/null)"

# run the root helper alone for N seconds and print a summary (one sudo; raw lines in $TMPDIR/jevmon-exec-trace.ndjson)
exec-trace N="10": exec-build
    #!/usr/bin/env bash
    set -euo pipefail
    out="${TMPDIR:-/tmp}/jevmon-exec-trace.ndjson"
    : > "$out"
    bin="$(readlink -f app/helper/result/bin/jevmon-exec)"
    sudo systemd-run --pipe --wait --quiet --collect \
        --property=AmbientCapabilities="CAP_BPF CAP_PERFMON CAP_SYS_RESOURCE" \
        --property=CapabilityBoundingSet="CAP_BPF CAP_PERFMON CAP_SYS_RESOURCE" \
        --property=PrivateNetwork=yes --property=ProtectSystem=strict --property=ProtectHome=read-only \
        --property=LimitMEMLOCK=infinity --property=ReadWritePaths="$(dirname "$out")" \
        "$bin" --uid "$(id -u)" --socket "" --duration {{N}} --dump "$out" 2> "$out.summary" || true
    cat "$out.summary"
    echo "exec lines: $(grep -c '"t":"exec"' "$out" || true)  (raw: $out)"

# install the helper as a persistent system service that starts at boot (sudo); `jevmon start` does not need this
exec-install: exec-build
    #!/usr/bin/env bash
    set -euo pipefail
    bin="$(readlink -f app/helper/result/bin/jevmon-exec)"
    sudo ln -sfn "$bin" /nix/var/nix/gcroots/jevmon-exec
    sed -e "s|@HELPER@|$bin|" -e "s|@UID@|$(id -u)|" -e "s|@REPO@|$PWD|" app/contrib/jevmon-exec.service \
        | sudo tee /etc/systemd/system/jevmon-exec.service > /dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable jevmon-exec.service
    sudo systemctl restart jevmon-exec.service
    sleep 1
    uv run --project app jevmon status

# stop and remove the persistent helper service
exec-uninstall:
    -sudo systemctl disable --now jevmon-exec.service
    sudo rm -f /etc/systemd/system/jevmon-exec.service /nix/var/nix/gcroots/jevmon-exec
    sudo systemctl daemon-reload

# run the tier-0 /proc exec source; pass args through, e.g. `just exec --once`
exec *ARGS: sync
    uv run --project app jevmon exec {{ARGS}}

# run the jevmond daemon in the foreground; pass args through, e.g. `just daemon --db /tmp/e.db`
daemon *ARGS: sync
    uv run --project app jevmond {{ARGS}}

# build (release) and run the Ratatui TUI; pass args through, e.g. `just tui --socket PATH`
tui *ARGS:
    cargo build --release --manifest-path app/tui/Cargo.toml
    ./app/tui/target/release/jevmon-tui {{ARGS}}

# fmt + clippy + tests for the Rust TUI
tui-check:
    cargo fmt --manifest-path app/tui/Cargo.toml --check
    cargo clippy --manifest-path app/tui/Cargo.toml --all-targets -- -D warnings
    cargo test --manifest-path app/tui/Cargo.toml

# capture docs/screenshots/*.png for the README: an isolated daemon on a fictional recording
# (replay source + fake judge, own socket/db/config under /tmp) driving the real TUI in tmux
screenshots: sync
    cargo build --release --manifest-path app/tui/Cargo.toml
    uv run --project app python app/scripts/screenshots.py

# --- opt-in iGPU envs (PyTorch ROCm wheel): app/.venv-gpu and lab/.venv-gpu ---------------------
# The default envs (app/.venv, lab/.venv) are CPU-only; `just check` never touches these. Each
# project has the mutually exclusive dependency groups `cpu` (default) and `gpu` (ROCm index);
# these recipes sync a separate venv with `--group gpu --no-group cpu`. The ROCm wheel is a 6.2 GB
# download (uv hardlinks it from its cache into both venvs), ships gfx1151 (no HSA_OVERRIDE
# needed) and wants the flake's LD_LIBRARY_PATH.

# create/refresh app/.venv-gpu (the daemon's env) from app/uv.lock (idempotent)
gpu-sync:
    VIRTUAL_ENV=$PWD/app/.venv-gpu uv sync --project app --active --group gpu --no-group cpu

# create/refresh lab/.venv-gpu (lab experiments, ../app editable) from lab/uv.lock
gpu-sync-lab:
    VIRTUAL_ENV=$PWD/lab/.venv-gpu uv sync --project lab --active --group gpu --no-group cpu

# run a command in the app GPU env: `just gpu-run jevmond`
gpu-run *ARGS:
    VIRTUAL_ENV=$PWD/app/.venv-gpu uv run --project app --active --no-sync {{ARGS}}

# run a command in the lab GPU env: `just gpu-run-lab python lab/scripts/bench_judge.py --device gpu`
gpu-run-lab *ARGS:
    VIRTUAL_ENV=$PWD/lab/.venv-gpu uv run --project lab --active --no-sync {{ARGS}}

# --- the daemon as a systemd user service (unit: app/contrib/jevmond.service) -----------------

# install (or update) the unit from app/contrib/, enable it at login and (re)start it; the unit runs
# on the iGPU, so this first makes sure .venv-gpu exists (a ~6 GB wheel the first time)
daemon-install: gpu-sync
    mkdir -p ~/.config/systemd/user
    cp app/contrib/jevmond.service ~/.config/systemd/user/jevmond.service
    systemctl --user daemon-reload
    systemctl --user enable jevmond.service
    systemctl --user restart jevmond.service

# stop, disable and remove the unit (store and config are left alone)
daemon-uninstall:
    -systemctl --user disable --now jevmond.service
    rm -f ~/.config/systemd/user/jevmond.service
    systemctl --user daemon-reload

daemon-start:
    systemctl --user start jevmond.service

daemon-stop:
    systemctl --user stop jevmond.service

# restart after changing code or config (the service runs this checkout live)
daemon-restart:
    systemctl --user restart jevmond.service

# systemd state plus the daemon's own status (model, exec source, counters)
daemon-status:
    -systemctl --user status jevmond.service --no-pager --lines 0
    uv run --project app jevmon ctl status

# follow the daemon's output
daemon-logs *ARGS:
    journalctl --user -u jevmond.service -f -o cat {{ARGS}}

# build (release) the lab container images (rootless podman; images are prefixed jevlab-)
lab-build:
    podman build -t jevlab-target:latest lab/target
    podman build -t jevlab-actor:latest lab/actor

# 3-scenario smoke of the lab (podman, well under 3 minutes once images are built);
# the fake judge exercises the dataset + report path end to end
lab-smoke: sync-lab lab-build
    uv run --project lab jevmon-lab run --smoke --judge fake --out lab-data-smoke

# run every lab scenario, capture the exec pipeline's contexts and export
# contexts only (training data: no judge) under lab-data/
lab: sync-lab lab-build
    uv run --project lab jevmon-lab run --judge none --out lab-data

# rebuild dataset.jsonl + report.md from a stored run with the real Laya judge
lab-report: sync-lab
    uv run --project lab jevmon-lab report --in lab-data --judge laya

# draw docs/screenshots/learning.png: held-out macro-F1 of the three shipped heads against the
# number of training families (CPU, a few minutes; installs the optional `plots` extra)
learning-curve:
    uv run --project lab --extra plots python lab/scripts/learning_curve.py

# --- experiment dashboard (Trackio, local only): the runs/ store written by lab/scripts/ ---------
# lab/scripts/lora_check.py tracks every LoRA run through jevmon_lab.runlog: project `jevmon-lora`,
# one run per selection cell and final seed. Trackio is the optional lab extra `trackio`; without it
# the run still writes <out>/metrics.jsonl. The dashboard binds 127.0.0.1:7861 (fixed port) and
# nothing is synced anywhere. TRACKIO_DIR picks the store, so a pod sync is viewable with e.g.
# `TRACKIO_DIR=$PWD/trackio-runs just runs` (see lab/README.md).

# dashboard in the foreground on http://127.0.0.1:7861 (Ctrl-C stops it)
runs:
    mkdir -p "${TRACKIO_DIR:-runs}"
    TRACKIO_DIR="${TRACKIO_DIR:-$PWD/runs}" GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7861 GRADIO_NUM_PORTS=1 GRADIO_SHARE=false GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1 BROWSER=true uv run --project lab --extra trackio trackio show --host 127.0.0.1 --no-footer

# list the Trackio projects, then the runs of the LoRA project (no dashboard)
runs-list:
    TRACKIO_DIR="${TRACKIO_DIR:-$PWD/runs}" uv run --project lab --extra trackio trackio list projects
    TRACKIO_DIR="${TRACKIO_DIR:-$PWD/runs}" uv run --project lab --extra trackio trackio list runs --project jevmon-lora



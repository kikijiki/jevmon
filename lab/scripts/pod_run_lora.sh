#!/usr/bin/env bash
# Run BOTH LoRA checks on a fresh Runpod PyTorch pod: Ettin, then the tracked Laya rerun.
#
# Workflow:
#   1. from the dev box, rsync the worktree to the pod at /workspace/jevmon. Include the Laya
#      baseline (lab-data-lora/results.json) and, if you want the paired tests, the frozen-embedding
#      caches (lab/cache/):
#        rsync -av --exclude '.venv*' --exclude '__pycache__' --exclude '.git' \
#              --exclude 'lab-data-lora-rerun/' --exclude 'trackio-runs/' \
#              /path/to/jevmon-lora-ettin/ root@<pod>:/workspace/jevmon/
#   2. on the pod:  bash /workspace/jevmon/lab/scripts/pod_run_lora.sh
#
# Installs the runtime deps with uv (torch CUDA stays the pod image's), then runs, sequentially and
# resumably under nohup (each command skips cells/seeds already finished):
#   Ettin : head -> select -> final -> report   -> lab-data-lora-ettin/
#   Laya  : head -> select -> final -> report   -> lab-data-lora-rerun/, --baseline lab-data-lora/
# Trackio's store is $TRACKIO_DIR (/workspace/trackio-runs) so it can be rsynced back and viewed.
#
# Env knobs (all optional):
#   REPO=/workspace/jevmon   DEVICE=cuda   BATCH=32   GRAD_CKPT=1   FOREGROUND=1
#   TRACKIO_DIR=/workspace/trackio-runs   HF_HOME=/workspace/.hf   LOG=$REPO/pod_run_lora.log
set -euo pipefail

REPO="${REPO:-/workspace/jevmon}"
DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-32}"
GRAD_CKPT="${GRAD_CKPT:-0}"
FOREGROUND="${FOREGROUND:-0}"
export TRACKIO_DIR="${TRACKIO_DIR:-/workspace/trackio-runs}"
export HF_HOME="${HF_HOME:-/workspace/.hf}"
LOG="${LOG:-$REPO/pod_run_lora.log}"
ETTIN_OUT="$REPO/lab-data-lora-ettin"
LAYA_OUT="$REPO/lab-data-lora-rerun"
mkdir -p "$ETTIN_OUT" "$LAYA_OUT" "$TRACKIO_DIR" "$HF_HOME"

# --- 1. deps (uv; pinned like lab/uv.lock; torch CUDA and numpy stay the pod image's) -------------
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv pip install --system --break-system-packages \
  "peft==0.21.2" "transformers==5.19.0" "trackio==0.39.0" "laya==0.4.6" \
  "safetensors==0.8.0" "tomlkit==0.15.2" "pyarrow==26.0.0"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device: this script expects a GPU pod")
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
PY

# --- 2. the run (resumable) -----------------------------------------------------------------------
GRAD_FLAG=""
[ "$GRAD_CKPT" = "1" ] && GRAD_FLAG="--grad-checkpointing"

cat > "$REPO/run_lora.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$REPO/app/src:$REPO/lab/src:$REPO/lab/scripts"
export HF_HOME="$HF_HOME"
export TRACKIO_DIR="$TRACKIO_DIR"
cd "$REPO"
train() { echo "[\$(date -Is)] \$*"; python lab/scripts/lora_check.py "\$@" --device "$DEVICE" --batch "$BATCH" $GRAD_FLAG --runs "$TRACKIO_DIR"; }

echo "[\$(date -Is)] === Ettin ==="
python lab/scripts/lora_check.py head --backbone ettin --out lab-data-lora-ettin --runs "$TRACKIO_DIR"
train select --backbone ettin --out lab-data-lora-ettin
train final  --backbone ettin --out lab-data-lora-ettin
python lab/scripts/lora_check.py report --backbone ettin --out lab-data-lora-ettin --runs "$TRACKIO_DIR"

echo "[\$(date -Is)] === Laya rerun ==="
python lab/scripts/lora_check.py head --backbone laya --out lab-data-lora-rerun --runs "$TRACKIO_DIR"
train select --backbone laya --out lab-data-lora-rerun
train final  --backbone laya --out lab-data-lora-rerun
python lab/scripts/lora_check.py report --backbone laya --out lab-data-lora-rerun \\
    --baseline lab-data-lora/results.json --runs "$TRACKIO_DIR"

echo "[\$(date -Is)] done -> lab-data-lora-ettin/ and lab-data-lora-rerun/ (+ $TRACKIO_DIR)"
EOF
chmod +x "$REPO/run_lora.sh"

cd "$REPO"
if [ "$FOREGROUND" = "1" ]; then
  echo "foreground; log tee'd to $LOG"
  bash "$REPO/run_lora.sh" 2>&1 | tee "$LOG"
else
  nohup bash "$REPO/run_lora.sh" > "$LOG" 2>&1 &
  echo "launched pid $! | log: $LOG"
  echo "follow:  tail -f $LOG"
  echo "rerun:   bash $REPO/lab/scripts/pod_run_lora.sh   (resumable)"
  echo "results: lab-data-lora-ettin/{results.json,summary.md}, lab-data-lora-rerun/{results.json,summary.md}"
  echo "trackio: $TRACKIO_DIR  (rsync back; view with TRACKIO_DIR=<dir> just runs)"
fi

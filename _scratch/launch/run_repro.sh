#!/bin/bash
# ============================================================================
# Expanded, self-contained 1-node 8xH200 DROID action-SFT test.
# The Lepton --command only clones the repo then runs this script:
#   git clone -b <branch> <url> /tmp/cf && bash /tmp/cf/_scratch/launch/run_repro.sh
# This script does everything else: system deps -> uv sync (cu130-train) ->
# venv checks -> dryrun -> 30-iter train at res480, max_samples_per_batch=64.
# No `source`, no base64 — read top to bottom. Markers
# (UV_SYNC_OK/IMPORTS_OK/DRYRUN_EXIT/TRAIN_EXIT) show exactly where it stops.
# ============================================================================
set +e
export DEBIAN_FRONTEND=noninteractive

echo "########## [0/4] system deps ##########"
apt-get update -qq && apt-get install -y -qq curl ffmpeg >/tmp/apt.log 2>&1
echo "curl=$(command -v curl) ffmpeg=$(command -v ffmpeg)"

echo "########## [1/4] uv + venv (cu130-train) ##########"
cd /tmp/cf || { echo "NO_REPO"; exit 1; }
echo "REPO head=$(git rev-parse --short HEAD 2>/dev/null) pyproject=$([ -f pyproject.toml ] && echo yes || echo NO)"
# ALWAYS install latest uv (container's may be too old for required-version >=0.11.3).
curl -LsSf https://astral.sh/uv/install.sh | sh >/tmp/uv.log 2>&1
export PATH="$HOME/.local/bin:$PATH"; hash -r
UV="$HOME/.local/bin/uv"; [ -x "$UV" ] || UV=uv
echo "uv=$($UV --version 2>&1)"
for att in 1 2 3 4 5 6 7 8; do
  echo "uv sync attempt $att"
  "$UV" sync --all-extras --group=cu130-train >/tmp/sync.log 2>&1 && { echo "UV_SYNC_OK"; break; } || { echo "sync $att failed:"; tail -8 /tmp/sync.log; sleep 15; }
done
VENV=/tmp/cf/.venv
[ -f "$VENV/bin/activate" ] && echo "VENV_OK" || echo "VENV_MISSING"
source "$VENV/bin/activate" && export LD_LIBRARY_PATH=
TORCHRUN="$VENV/bin/torchrun"   # force venv torchrun (never base-image python)
if [ ! -x "$VENV/bin/python" ] || [ ! -x "$TORCHRUN" ]; then
  echo "FATAL_VENV_INCOMPLETE: missing $VENV/bin/python or torchrun"
  echo "----- sync.log tail -----"; tail -40 /tmp/sync.log; echo "ALL_DONE"; exit 1
fi
"$VENV/bin/python" -c "import torch,loguru;print('IMPORTS_OK torch',torch.__version__,'devs',torch.cuda.device_count())" 2>&1 | tail -2 \
  || { echo "FATAL_IMPORT_FAIL"; tail -40 /tmp/sync.log; echo "ALL_DONE"; exit 1; }

echo "########## [2/4] env + data paths ##########"
export HF_HOME=/fwd4/.cache/huggingface
VAE=$(find /fwd4/.cache/huggingface /fwd4/cosmos3_action_runs -name Wan2.2_VAE.pth 2>/dev/null | head -1)
export WAN_VAE_PATH="$VAE"
echo "WAN_VAE_PATH=$VAE (exists=$( [ -f "$VAE" ] && echo yes || echo NO ))"
export DROID_ROOT=/fwd4/droid_plus_lerobot_640x360_20260412/success
export BASE_CHECKPOINT_PATH=/fwd4/cosmos3_action_runs/cosmos3_nano_dcp
echo "DROID_ROOT=$( [ -d "$DROID_ROOT" ] && echo yes || echo NO ) BASE_CKPT=$( [ -d "$BASE_CHECKPOINT_PATH" ] && echo yes || echo NO )"
export IMAGINAIRE_OUTPUT_ROOT=/fwd4/cosmos3_action_runs/repro_test
# Reduce CUDA fragmentation (the OOM at batch 64 left ~5 GiB reserved-unallocated).
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TOML=examples/toml/sft_config/action_policy_droid_repro.toml
# batch 64 @ res480 OOMs on 139GiB H200 (internal used 128 on GB200). 32 per rank fits.
BATCH="${MAX_SAMPLES:-32}"
OPTS="dataloader_train.max_samples_per_batch=${BATCH} job.wandb_mode=disabled"
LOGD=/fwd4/cosmos3_action_runs/repro_test_log; rm -rf "$LOGD"; mkdir -p "$LOGD"

echo "########## [3/4] DRYRUN (res480, batch64) ##########"
"$TORCHRUN" --nproc_per_node=8 --standalone -m cosmos_framework.scripts.train \
  --sft-toml="$TOML" --dryrun -- $OPTS > /tmp/dry.log 2>&1
echo "DRYRUN_EXIT=$?"; tail -30 /tmp/dry.log

echo "########## [4/4] TRAIN 30 iters ##########"
"$TORCHRUN" --nproc_per_node=8 --standalone --tee 3 --redirects 3 --log-dir "$LOGD" \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- $OPTS trainer.max_iter=30
echo "TRAIN_EXIT=$?"
echo "=== rank0 stderr tail ==="; tail -90 "$LOGD"/*/attempt_0/0/stderr.log 2>/dev/null
echo "ALL_DONE"
echo "########## KEEPALIVE 900s (stream log before purge) ##########"
sleep 900

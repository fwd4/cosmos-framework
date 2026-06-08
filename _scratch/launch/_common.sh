#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Shared setup for cdg Lepton launch scripts. Sourced after the repo is cloned
# to /tmp/cf by the bootstrap command. NOT part of the OSS recipe (scratch only).
set +e
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh >/tmp/uv.log 2>&1; source "$HOME/.local/bin/env" 2>/dev/null; }
uv --version || echo "UV_MISSING"
cd /tmp/cf || { echo "NO_REPO"; exit 1; }
for att in 1 2 3 4 5; do
  echo "uv sync attempt $att"
  uv sync --all-extras --group=cu130-train >/tmp/sync.log 2>&1 && { echo UV_SYNC_OK; break; } || { echo "sync $att failed:"; tail -6 /tmp/sync.log; sleep 10; }
done
VENV=/tmp/cf/.venv
[ -f "$VENV/bin/activate" ] && echo "VENV_OK" || echo "VENV_MISSING"
source "$VENV/bin/activate" && export LD_LIBRARY_PATH=
# Force the venv interpreter/torchrun so we never silently fall back to the
# base image's system Python (which lacks project deps like loguru).
export TORCHRUN="$VENV/bin/torchrun"
if [ ! -x "$VENV/bin/python" ] || [ ! -x "$TORCHRUN" ]; then
  echo "FATAL_VENV_INCOMPLETE: missing $VENV/bin/python or torchrun — uv sync did not finish."
  echo "----- sync.log tail -----"; tail -40 /tmp/sync.log; echo "----- end sync.log -----"
  echo "ALL_DONE"; exit 1
fi
"$VENV/bin/python" -c "import torch,loguru;print('IMPORTS_OK torch',torch.__version__,torch.__file__,'devs',torch.cuda.device_count())" 2>&1 | tail -3 \
  || { echo "FATAL_IMPORT_FAIL (loguru/torch not importable in venv)"; tail -40 /tmp/sync.log; echo "ALL_DONE"; exit 1; }
export HF_HOME=/fwd4/.cache/huggingface
VAE=$(find /fwd4/.cache/huggingface /fwd4/cosmos3_action_runs -name Wan2.2_VAE.pth 2>/dev/null | head -1)
export WAN_VAE_PATH="$VAE"
echo "WAN_VAE_PATH=$VAE (exists=$( [ -f "$VAE" ] && echo yes || echo NO ))"
export DROID_ROOT=/fwd4/droid_plus_lerobot_640x360_20260412/success
export BASE_CHECKPOINT_PATH=/fwd4/cosmos3_action_runs/cosmos3_nano_dcp
echo "DROID_ROOT=$( [ -d "$DROID_ROOT" ] && echo yes || echo NO ) BASE_CKPT=$( [ -d "$BASE_CHECKPOINT_PATH" ] && echo yes || echo NO )"
export TOML=examples/toml/sft_config/action_policy_droid_repro.toml
export OPTS="dataloader_train.max_samples_per_batch=${MAX_SAMPLES:-64} job.wandb_mode=disabled"

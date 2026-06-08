#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# REAL RUN: 8-node / 64-rank DROID action-SFT (HSDP shard 8 x replicate=nnodes).
# global batch = 32/rank x 64 ranks = 2048. res480, max_iter=10000, ckpt/1000.
# wandb online (WANDB_API_KEY from Lepton secret). Output persisted on /fwd4 so
# re-submitting the SAME job.name auto-resumes from latest checkpoint.
# Launch cmd: git clone ... /tmp/cf && bash /tmp/cf/_scratch/launch/run_real_8node.sh
# ============================================================================
set +e
export DEBIAN_FRONTEND=noninteractive
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "########## [0/4] system deps ##########"
apt-get update -qq && apt-get install -y -qq curl ffmpeg >/tmp/apt.log 2>&1

echo "########## [1/4] uv + venv (cu130-train) ##########"
cd /tmp/cf || { echo "NO_REPO"; exit 1; }
echo "REPO head=$(git rev-parse --short HEAD 2>/dev/null)"
# ALWAYS install latest uv (container uv too old for required-version >=0.11.3)
curl -LsSf https://astral.sh/uv/install.sh | sh >/tmp/uv.log 2>&1
export PATH="$HOME/.local/bin:$PATH"; hash -r
UV="$HOME/.local/bin/uv"; [ -x "$UV" ] || UV=uv
echo "uv=$($UV --version 2>&1)"
for att in 1 2 3 4 5 6 7 8; do
  echo "uv sync attempt $att"
  "$UV" sync --all-extras --group=cu130-train >/tmp/sync.log 2>&1 && { echo "UV_SYNC_OK"; break; } || { echo "sync $att failed:"; tail -8 /tmp/sync.log; sleep 15; }
done
VENV=/tmp/cf/.venv
source "$VENV/bin/activate" && export LD_LIBRARY_PATH=
TORCHRUN="$VENV/bin/torchrun"
if [ ! -x "$TORCHRUN" ]; then echo "FATAL_VENV_INCOMPLETE"; tail -40 /tmp/sync.log; sleep 300; exit 1; fi
"$VENV/bin/python" -c "import torch,loguru;print('IMPORTS_OK torch',torch.__version__,'devs',torch.cuda.device_count())" 2>&1 | tail -2

echo "########## [2/4] env + data + HSDP + wandb ##########"
export HF_HOME=/fwd4/.cache/huggingface
export WAN_VAE_PATH=$(find /fwd4/.cache/huggingface /fwd4/cosmos3_action_runs -name Wan2.2_VAE.pth 2>/dev/null | head -1)
export DROID_ROOT=/fwd4/droid_plus_lerobot_640x360_20260412/success
export BASE_CHECKPOINT_PATH=/fwd4/cosmos3_action_runs/cosmos3_nano_dcp
export IMAGINAIRE_OUTPUT_ROOT=/fwd4/cosmos3_action_runs/droid_repro_real   # persistent → resumable
echo "WANDB_API_KEY set=$([ -n "$WANDB_API_KEY" ] && echo yes || echo NO)"
NNODES="${LEPTON_JOB_TOTAL_WORKERS:-8}"
NRANK="${LEPTON_JOB_WORKER_INDEX:-0}"
TOML=examples/toml/sft_config/action_policy_droid_repro.toml
# HSDP: shard 8 (intra-node) x replicate=NNODES (across nodes) = 8*NNODES ranks
sed -i "s/data_parallel_replicate_degree = 1/data_parallel_replicate_degree = ${NNODES}/" "$TOML"
echo "PARALLELISM_$(grep -E 'replicate_degree|shard_degree' "$TOML" | tr '\n' '_')"
# global batch = 32/rank * (8*NNODES) ranks ; save every 1000 iters
OPTS="dataloader_train.max_samples_per_batch=32 checkpoint.save_iter=1000"

echo "########## [3/4] rendezvous ##########"
MASTER="${LEPTON_JOB_NAME}-0.${LEPTON_SUBDOMAIN}.ws-${LEPTON_WORKSPACE_ID}.svc.cluster.local"
echo "RDZV nnodes=$NNODES node_rank=$NRANK master=$MASTER"
getent hosts "$MASTER" && echo "MASTER_DNS_OK" || echo "MASTER_DNS_FAIL"
LOGD=/fwd4/cosmos3_action_runs/real_log_${NRANK}; mkdir -p "$LOGD"

echo "########## [4/4] REAL TRAIN max_iter=10000, global batch 2048 ##########"
"$TORCHRUN" --nnodes="$NNODES" --node_rank="$NRANK" --nproc_per_node=8 \
  --master_addr="$MASTER" --master_port=29500 \
  --tee 3 --redirects 3 --log-dir "$LOGD" \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- $OPTS
echo "TRAIN_EXIT=$?_NODE_${NRANK}"
echo "=== local rank0 stderr tail ==="; tail -60 "$LOGD"/*/attempt_0/0/stderr.log 2>/dev/null
echo "ALL_DONE_NODE_${NRANK}"

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 2-node / 16-rank test: HSDP shard 8 x replicate 2, res480, max_samples_per_batch=64,
# 30-iter train. Rendezvous via Lepton multi-worker env (confirmed via mw-probe3).
source /tmp/cf/_scratch/launch/_common.sh
export IMAGINAIRE_OUTPUT_ROOT=/fwd4/cosmos3_action_runs/repro_2node

MASTER="${LEPTON_JOB_NAME}-0.${LEPTON_SUBDOMAIN}.ws-${LEPTON_WORKSPACE_ID}.svc.cluster.local"
NNODES="${LEPTON_JOB_TOTAL_WORKERS:-2}"
NRANK="${LEPTON_JOB_WORKER_INDEX:-0}"
echo "RDZV nnodes=$NNODES node_rank=$NRANK master=$MASTER"

# HSDP: 8 (intra-node shard) x 2 (cross-node replicate) = 16 ranks
sed -i 's/data_parallel_replicate_degree = 1/data_parallel_replicate_degree = 2/' "$TOML"
LOGD=/fwd4/cosmos3_action_runs/repro_2node_log_${NRANK}; rm -rf "$LOGD"; mkdir -p "$LOGD"

echo "===== 2-NODE TRAIN 30 iters (16 ranks) ====="
"$TORCHRUN" --nnodes="$NNODES" --node_rank="$NRANK" --nproc_per_node=8 \
  --master_addr="$MASTER" --master_port=29500 \
  --tee 3 --redirects 3 --log-dir "$LOGD" \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- $OPTS trainer.max_iter=30
echo "TRAIN_EXIT=$? (node_rank=$NRANK)"
echo "=== local rank0 stderr tail ==="; tail -60 "$LOGD"/*/attempt_0/0/stderr.log 2>/dev/null
echo "ALL_DONE_NODE_${NRANK}"
echo "===== KEEPALIVE 600s ====="
sleep 600

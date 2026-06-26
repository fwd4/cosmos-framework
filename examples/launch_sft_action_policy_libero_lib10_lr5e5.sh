#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# "lib10_lr5e5" launch: libero_10-ONLY SFT with the reference LR recipe
# (lr 5e-5, warmup 500, cycle 16000) on HSDP 8x8 (64 GPUs), global batch 2048.
# Head-to-head with lr1e4c10k (lr 1e-4) and the 4-suite run (same LR, 4 suites).
#
# Global batch = max_samples_per_batch x (shard x replicate) x grad_accum.
# The TOML pins shard=8/replicate=8 (64 ranks) and grad_accum=1; this wrapper
# pins max_samples_per_batch=32 so 32 x 64 x 1 = 2048.
#
# Required: LIBERO_ROOT = local libero_10 LeRobot dir (meta/info.json present).
# Optional: BASE_CHECKPOINT_PATH, WAN_VAE_PATH, OUTPUT_ROOT.
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR per node.

TOML_FILE="examples/toml/sft_config/action_policy_libero_repro_lib10_lr5e5.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

export LIBERO_ROOT="${LIBERO_ROOT:-}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\''). Pre-sync: hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include '\''libero_10/**'\'' --local-dir <dir> (then LIBERO_ROOT=<dir>/libero_10)." >&2; exit 1; }'

TAIL_OVERRIDES=(
    dataloader_train.max_samples_per_batch=32
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

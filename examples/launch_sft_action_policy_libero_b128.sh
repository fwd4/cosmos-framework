#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# "b128" launch for Cosmos3-Nano LIBERO action-policy SFT — small GLOBAL BATCH
# (128) matching the paper / issue-#50 ~2.7-passes regime, on HSDP 4x4 (16 GPUs).
# Loss is the GA default (loss_scale=10, action_loss_weight=10) — NOT overridden.
#
# Global batch = max_samples_per_batch x (shard x replicate) x grad_accum.
# The TOML pins shard=4/replicate=4 (16 ranks) and grad_accum=1; this wrapper
# pins max_samples_per_batch=8 (a dataloader_train field, set via Hydra opt) so
# 8 x 16 x 1 = 128. See action_policy_libero_repro_b128.toml.
#
# Required / optional env vars: same as launch_sft_action_policy_libero.sh.
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR per node. Extra overrides
# (wandb/name) via EXTRA_TAIL_OVERRIDES.
#
# Usage (2 nodes x 8 GPU = HSDP 4x4): bash examples/launch_sft_action_policy_libero_b128.sh

TOML_FILE="examples/toml/sft_config/action_policy_libero_repro_b128.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

export LIBERO_ROOT="${LIBERO_ROOT:-}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\''). Pre-sync: hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include '\''libero_10/**'\'' --local-dir <dir> (then LIBERO_ROOT=<dir>/libero_10). See docs/action_policy_libero_sft.md" >&2; exit 1; }'

# b128 batch knob: 8 samples/rank x 16 ranks x grad_accum1 = global 128.
# Caller's EXTRA_TAIL_OVERRIDES (wandb/name) appended last so it can override.
TAIL_OVERRIDES=(
    dataloader_train.max_samples_per_batch=8
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

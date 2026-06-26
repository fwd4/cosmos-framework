#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# "4suite_lr5e5" launch for Cosmos3-Nano LIBERO action-policy SFT — the reference
# DATA regime (4 suites, equal ratio) at lr 5e-5 / 16k iters on HSDP 8x8 (64 GPUs),
# global batch 2048. Loss is the GA default (loss_scale=10) — NOT overridden.
#
# Global batch = max_samples_per_batch x (shard x replicate) x grad_accum.
# The TOML pins shard=8/replicate=8 (64 ranks) and grad_accum=1; this wrapper
# pins max_samples_per_batch=32 so 32 x 64 x 1 = 2048.
# See action_policy_libero_repro_4suite_lr5e5.toml.
#
# Required: LIBERO_ROOT_BASE = local dir holding all 4 LeRobot suites as subdirs
#   (libero_10/ libero_object/ libero_spatial/ libero_goal/). Pre-sync once:
#     hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset \
#       --include 'libero_10/**' 'libero_object/**' 'libero_spatial/**' \
#       'libero_goal/**' --local-dir <base>     # LIBERO_ROOT_BASE=<base>
# Optional: BASE_CHECKPOINT_PATH, WAN_VAE_PATH, OUTPUT_ROOT.
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR per node.
#
# Usage (8 nodes x 8 GPU = HSDP 8x8): bash examples/launch_sft_action_policy_libero_4suite_lr5e5.sh

TOML_FILE="examples/toml/sft_config/action_policy_libero_repro_4suite_lr5e5.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

export LIBERO_ROOT_BASE="${LIBERO_ROOT_BASE:-}"

EXTRA_DATASET_CHECK='for _s in libero_10 libero_object libero_spatial libero_goal; do [[ -f "$LIBERO_ROOT_BASE/$_s/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT_BASE must hold all 4 LeRobot suites; missing $_s/meta/info.json under '\''$LIBERO_ROOT_BASE'\''. Pre-sync: hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include '\''libero_10/**'\'' '\''libero_object/**'\'' '\''libero_spatial/**'\'' '\''libero_goal/**'\'' --local-dir <base> (then LIBERO_ROOT_BASE=<base>)." >&2; exit 1; }; done'

# Batch knob: 32 samples/rank x 64 ranks x grad_accum1 = global 2048.
# Caller's EXTRA_TAIL_OVERRIDES (wandb/name) appended last so it can override.
TAIL_OVERRIDES=(
    dataloader_train.max_samples_per_batch=32
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

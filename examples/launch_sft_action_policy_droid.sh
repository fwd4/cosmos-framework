#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# DRAFT / NOT YET RUNNABLE.
# Structured-TOML launch for action_policy_droid_nano — DROID action-policy SFT
# on Cosmos3-Nano (8B MoT). Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_droid_nano.toml.
#
# Requires the action-training stack to land in cosmos-framework first
# (registered experiment + full DROID dataset class + dataset-prep). See
# docs/action_policy_droid_posttraining.md "Release prerequisites".
#
# Optional env vars (defaults under examples/; override for other filesystems):
#   DATASET_PATH          DROID LeRobot v3.0 root (…/lerobot_v30/droid_lerobot)
#   BASE_CHECKPOINT_PATH  DCP of the 8B midtrain checkpoint (from Step 2)
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   WANDB_API_KEY         for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8; use 4 on GB200)
#
# Smoke (single node, few iters):
#   TAIL_OVERRIDES=(trainer.max_iter=10 checkpoint.save_iter=10 \
#                   model.config.parallelism.data_parallel_shard_degree=8)
#   NPROC_PER_NODE=8 bash examples/launch_sft_action_policy_droid.sh
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_policy_droid_nano.toml"
: "${DATASET_PATH:=examples/data/lerobot_v30/droid_lerobot}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/meta/info.json" ]] || { echo "ERROR: missing $DATASET_PATH/meta/info.json (run DROID LeRobot v3.0 prep — see docs/action_policy_droid_posttraining.md)" >&2; exit 1; }'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

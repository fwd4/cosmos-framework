#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# "ls1" A/B launch for the Cosmos3-Nano LIBERO action-policy SFT — identical to
# launch_sft_action_policy_libero.sh except it removes the global 10x loss scale:
#   loss_scale 10 -> 1  AND  action_loss_weight 10 -> 1
# (relative vision:action balance unchanged at 1:1; only the global magnitude
# drops 10x). Under FusedAdam this is ~no-op on the update direction except that
# it stops saturating grad_clip(clip_norm=1.0). Tests whether the 10x was ever
# doing anything but clipping. See action_policy_libero_repro_ls1.toml.
#
# These two knobs live in model.config.rectified_flow_training_config, which the
# structured TOML schema does not expose, so they are applied here as trailing
# Hydra opts (the proven model.config.* / trainer.* override path).
#
# Required / optional env vars: same as launch_sft_action_policy_libero.sh
# (LIBERO_ROOT required; BASE_CHECKPOINT_PATH, WAN_VAE_PATH, OUTPUT_ROOT optional).
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR per node (the launcher appends
# them to torchrun). Extra overrides (wandb/name/topology) via EXTRA_TAIL_OVERRIDES.
#
# Usage (2 nodes x 8 GPU): bash examples/launch_sft_action_policy_libero_ls1.sh

TOML_FILE="examples/toml/sft_config/action_policy_libero_repro_ls1.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# LIBEROLeRobotDataset reads ${oc.env:LIBERO_ROOT} directly (a LOCAL LeRobot dir).
export LIBERO_ROOT="${LIBERO_ROOT:-}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\''). Pre-sync: hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include '\''libero_10/**'\'' --local-dir <dir> (then LIBERO_ROOT=<dir>/libero_10). See docs/action_policy_libero_sft.md" >&2; exit 1; }'

# ls1 A/B knobs (the whole point of this wrapper) first; caller's EXTRA_TAIL_OVERRIDES
# (wandb/name/topology) appended last so it can override anything here.
TAIL_OVERRIDES=(
    model.config.rectified_flow_training_config.loss_scale=1.0
    model.config.rectified_flow_training_config.action_loss_weight=1.0
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

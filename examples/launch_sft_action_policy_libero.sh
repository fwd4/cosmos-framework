#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_libero_nano — Cosmos3-Nano LIBERO
# action-policy SFT (8-GPU FSDP, full SFT, no LoRA). Reproduces the Table-20
# LIBERO-10 result (~97.4% @ ckpt 2000). Drives cosmos_framework.scripts.train
# against examples/toml/sft_config/action_policy_libero_repro.toml.
#
# REPRODUCTION: point LIBERO_ROOT at the libero_10 LeRobot conversion ONLY. The
# 4-suite mix dilutes libero_10 to ~1 pass in 2000 steps (~82%); libero_10 alone
# is ~2.7 passes (~97%). See docs/action_policy_libero_sft.md.
#
# Required env vars:
#   LIBERO_ROOT           local LIBERO-10 LeRobot dataset dir (no default)
# Optional env vars (defaults below; override to relocate data/checkpoints):
#   LIBERO_REPO_ID        default: lerobot/libero_10 (identifier; local root wins)
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   LIBERO_ROOT=/path/to/libero_10_lerobot bash examples/launch_sft_action_policy_libero.sh

TOML_FILE="examples/toml/sft_config/action_policy_libero_repro.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# LIBEROLeRobotDataset reads ${oc.env:LIBERO_ROOT} / ${oc.env:LIBERO_REPO_ID} directly;
# export them so torchrun (launched in this shell) inherits them.
: "${LIBERO_REPO_ID:=lerobot/libero_10}"
export LIBERO_REPO_ID
export LIBERO_ROOT="${LIBERO_ROOT:-}"

# LIBERO_ROOT is optional: if set it must be a local LeRobot dir; if empty, the
# dataset downloads $LIBERO_REPO_ID from the HF Hub.
EXTRA_DATASET_CHECK='if [[ -n "$LIBERO_ROOT" ]]; then [[ -d "$LIBERO_ROOT" ]] || { echo "ERROR: LIBERO_ROOT is set but not a directory: '\''$LIBERO_ROOT'\''. Unset it to download $LIBERO_REPO_ID from HF, or point it at a local libero_10 LeRobot dir. See docs/action_policy_libero_sft.md" >&2; exit 1; }; else echo ">>> LIBERO_ROOT unset -> will use HF dataset $LIBERO_REPO_ID"; fi'

# Extra Hydra overrides from the environment: a space-separated string word-split into
# the TAIL_OVERRIDES array. An exported string survives `bash <wrapper>` (a child
# process), unlike a TAIL_OVERRIDES array set in your shell. Use it for smoke runs,
# e.g. EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 job.wandb_mode=offline".
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

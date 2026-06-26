# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_libero_nano_4suite`` — 4-suite LIBERO action-policy SFT recipe.

Identical to :mod:`action_policy_libero_nano` (same Cosmos3-Nano GA base, model
config, optimizer, scheduler, checkpoint wiring) EXCEPT the training data is the
full 4-suite LIBERO mix — ``libero_10`` + ``libero_object`` + ``libero_spatial``
+ ``libero_goal`` — at equal sampling ratio (1:1:1:1), mirroring the reference
ablation run (``...exp506_midtraining_v1...16k_lr5e5_wu500``, wandb hqwbt7ve)
which trained on all four suites combined.

WHY: the libero_10-only recipe peaks ~95% then OVERFITS task 0 by iter 2000
(90% -> 70%). The reference reaches ~97% by training the 4-suite mix at lr 5e-5
for 16k iters; the data diversity is hypothesized to curb the single-task
over-fit. This recipe reproduces THAT data regime on the public GA base (the
reference's internal ``exp506`` mid-training base is not used here).

Each suite is read as a separate LeRobot dir under ``LIBERO_ROOT_BASE``:

    hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset \\
      --include 'libero_10/**' 'libero_object/**' 'libero_spatial/**' \\
      'libero_goal/**' --local-dir <base>     # LIBERO_ROOT_BASE=<base>

Usage (8 node x 8 GPU = HSDP 8x8, global batch 2048, lr 5e-5)::

    LIBERO_ROOT_BASE=/path/to/LIBERO_LeRobot_v3 \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    python -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_libero_repro_4suite_lr5e5.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import get_action_libero_sft_dataset

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano import (
    action_policy_libero_nano,
)

cs = ConfigStore.instance()


def _libero_suite_entry(suite: str) -> dict:
    """One ratio-1 dataset entry for a single LIBERO suite (same args as the
    libero_10-only recipe; only ``root`` differs)."""
    return dict(
        ratio=1,
        dataset=L(get_action_libero_sft_dataset)(
            root=f"${{oc.env:LIBERO_ROOT_BASE}}/{suite}",
            fps=20,  # metadata only (FPS-agnostic loader reads native fps from info.json)
            chunk_length=16,
            image_size=256,  # concat_view -> 256x512
            mode="policy",
            camera_mode="concat_view",
            action_space="frame_wise_relative",
            rotation_space="6d",
            pose_coordinate_frame="native",
            action_normalization="quantile_rot",
            val_ratio=0.01,
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            resolution=None,
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
        ),
    )


# Clone the maintained libero_10-only recipe and swap ONLY the name + dataset mix.
action_policy_libero_nano_4suite = copy.deepcopy(action_policy_libero_nano)
action_policy_libero_nano_4suite.job.name = "action_policy_libero_nano_4suite"
action_policy_libero_nano_4suite.dataloader_train.dataloader.datasets = dict(
    libero_10=_libero_suite_entry("libero_10"),
    libero_object=_libero_suite_entry("libero_object"),
    libero_spatial=_libero_suite_entry("libero_spatial"),
    libero_goal=_libero_suite_entry("libero_goal"),
)


for _item in [action_policy_libero_nano_4suite]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)

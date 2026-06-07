<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# DROID Action-Policy Post-Training — `Cosmos3-Nano-Policy-DROID`

> **STATUS: DRAFT / NOT YET RUNNABLE IN `cosmos-framework`.**
> This runbook documents the validated internal reproduction recipe and the OSS-facing
> command shape, but the public package does **not yet ship the action-training stack**
> needed to run it. See [Release prerequisites](#release-prerequisites) for the exact
> gating items. The internal, **validated** source-of-truth command is in
> [Appendix A](#appendix-a--internal-source-of-truth-command-validated).

Fine-tune `Cosmos3-Nano` (the 8B MoT) into an action policy on the public **DROID LeRobot**
dataset, reproducing `Cosmos3-Nano-Policy-DROID`. The policy is initialized from the 8B
**midtrain** checkpoint and trained with absolute joint-position actions + proprioceptive
state at 480p.

______________________________________________________________________

## Release prerequisites

These must land in `cosmos-framework` before this recipe is runnable here (today they exist
only in the internal `imaginaire4` tree). Tracked as the gating items for this deliverable:

1. **Action experiment config** — `configs/base/experiment/action/**` is currently empty
   scaffolding. Needs the registered experiment (internal `droid_lerobot_8b_policy`), its
   `make_*_experiment`/`register_modes` helpers, and `droid_lerobot_dataset_config`.
2. **Full DROID dataset class** — the public `data/vfm/action/datasets/droid_lerobot_dataset.py`
   is a stripped-down, `concat_view`-only, EE-pose reader. It lacks the recipe knobs
   (`action_space=joint_pos`, `use_state`, `use_image_augmentation`, success-only
   `use_filter_dict`, `video_mode`). Needs the full training dataset class.
3. **Dataset-prep scripts** — the LeRobot v2.0→v3.0 conversion pipeline (internal
   `pipelines/action/datasets/droid_lerobot/`) is not in the public repo.
4. **A public/​documented base checkpoint** — `--use-8b-midtrain-ckpt` maps to an internal
   object-store path; release needs a public or documented checkpoint source.

The release of (1)–(4) should be owned/reviewed by the action team (it decides which internal
code becomes public).

## Hardware

| | |
|---|---|
| Validated on | 64 × GB200 NVL72 nodes, 4 GPU/node = **256 GPUs** |
| Throughput | ~**9.2 s/iter** at global batch 8192 (256 ranks × 32 samples) |
| Parallelism | HSDP, `data_parallel_shard_degree=8` × replicate=32 |
| Smoke | runnable on far fewer GPUs for config/import/data sanity (see below) |

10k iters ≈ **~25 h**; the full **100k**-iter target is a multi-day run that must resume across
`batch_long` (168 h) windows from its checkpoints.

## Dataset — DROID LeRobot

Public source: **DROID LeRobot** (`IPEC-COMMUNITY/droid_lerobot`), converted to **LeRobot v3.0**.
Expected on-disk layout after prep:

```
lerobot_v30/droid_lerobot/
├── meta/{info.json, episodes/chunk-*/file-*.parquet, tasks.parquet, stats.json}
├── data/chunk-*/file-*.parquet
└── videos/...
```

The training recipe consumes the **success-filtered** split — **57,639 episodes (~300 hours)**.
Prep = HF download (v2.0) → v2.0→v2.1 → v2.1→v3.0 conversion (internal
`pipelines/action/datasets/droid_lerobot/01_download.sh … 05_test_lerobot_dataset.sh`; pending
port — prerequisite #3). Document filtering as a preprocessed input until the public filter script lands.

## Recipe (canonical knobs — preserve these)

| knob | value |
|---|---|
| init | 8B **midtrain** checkpoint |
| action space | `joint_pos` (absolute joint position, 8-D incl. gripper) |
| state | `use_state=true` (proprioception; valid only with `joint_pos`) |
| resolution | `480` |
| viewpoint / video | `concat_view` / `video_mode=null` |
| chunk length | `32` (tokenizer `encode_exact_durations=[33]`) |
| image augmentation | enabled |
| lr | `2e-4` |
| samples/rank | `32` → global batch `8192` at 256 ranks |
| eval | disabled for the reproduction run |

## Full reproduction (OSS command shape — pending prerequisites)

Once the action experiment + dataset class land (prereqs 1–2), the OSS flow mirrors the other
recipes (see [docs/training.md](./training.md)):

```shell
# Step 1: prepare DROID LeRobot v3.0 (prereq #3 prep scripts) -> $DATASET_PATH
# Step 2: convert base checkpoint -> $BASE_CHECKPOINT_PATH
python -m cosmos_framework.scripts.convert_model_to_dcp -o $BASE_CHECKPOINT_PATH --checkpoint-path Cosmos3-Nano

# Step 3: launch (64 nodes x 4 GPU); recipe TOML selects experiment + scalars,
# dataset/action knobs come from the registered experiment.
NPROC_PER_NODE=4 bash examples/launch_sft_action_policy_droid.sh
```

The recipe TOML (`examples/toml/sft_config/action_policy_droid_nano.toml`, DRAFT) sets the
scalar knobs (`lr=2e-4`, `max_samples_per_batch=32`, `save_iter`, `max_iter`, parallelism);
the dataset/action knobs (`joint_pos`, `use_state`, aug, 480p, chunk 32) live in the registered
experiment per the schema's design.

## Smoke reproduction

Config/import/data sanity without burning a full run: small node count + a handful of iters via
`--config-overrides "trainer.max_iter=10" "checkpoint.save_iter=10"` (and a small
`data_parallel_shard_degree`). Use this to validate the recipe composes and the dataset opens
before any large allocation.

## Checkpoints

- Saved every `save_iter` iters (1000 in the validated run) to the object store, at
  `<bucket>/<project>/<group>/<job.name>/checkpoints/iter_<N>/`.
- The run is **resumable** from the latest checkpoint — required for the multi-window 100k run.
- Export to HF safetensors via `cosmos_framework.scripts.export_model` (see training.md).

## Non-goals

- **Closed-loop / action evaluation is out of scope** for this reproduction pass (training
  reproduction only), unless explicitly expanded.

______________________________________________________________________

## Appendix A — internal source-of-truth command (validated)

Run today in `imaginaire4` (the launcher hardcodes internal container/account/object-store
assumptions; replace `WANDB_API_KEY`, `--user`, `--output-root`). This is the recipe the OSS
flow above mirrors:

```bash
WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH=. python \
  projects/cosmos3/vfm/configs/base/experiment/action/_submit_droid_lerobot.py \
  --nnode 64 --experiment droid_lerobot_8b_policy --user $USER \
  --output-root <your-output-root> \
  --training-iterations 100000 --resolution 480 --video-mode null \
  --action-space joint_pos --use-state --use-8b-midtrain-ckpt --no-eval \
  --use-image-augmentation --chunk-length 32 --lr 2e-4 --max-samples-per-batch 32
```

## Appendix B — internal validation evidence (2026-06-04)

A 10k-iter reproduction (Slurm job, 64 nodes × 4 GPU = 256 ranks) confirmed the recipe end to
end:

- Loaded the 8B midtrain checkpoint (`iter_000040000`) from the object store; built the DROID
  dataloader with the canonical knobs; W&B logging healthy.
- Loss: **27.2 (it1) → 8.8 (it5) → ~1.2 (it1000) → ~1.1–1.4 (steady)**; ~9.2 s/iter.
- Checkpoints written every 1000 iters to the object store (resumable).
- The four prerequisite MRs (9584/9586/9464/9588) were cherry-picked onto `main` for this run.

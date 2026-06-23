# Cosmos3-Nano LIBERO action-policy SFT (reproduction)

Reproduces the Cosmos3-Nano LIBERO-10 result (technical report Table 20, ~97.4%
success at checkpoint 2000) as an action policy: vision + language in, action
chunks out. Full SFT (no LoRA) on the public `nvidia/Cosmos3-Nano` base.

Pieces:

| Piece | Path |
| --- | --- |
| Dataset | `cosmos_framework/data/vfm/action/datasets/libero_lerobot_dataset.py` (`LIBEROLeRobotDataset`) |
| SFT wrapper | `get_action_libero_sft_dataset` in `.../datasets/action_sft_dataset.py` |
| Norm stats | `.../datasets/stats/libero_native_frame_wise_relative_rot6d.json` |
| Experiment | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_libero_nano.py` |
| Run TOML | `examples/toml/sft_config/action_policy_libero_repro.toml` |
| Launch | `examples/launch_sft_action_policy_libero.sh` |
| Inference server | `cosmos_framework/scripts/action_policy_server_libero.py` |
| Closed-loop eval | `cosmos_framework/simulation/libero/closed_loop_eval.py` |

## 1. Data

`LIBEROLeRobotDataset` reads a **local** LeRobot dir directly (parquet + video,
like `DROIDLeRobotDataset`) — set `LIBERO_ROOT` to it. Pre-sync the HF dataset
once to shared storage:

```bash
hf download lerobot/libero_10 --repo-type dataset --local-dir <nfs>/libero_10
export LIBERO_ROOT=<nfs>/libero_10
```

**For the Table-20 number, use `libero_10` ALONE.** Training on the full 4-suite
mix (libero_10 / object / spatial / goal) gives libero_10 only ~1 pass in 2000
steps (~82%); libero_10 alone is ~2.7 passes (~97%). For more suites, add more
`datasets=dict(...)` entries to the experiment's dataloader.

It uses `frame_wise_relative` rot6d actions (10D = `pos(3) + rot6d(6) +
gripper(1)`), `concat_view` (third-person + wrist, each resized to 256×256,
concatenated horizontally → 256×512), normalized with `quantile_rot` against the
bundled stats.

**FPS-agnostic.** The loader windows by frame index and decodes video at each
frame's real timestamp, so it works with any LeRobot LIBERO dataset regardless of
FPS (no `delta_timestamps` grid). `fps` is metadata only (`conditioning_fps` +
prompt duration). **Fidelity caveat:** the public `lerobot/libero_*` datasets are
**10 FPS**, but the bundled `quantile_rot` stats were computed on NVIDIA's **20
FPS** conversion; per-frame deltas at 10 FPS span 2× the wall-clock motion, so for
a faithful Table-20 reproduction use a 20 FPS LIBERO dataset (or recompute stats
for the dataset's FPS). See [§5](#5-fps--stats).

**Model-input resolution = 192×320.** The 256×512 concat is aspect-2.0, so with
`resolution=None` the `ActionTransformPipeline` snaps it to the closest `"256"`
tier canvas — 16:9 → **320×192 (w×h) = 192×320 (h×w)** — by aspect-preserving
resize + bottom reflection pad. The training prompt therefore reads
`"...is of 192x320 resolution."`. Keep this; the eval server reproduces the same
snap (see §4).

## 2. Train (1 node, 8 GPUs)

```bash
export LD_LIBRARY_PATH=''                      # NGC/PyTorch container: avoid torch._C import error
export LIBERO_ROOT=/path/to/libero_10_lerobot  # libero_10 conversion ONLY
export BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir>
export WAN_VAE_PATH=<Wan2.2_VAE.pth>
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output_root

bash examples/launch_sft_action_policy_libero.sh
```

Or drive `cosmos_framework.scripts.train` directly:

```bash
torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_policy_libero_repro.toml
```

Recipe knobs live in the registered `action_policy_libero_nano` experiment (full
SFT of the generation + action heads at lr 5e-5 with a 5× LR multiplier on the
action bridge, FusedAdam, selective activation checkpointing, `quantile_rot`
actions, action heads init fresh from the base via `keys_to_skip_loading`). The
TOML sets only run-level scalars: DP=8, `max_iter=10000`, `warm_up_steps=2000`,
`grad_accum_iter=2`, `save_iter=1000`. Checkpoint 2000 is the reference. On
lower-memory GPUs reduce the per-rank batch:
`--opts dataloader_train.max_samples_per_batch=32`.

## 3. Closed-loop eval

Start the policy server on the trained checkpoint, then run the LIBERO
simulator client against it.

```bash
# Server (loads the checkpoint, denormalizes actions). Use quantile_rot + the
# bundled libero rot6d stats so denormalization matches training. See
# `python -m cosmos_framework.scripts.action_policy_server_libero --help`.
python -m cosmos_framework.scripts.action_policy_server_libero \
  --port 8000 \
  --action-normalization quantile_rot \
  --action-stats-path cosmos_framework/data/vfm/action/datasets/stats/libero_native_frame_wise_relative_rot6d.json \
  --raw-action-dim 10 \
  <checkpoint / experiment flags>

# Client (LIBERO sim; runs in a LIBERO + robosuite + mujoco env, separate from
# the training install). Match the training view (concat agentview + wrist):
PYTHONPATH=. python cosmos_framework/simulation/libero/closed_loop_eval.py \
  --server_url http://localhost:8000 \
  --task_suite libero_10 \
  --num_trials_per_task 10 \
  --action_horizon 16 \
  --camera agentview,wrist \
  --image_size 256 \
  --action_space frame_wise_relative --rotation_space 6d --action_dim 10 \
  --save_gifs --gif_fps 20 \
  --output_dir results/libero_closed_loop_10
```

## 4. Gotchas (from NVIDIA/cosmos-framework#50)

These cost real accuracy if missed; the shipped eval client already handles the
first two, but verify them against your checkpoint:

- **Train ↔ serve parity (resolution + prompt).** Training snaps the 256×512
  concat to a **192×320** model-input canvas (see §1) and the prompt suffix
  encodes that resolution + clip duration (`append_resolution_info` /
  `append_duration_fps_timestamps`). The server applies the *same* snap
  (`get_vision_data_resolution` + `find_closest_target_size` + reflection pad),
  so parity is automatic **as long as the client sends the same 2:1 concat
  layout** — run `closed_loop_eval` with `--camera agentview,wrist --image_size
  256` (agentview left, wrist right, matching training). A single-view client (or
  an old server that skipped the snap) sends a different aspect → different
  canvas → the reported 192×320-train vs 256×512-serve mismatch and ~62% (vs
  ~97%). This is the first thing to check if numbers are low. Note the clip
  *duration* string is computed slightly differently on each side (training's
  rounds to `0.0s`); resolution is the dominant factor — verify both against a
  `--dump_dir` server capture if accuracy is off.
- **Gripper.** The model emits gripper in `[0, 1]`; the LIBERO env wants
  `[-1, 1]` with negative = open. `closed_loop_eval._remap_gripper_to_neg1_pos1`
  applies `1 - 2·g`. If the gripper never opens, the sign is inverted for your
  data — flip it.
- **Image orientation.** Sim frames are rotated 180° relative to training;
  `closed_loop_eval` rotates them back (`img[::-1, ::-1]`).
- **Normalization.** Always start the server with `--action-normalization
  quantile_rot` and the bundled libero rot6d stats file, or actions come out at
  the wrong scale.

## 5. FPS & stats

`LIBEROLeRobotDataset` follows `DROIDLeRobotDataset`: it reads the LeRobot parquet
directly, windows by **frame index**, and decodes video at each frame's **real
timestamp** — so it never builds LeRobot's `delta_timestamps` grid and works at
any native FPS. (The earlier `delta_timestamps` port failed on the 10 FPS public
dataset because a 1/20 s grid doesn't land on 10 FPS frames.)

- **Public `lerobot/libero_*` = 10 FPS.** NVIDIA's internal conversion is 20 FPS,
  which the bundled `quantile_rot` stats were computed on. There is no published
  20 FPS LeRobot LIBERO dataset (the LIBERO sim is ~20 Hz, so one is producible
  from the raw HDF5 via `huggingface/lerobot-libero`).
- **What this means:** at 10 FPS each stored per-frame action delta covers 2× the
  motion of a 20 FPS step, so the 20 FPS `quantile_rot` stats under-scale 10 FPS
  actions. For a faithful Table-20 reproduction, either (a) use a 20 FPS LIBERO
  conversion, or (b) recompute `quantile_rot` stats on the 10 FPS data and bundle
  them via `action_stats_path`.
- `fps` in the recipe only sets `conditioning_fps` and the prompt duration; it
  does not change which frames are sampled.

# Cosmos3-Nano LIBERO-10 Action-Policy SFT — Run Report

**Repo:** `fwd4/cosmos-framework` · **Branch:** `haolia/libero-policy-sft` · **Tip:** `db69857`
**Cluster:** Lepton CDG (`neb-cdg-lepton-001-trrjqlvo`), 8×H200 nodes · NFS `/fwd4` → `/workspace`
**Date:** 2026-06-24 · **Table-20 reference:** 97.4% @ ckpt 2000

## 1. Inputs / Data (NFS paths, as resolved inside jobs)

| Asset | Path |
|---|---|
| Base Cosmos3-Nano DCP | `/workspace/cosmos3_action_runs/cosmos3_nano_dcp` |
| Wan2.2 VAE | `/workspace/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth` |
| Dataset (20 fps, libero_10) | `/workspace/libero_datasets/LIBERO_LeRobot_v3/libero_10` |

**Dataset:** `nvidia/LIBERO_LeRobot_v3`, **libero_10 suite only** (full 4-suite mix dilutes to ~82%;
libero_10-alone ≈ 2.7 passes in 2k steps). Native **20 fps**. Loader: `LIBEROLeRobotDataset` —
`frame_wise_relative` rot6d actions (10D = pos3 + rot6d6 + grip1), `quantile_rot` normalization,
`concat_view` (agentview + wrist, 256×256 each → 256×512, snapped to 192×320 model canvas via
resize + reflect-pad), gripper `[0,1]`.

## 2. Recipe (shared by all runs)

Experiment `action_policy_libero_nano` + TOML `action_policy_libero_repro*.toml`. Full SFT (no LoRA),
trains generation + action heads; action heads init fresh from base (`keys_to_skip_loading` =
`net_ema.`, `action2llm`, `llm2action`, `action_modality_embed`, `action_pos_embed`).

| Knob | Value |
|---|---|
| Optimizer | FusedAdamW, lr `5e-5`, betas (0.9, 0.99), eps `1e-8`, wd `0.05` |
| LR multipliers | `5×` on `action2llm` / `llm2action` / `action_modality_embed` |
| LR schedule | LambdaLinear, `cycle_lengths=[10000]`, `warm_up_steps=[2000]` |
| Grad clip | global-norm `1.0` |
| Precision | bf16 compute, **fp32** grad-reduce + master weights (GradScaler off) |
| Packing | `max_num_tokens_after_packing=74000`, selective act-ckpt |
| Batch | `max_samples_per_batch=128` × shard8 × grad_accum2 = **global 2048** |

## 3. Runs

### run20 — headline (loss_scale=10, action_loss_weight=10)

- **Training:** single node, 0→2000 iters, ckpt @ 1000 & 2000. wandb `cosmos3_libero_repro/libero_20fps_run1`.
- **Checkpoints:**
  - `…/libero_repro_20fps/cosmos3_libero_repro/action_sft/libero_20fps_run1/checkpoints/iter_000001000`
  - `…/iter_000002000`  ← **headline**

| Eval | Ckpt | Envs | Episodes | Success |
|---|---|---|---|---|
| **Headline** | iter_2000 | 8 | 500 (10×50) | **473/500 = 94.6%** |
| Full | iter_1000 | 16 | 500 (10×50) | 199/500 = 39.8% |
| Spot | iter_1000 | 8 | 50 (10×5) | ~33/50 = 66% |
| (vec25) | iter_1000 | 25 | — | broken (batch too slow → client timeout, all `steps=10`) |

**Per-task SR (iter_2000):** T0 88 · T1 98 · T2 96 · T3 94 · T4 100 · T5 92 · T6 98 · T7 98 · T8 90 · T9 92
→ **94.6%** (vs 97.4% ref; 8/10 tasks ≥92%).

**Per-task SR (iter_1000, envs=16):** T0 40 · T1 62 · T2 48 · T3 18 · T4 60 · T5 50 · T6 58 · T7 42 · T8 4 · T9 16
→ **39.8%** (mid-warmup, far from converged & high-variance).

### ls1 — A/B (loss_scale=1, action_loss_weight=1) — IN PROGRESS (queueing)

- **Only change vs run20:** global 10× loss scale removed (relative vision:action balance unchanged at
  1:1). Tests whether the 10× ever did anything but saturate grad-clip; expected to land near 94.6%.
- **Config:** single node, replicate=1/grad_accum=2 → same **global 2048**, identical LR schedule.
  2000 iters, **ckpt every 500**.
- **Job:** `oss-libero-ls1-1n-55fx` · **wandb** `cosmos3_libero_repro/libero_20fps_ls1`.
- **Output root:** `/workspace/cosmos-framework/outputs/libero_repro_20fps_ls1` → checkpoints at
  `…/cosmos3_libero_repro/action_sft/libero_20fps_ls1/checkpoints/iter_000000500` (then 1000/1500/2000).
- **Status:** Queueing for a free 8-GPU node (no results yet).
- *2-node HSDP variant (replicate=2/grad_accum=1, same 2048, ~½ wall-clock) committed in comments;
  not used since two full nodes weren't co-schedulable.*

| Eval | Ckpt | Envs | Episodes | Success |
|---|---|---|---|---|
| (pending) | iter_000000500 | — | — | — |
| (pending) | iter_000001000 | — | — | — |
| (pending) | iter_000001500 | — | — | — |
| (pending) | iter_000002000 | — | — | — |

## 4. Eval methodology

HTTP policy server (`action_policy_server_libero.py`: UniPC, 30 steps, guidance 1.0,
`action_chunk_size=16`, `raw_action_dim=10`, `quantile_rot` + bundled rot6d stats) + LIBERO sim client
(`closed_loop_eval.py`). **Fast path:** `SubprocVectorEnv` (N parallel envs) + batched `/predict_batch`
(one diffusion forward of N), ~5× faster (~16 s/episode). **Eval-env pins:** `robosuite==1.4.1`,
`mujoco==2.3.7`, `torch<2.6`, `MUJOCO_GL=egl`, `NVIDIA_DRIVER_CAPABILITIES=all`, per-worker EGL device
pin, `--gripper_mode zero_one`.

**Vectorization finding:** num_envs **8 and 16 work**; **25 breaks** (6–10 min/batch → client timeout).
16 is the safe upper bound on one H200.

**Eval outputs (gifs + `summary.json`):** `…/outputs/eval_run20_2000_vec` (headline),
`eval_run20_1000_vec`, `eval_run20_1000`, `eval_run20_vecval`.

## 5. Committed artifacts (branch `haolia/libero-policy-sft`)

| Commit | Content |
|---|---|
| `db69857` | ls1 TOML → single-node topology |
| `8daf13e` | ls1 A/B recipe: `action_policy_libero_repro_ls1.toml` + `launch_sft_action_policy_libero_ls1.sh` |
| `5864093` | vec eval: per-worker EGL pin |
| `713b80f` | closed_loop_eval: per-task env.close (EGL leak fix) |
| `4041ac2` | batched `/predict_batch` + vectorized client |
| `1c31e3a` | recipe batch config 128/ga2/74000 |
| `9282e11` | `--gripper_mode` |

---

**Bottom line:** LIBERO-10 reproduced at **94.6%** (vs 97.4% ref) on the canonical 20 fps setup; the
loss_scale=1 A/B is committed and queued to confirm the global-scale-is-clipping hypothesis.

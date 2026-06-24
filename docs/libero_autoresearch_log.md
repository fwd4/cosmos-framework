# LIBERO-10 SFT autoresearch log

**Goal:** reach **97.4%** overall SR (Table-20) on `libero_10`, 10×50 (500-episode) sharded eval.
**Baseline to beat:** run20 = 94.6% (batch 2048, policy). **Gap to target: +2.8 pts.**

## Method (feedback loop)
1. Pick next config from the frontier (best-so-far), varying one factor (guided, not random).
2. Train on reservation `cosmos3-sim-eval` (HSDP, 2000 steps unless noted, ckpt every 500).
3. Sharded eval (5 jobs × 2 tasks, num_envs=8, continuous `1−2g` gripper) on the BEST checkpoint
   AND a checkpoint sweep (500/1000/1500/2000) when promising.
4. Record per-task + overall SR here; mark verdict; propose next.
5. Stop when overall ≥ 97.4% (500-ep) or the round budget is exhausted.

## Fixed/known-correct (do not re-test)
- Dataset = NVIDIA `LIBERO_LeRobot_v3` libero_10, 20 fps, gripper `[0,1]` (verified vs community 10fps `{-1,+1}`).
- mode default = `policy`; loss 10/10; lr 5e-5; cycle=10000/warmup=2000; concat_view; quantile_rot rot6d stats.
- `encode_exact_durations=[17,61,73]`, `max_num_tokens_after_packing=74000` (reference values).
- Eval caveats handled: rotate_180=True, gripper `1−2g`, server quantile_rot + rot6d stats, prompt parity.

## Experiment table

| # | Config (Δ vs baseline) | Train job | Eval (best ckpt) | Per-task | Overall | Verdict |
|---|---|---|---|---|---|---|
| 1 | run20: batch 2048, policy (reference-faithful) | run20c | iter_2000, envs8 | T0 88·T1 98·T2 96·T3 94·T4 100·T5 92·T6 98·T7 98·T8 90·T9 92 | **94.6%** | baseline; +2.8 to target |
| 2 | b128: batch 128 + `[17]` + cap `-1` (BUGGED) | b128 | iter_2000, envs16 | T8/T9=0 | 60.8% | regressed (my bad edits) |
| 3 | b128v2: batch 128, policy, durations/cap reverted | b128v2-2dc8 | iter_2000 ⏳ | — | ⏳ | tests durations/cap (loss == b128 ⇒ expect ~60%) |
| 4 | b128joint: batch 128, **mode=joint** | b128joint-66f7 | iter_2000 ⏳ | — | ⏳ | tests joint @ small batch |

## Candidate knobs (priority order)
**Eval-side (cheap — no retrain):**
- E1. Checkpoint sweep of run20 (500/1000/1500/2000) — we only evaled iter_2000; a later/earlier ckpt may be higher.
- E2. More trials (10×50 already; bump seeds / 10×100) to tighten the estimate near 94–97%.
- E3. Diffusion steps (30→50) / guidance at inference.

**Training-side (expensive — retrain):**
- T1. More steps / passes (run20 to 3000–4000 iters).
- T2. EMA: `load_ema_to_reg=True` (eval EMA weights) vs current reg.
- T3. lr sweep {3e-5, 5e-5, 1e-4} and warmup.
- T4. batch size sweep (128 / 512 / 2048) × mode (policy/joint).
- T5. action_loss_weight / loss_scale ratio.

## Decisions / notes
- (pending rows 3–4 results before launching the loop)

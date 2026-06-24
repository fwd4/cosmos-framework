# LIBERO-10 SFT autoresearch log

**Goal:** reach **97.4%** overall SR (Table-20) on `libero_10`, 10×50 (500-episode) sharded eval.
**HARD CONSTRAINT: global batch = 128** (≈2.7 passes / 2000 steps — the reporter's regime).
run20's 94.6% was at batch 2048 and is therefore NOT an admissible solution — it's only a
reference ceiling. The task: find the batch-128 config that reaches 97.4%.

**Mode of operation:** fully autonomous, budget ≈ 8–10 training runs; ping user on new-best /
target-hit / budget-exhausted. All runs hold global batch = 128.

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

## Candidate knobs (priority order) — ALL at fixed global batch = 128
**Eval-side (cheap — no retrain; test on existing batch-128 ckpts):**
- E1. **EMA-eval**: load `net_ema.*` (use_ema_weights) instead of reg net. Checkpoints already store EMA;
  often +1–3% SR. Re-eval best of {b128v2, b128joint} — NO retrain. Highest cheap-win.
- E2. Checkpoint sweep 500/1000/1500/2000 (b128v2/joint save every 500) — pick best iter.
- E3. Diffusion steps 30→50 / guidance at inference.
- E4. More eval trials / seeds to tighten the estimate near target.

**Training-side (retrain, batch=128):**
- T1. mode = joint vs policy (row 4 tests this).
- T2. More steps / passes (2000 → 4000 / 6000) — more passes at batch 128.
- T3. lr sweep {3e-5, 5e-5, 1e-4} + warmup fraction.
- T4. action_loss_weight / loss_scale ratio.
- T5. EMA rate / power-EMA schedule (if EMA-eval helps).
- (batch size is FIXED at 128 — not a knob.)

## Decisions / notes
- Batch is fixed at 128. Frontier seeds = rows 3 (b128v2 policy) + 4 (b128joint). Loop starts from
  the better of those once both report; first move = E1 (EMA-eval, free) on the frontier ckpts.

# Cosmos3-Nano LIBERO-10 Action-Policy SFT — Reproduction Report

**Goal:** reproduce Table-20 LIBERO-10 (**97.4%**, released recipe 97.5%) for Cosmos3-Nano
action-policy SFT on the public GA base.

**Eval protocol (all numbers below):** closed-loop LIBERO sim, `libero_10`, **10 tasks × 50
trials = 500 episodes**, `seed 0`, vectorized sim (`num_envs=8`) + batched serving. Harness
verified **i4-faithful** (identical `rotate_180`, `TASK_MAX_STEPS[libero_10]=520`, full-chunk
horizon, guidance 1.0, gripper `1−2g`, quantile_rot rot6d stats). Vectorization **vindicated**:
serial (`num_envs=1`) 92.6% ≈ vec (`num_envs=8`) 93.6% on the reference checkpoint.

## Headline findings

1. **Best achieved: 95.2%** — tied by `lib10_lr5e5 @ iter_1500` and `lr1e4c10k @ iter_1000`
   (≈2.2 pt short of 97.4%).
2. **~95% is the real ceiling under this (i4-faithful) harness** — proven independently: the
   **reference's own downloaded checkpoints peak at 94.6%** (iter_8000), trajectory
   93.6→**94.6**→94.0. The 97.4/97.5% headline **does not reproduce at face value** here.
3. **LR schedule drives stability, not the ceiling.** lr 1e-4 peaks fast then **over-fits task 0
   (90→70)** and regresses; the gentle reference LR (5e-5, wu500, cyc16k) **keeps T0 healthy
   (88–92) throughout** → recommended recipe.
4. **4-suite curbs the T0 over-fit** (T0=94 @ 2k) but dilutes libero_10 (~1/4 exposure; still
   climbing, 87% @ 2k).
5. **Bug fixed:** the OSS LIBERO dataset dropped i4's resample-on-decode-failure guard → one
   corrupt packed-mp4 frame crashed multi-node runs. Restored (i4-parity).

## Recommended recipe — `action_policy_libero_repro.toml`
libero_10-only · lr 5e-5 · warmup 500 · cycle 16000 · global batch 2048 · save every 500 →
**~95.2% @ iter_1500**, T0 stable.

## Full results (libero_10, 500 ep, seed 0)

### A. libero_10-only, public GA base

| Run · config | iter | SR | T0·T1·T2·T3·T4·T5·T6·T7·T8·T9 |
|---|---|---|---|
| **lib10_lr5e5** · lr5e-5, wu500, cyc16k | 500 | 65.2% | 54·82·96·54·84·64·88·92·18·20 |
| | 1000 | 92.8% | 88·98·100·94·98·90·96·100·84·80 |
| | **1500** | **95.2%** | 92·98·100·94·92·92·94·100·98·92 |
| | 2000 | 94.0% | 88·98·92·96·96·92·96·96·98·88 |
| **lr1e4c10k** · lr1e-4, wu100, cyc10k | 1000 | **95.2%** | 90·98·94·88·96·92·98·100·100·96 |
| | 2000 | 93.0% | 70·98·96·98·92·92·94·100·94·96 |
| run20 · lr5e-5, wu2000, cyc10k | 2000 | 94.6% | 88·98·96·94·100·92·98·98·90·92 |
| b128long · gbs128, lr5e-5 | 2k→10k | 63.6→79.6→89.6→90.4→**92.6** | (10k) 68·98·96·98·94·92·90·100·96·94 |
| b128v2 · gbs128 | 2000 | 63.6% | 34·90·86·50·86·90·96·88·12·4 |
| b128joint · gbs128, mode=joint | 2000 | 28.8% | 30·60·34·36·26·18·32·50·0·2 |

Overall-only: lr2e4-g8x8 iter_2000 cross-seed 91.8/93.4/93.6/93.8 (~93.2%) · lr1e4_2k (cyc2000) ~92.4% · lr5e4 ~92% peak then diverged.

### B. 4-suite run (public GA base, lr5e-5, wu500, cyc16k)

| iter | SR | T0·T1·T2·T3·T4·T5·T6·T7·T8·T9 |
|---|---|---|
| 500 | 34.0% | 20·90·48·16·32·36·64·34·0·0 |
| 1000 | 61.0% | 62·84·96·44·82·54·82·74·14·18 |
| 1500 | 72.6% | 82·92·94·66·82·88·92·70·28·32 |
| 2000 | 87.0% | 94·100·100·90·84·90·94·98·54·66 |

### C. Reference checkpoint (internal `exp506` midtrain base, 4-suite) — downloaded, our harness

| iter | SR | T0·T1·T2·T3·T4·T5·T6·T7·T8·T9 |
|---|---|---|
| 2000 (vec) | 93.6% | 92·100·92·100·96·92·86·100·90·88 |
| 2000 (serial num_envs=1) | 92.6% | 90·98·98·90·90·92·90·100·94·84 |
| **8000** (peak) | **94.6%** | 78·100·96·96·96·92·94·100·94·100 |
| 15000 (final) | 94.0% | 66·100·98·96·98·92·92·100·98·100 |

## Conclusion
On the public GA base with an i4-faithful eval harness, LIBERO-10 reproduces to **~95%**
(best 95.2%), not 97.4%. The reference's own checkpoints also top out at ~94.6% under this
harness, indicating the 97.4/97.5% headline reflects a different eval protocol and/or the
internal mid-training base — not an OSS bug. The **gentle-LR recipe (lr5e-5, wu500, cyc16k)
is recommended**: same peak as higher-lr, but no task-0 over-fit collapse.

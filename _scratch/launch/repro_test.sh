#!/bin/bash
# 1-node 8xH200 test: dryrun + 30-iter train at res480, max_samples_per_batch=64.
source /tmp/cf/_scratch/launch/_common.sh
export IMAGINAIRE_OUTPUT_ROOT=/fwd4/cosmos3_action_runs/repro_test
LOGD=/fwd4/cosmos3_action_runs/repro_test_log; rm -rf "$LOGD"; mkdir -p "$LOGD"

echo "===== PHASE1 DRYRUN (res480, batch ${MAX_SAMPLES:-64}) ====="
torchrun --nproc_per_node=8 --standalone -m cosmos_framework.scripts.train \
  --sft-toml="$TOML" --dryrun -- $OPTS > /tmp/dry.log 2>&1
echo "DRYRUN_EXIT=$?"; tail -30 /tmp/dry.log

echo "===== PHASE2 TRAIN 30 iters ====="
torchrun --nproc_per_node=8 --standalone --tee 3 --redirects 3 --log-dir "$LOGD" \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- $OPTS trainer.max_iter=30
echo "TRAIN_EXIT=$?"
echo "=== rank0 stderr tail ==="; tail -90 "$LOGD"/*/attempt_0/0/stderr.log 2>/dev/null
echo "ALL_DONE"
echo "===== KEEPALIVE 900s (stream the log before purge) ====="
sleep 900

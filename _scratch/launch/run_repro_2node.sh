#!/bin/bash
# ============================================================================
# Self-contained 2-node / 16-rank DROID action-SFT test (HSDP shard 8 x repl 2).
# Launch cmd only clones + runs this:
#   git clone -b <branch> <url> /tmp/cf && bash /tmp/cf/_scratch/launch/run_repro_2node.sh
# Rendezvous via Lepton multi-worker env (confirmed via mw-probe3). res480,
# max_samples_per_batch=32 (validated on 1 node), 30-iter train.
# ============================================================================
set +e
export DEBIAN_FRONTEND=noninteractive
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "########## [0/4] system deps ##########"
apt-get update -qq && apt-get install -y -qq curl ffmpeg >/tmp/apt.log 2>&1

echo "########## [1/4] uv + venv (cu130-train) ##########"
cd /tmp/cf || { echo "NO_REPO"; exit 1; }
echo "REPO head=$(git rev-parse --short HEAD 2>/dev/null)"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh >/tmp/uv.log 2>&1; source "$HOME/.local/bin/env" 2>/dev/null; }
uv --version || echo "UV_MISSING"
for att in 1 2 3 4 5 6 7 8; do
  echo "uv sync attempt $att"
  uv sync --all-extras --group=cu130-train >/tmp/sync.log 2>&1 && { echo "UV_SYNC_OK"; break; } || { echo "sync $att failed:"; tail -8 /tmp/sync.log; sleep 15; }
done
VENV=/tmp/cf/.venv
source "$VENV/bin/activate" && export LD_LIBRARY_PATH=
TORCHRUN="$VENV/bin/torchrun"
if [ ! -x "$TORCHRUN" ]; then echo "FATAL_VENV_INCOMPLETE"; tail -40 /tmp/sync.log; echo "ALL_DONE_NODE_${LEPTON_JOB_WORKER_INDEX:-0}"; sleep 300; exit 1; fi
"$VENV/bin/python" -c "import torch,loguru;print('IMPORTS_OK torch',torch.__version__,'devs',torch.cuda.device_count())" 2>&1 | tail -2

echo "########## [2/4] env + data + HSDP parallelism ##########"
export HF_HOME=/fwd4/.cache/huggingface
export WAN_VAE_PATH=$(find /fwd4/.cache/huggingface /fwd4/cosmos3_action_runs -name Wan2.2_VAE.pth 2>/dev/null | head -1)
export DROID_ROOT=/fwd4/droid_plus_lerobot_640x360_20260412/success
export BASE_CHECKPOINT_PATH=/fwd4/cosmos3_action_runs/cosmos3_nano_dcp
export IMAGINAIRE_OUTPUT_ROOT=/fwd4/cosmos3_action_runs/repro_2node
TOML=examples/toml/sft_config/action_policy_droid_repro.toml
# HSDP: shard 8 (intra-node) x replicate 2 (across 2 nodes) = 16 ranks
sed -i 's/data_parallel_replicate_degree = 1/data_parallel_replicate_degree = 2/' "$TOML"
echo "PARALLELISM_$(grep -E 'replicate_degree|shard_degree' "$TOML" | tr '\n' '_')"
OPTS="dataloader_train.max_samples_per_batch=32 job.wandb_mode=disabled"

echo "########## [3/4] rendezvous (Lepton multi-worker env) ##########"
NNODES="${LEPTON_JOB_TOTAL_WORKERS:-2}"
NRANK="${LEPTON_JOB_WORKER_INDEX:-0}"
MASTER="${LEPTON_JOB_NAME}-0.${LEPTON_SUBDOMAIN}.ws-${LEPTON_WORKSPACE_ID}.svc.cluster.local"
echo "RDZV nnodes=$NNODES node_rank=$NRANK master=$MASTER"
getent hosts "$MASTER" && echo "MASTER_DNS_OK" || echo "MASTER_DNS_FAIL"
LOGD=/fwd4/cosmos3_action_runs/repro_2node_log_${NRANK}; rm -rf "$LOGD"; mkdir -p "$LOGD"

echo "########## [4/4] 2-NODE TRAIN 30 iters (16 ranks) ##########"
"$TORCHRUN" --nnodes="$NNODES" --node_rank="$NRANK" --nproc_per_node=8 \
  --master_addr="$MASTER" --master_port=29500 \
  --tee 3 --redirects 3 --log-dir "$LOGD" \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- $OPTS trainer.max_iter=30
echo "TRAIN_EXIT=$?_NODE_${NRANK}"
echo "=== local rank0 stderr tail ==="; tail -70 "$LOGD"/*/attempt_0/0/stderr.log 2>/dev/null
echo "ALL_DONE_NODE_${NRANK}"
echo "########## KEEPALIVE 600s ##########"
sleep 600

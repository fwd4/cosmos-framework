#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# 2-node NCCL all-reduce bandwidth + transport probe: tells us whether inter-node
# uses IB (NET/IB, GDRDMA) or falls back to sockets/host (slow ~GB/s). System torch.
set +e
cd /tmp/cf || exit 1
echo "===IB devices==="; ls /dev/infiniband 2>/dev/null; ibstat 2>/dev/null | grep -iE "State|Rate" | head
NNODES="${LEPTON_JOB_TOTAL_WORKERS:-2}"; NRANK="${LEPTON_JOB_WORKER_INDEX:-0}"
MASTER="${LEPTON_JOB_NAME}-0.${LEPTON_SUBDOMAIN}.ws-${LEPTON_WORKSPACE_ID}.svc.cluster.local"
if [ "$NRANK" != "0" ]; then
  for i in $(seq 1 180); do
    getent hosts "$MASTER" >/dev/null 2>&1 && (exec 3<>/dev/tcp/"$MASTER"/29500) 2>/dev/null && { exec 3>&- 3<&-; echo MASTER_REACHABLE; break; }
    sleep 5
  done
fi
echo "===NCCL 2-node bench (node_rank=$NRANK)==="
NCCL_DEBUG=INFO torchrun --nnodes="$NNODES" --node_rank="$NRANK" --nproc_per_node=8 \
  --master_addr="$MASTER" --master_port=29500 _scratch/launch/nccl_bench.py 2>&1 | tee /tmp/b.log \
  | grep -aE 'NET/IB|NET/Socket|GDRDMA|GPU Direct RDMA|Using network|NCCL_IB|via NET|mlx|ib[0-9]' | head -25
echo "--- ALLREDUCE result (node_rank=$NRANK) ---"
grep -aE 'ALLREDUCE|NCCL_BENCH_DONE' /tmp/b.log
echo "DIAG2_DONE_${NRANK}"
sleep 300

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# 1-node NVLink/NCCL diagnostic: GPU topology, NVLink link status, and NCCL
# all-reduce bandwidth (system torch — no uv sync). Run via: clone repo + bash this.
set +e
echo "===TOPO==="
nvidia-smi topo -m
echo "===NVLINK==="
nvidia-smi nvlink --status 2>&1 | head -60
echo "===NCCLBENCH==="
cd /tmp/cf || exit 1
NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=8 _scratch/launch/nccl_bench.py 2>&1 \
  | grep -aE 'ALLREDUCE|NCCL_BENCH_DONE|via NVLink|via P2P|via SHM|via NET|NVLS|Could not|Error|Channel' | head -40
echo "DIAG_DONE"
sleep 600

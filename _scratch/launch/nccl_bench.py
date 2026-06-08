# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tiny NCCL all-reduce bandwidth probe (run via torchrun). Reports algbw/busbw
per message size so we can tell whether GPU<->GPU comms is on NVLink
(~hundreds of GB/s) or falling back to a slow path (~single GB/s)."""
import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
torch.cuda.set_device(local)
dev = torch.device("cuda", local)

for nbytes in (256 << 20, 1 << 30):  # 256 MiB, 1 GiB
    n = nbytes // 4
    x = torch.ones(n, dtype=torch.float32, device=dev)
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    iters = 20
    t = time.time()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    sz = n * 4 / 1e9
    algbw = sz / dt
    busbw = 2 * (world - 1) / world * algbw
    if rank == 0:
        print(
            f"ALLREDUCE world={world} size={sz:.2f}GB time={dt * 1000:.1f}ms "
            f"algbw={algbw:.1f}GB/s busbw={busbw:.1f}GB/s",
            flush=True,
        )
dist.destroy_process_group()
if rank == 0:
    print("NCCL_BENCH_DONE", flush=True)

---
title: Examples
description: Runnable code examples for every cuda.zig feature.
---

# Examples

Each example is a standalone Zig program in the [`examples/`](https://github.com/muhammad-fiaz/cuda.zig/tree/main/examples) directory. Run any of them with:

```sh
zig build example-01   # device info
zig build example-02   # memory transfer
# ... etc
```

| # | Name | What it demonstrates |
|---|---|---|
| [01](/examples/01-device-info) | Device Info | Enumerate GPUs and read hardware properties |
| [02](/examples/02-memory-transfer) | Memory Transfer | H2D, D2H, async copies, timing |
| [03](/examples/03-kernel-launch) | Kernel Launch | Load PTX, launch with grid/block config |
| [04](/examples/04-streams-events) | Streams & Events | Async pipelines, event timing |
| [05](/examples/05-tensor-ops) | Tensor Operations | Elementwise, reduction, matmul |
| [06](/examples/06-multi-gpu) | Multi-GPU | Peer access, cross-device copy |
| [07](/examples/07-cpu-fallback) | CPU Fallback | Run on host without a GPU |
| [08](/examples/08-managed-memory) | Managed Memory | Unified Memory, prefetch, advise |
| [09](/examples/09-nvrtc-compilation) | NVRTC Compilation | Runtime PTX compilation and launch |
| [10](/examples/10-memory-pools-pitched) | Memory Pools & 2D Pitched | Stream-ordered pool allocations & 2D pitched memory |
| [11](/examples/11-occupancy-profiler) | Occupancy & Profiler | Occupancy calculator, stream priority & profiler markers |

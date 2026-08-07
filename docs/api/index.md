---
title: API Overview
description: Index of all modules, abstractions, and FFI layers provided by cuda.zig.
---

# API Overview

`cuda.zig` provides a layered architecture: high-level typed abstractions built on top of low-level runtime and driver bindings, all resolving dynamically with zero link-time dependencies.

## Top-Level Namespaces

| Module | Description |
|--------|-------------|
| [`cuda.device`](/api/device) | GPU enumeration, properties, selection, and P2P peer access |
| [`cuda.memory`](/api/memory) | Typed buffers (`DeviceBuffer`, `PinnedBuffer`, `UnifiedBuffer`, `PoolBuffer`) and raw allocators |
| [`cuda.stream`](/api/stream) | Asynchronous execution streams, priorities, and events |
| [`cuda.kernel`](/api/kernel) | Driver kernel launch, `LaunchConfig`, modules, and `cuda.occupancy` |
| [`cuda.tensor`](/api/tensor) | High-level `Tensor(T)` with cuBLAS & CPU fallback matrix operations |
| [`cuda.nvrtc`](/api/nvrtc) | Runtime CUDA C++ compilation to PTX via NVRTC |
| [`cuda.fallback`](/api/fallback) | Automatic CPU fallback execution engine |
| [`cuda.profiler`](/api/core) | Profiler session markers (`start`, `stop`, `ProfilerGuard`) |
| [`cuda.version`](/api/core) | Driver & Runtime version detection and capability flags |

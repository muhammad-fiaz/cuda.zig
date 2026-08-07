---
layout: home
title: "NVIDIA CUDA Bindings & GPU Programming for Zig | CUDA.zig"
description: >
  Production-grade CUDA Runtime and Driver API for Zig. Zero link-time dependencies,
  automatic CPU fallback, Tensor operations, NVRTC, CUDA Graphs, Memory Pools, Occupancy,
  multi-GPU peer access, and 40×+ GPU speedup benchmarks.

hero:
  name: cuda.zig
  text: GPU Computing for Zig
  tagline: Production-grade CUDA Runtime API with zero link-time dependencies, automatic CPU fallback, memory pools, occupancy profiling, and full Zig 0.16 support.
  image:
    src: /favicon.png
    alt: cuda.zig logo
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/muhammad-fiaz/cuda.zig

features:
  - icon: ⚡
    title: Zero Link-Time Dependencies
    details: CUDA is loaded dynamically at runtime. Ship your binary anywhere — CUDA need not be present at compile time.
  - icon: 🔄
    title: Automatic CPU Fallback
    details: No GPU? No problem. Every API surface transparently falls back to host-side execution without a single code change.
  - icon: 🔬
    title: NVRTC Runtime Compilation
    details: Compile CUDA C/C++ kernels to PTX at runtime, cache them, and launch without a separate build step.
  - icon: 🧮
    title: Tensor Operations
    details: High-level Tensor<T> with elementwise ops, reductions, matmul, and optional cuBLAS delegation.
  - icon: 🔀
    title: Streams & Events
    details: Full async streams, event-based synchronisation, elapsed-time measurement, and priority stream creation.
  - icon: 🌐
    title: Multi-GPU Peer Access
    details: Query and enable peer-to-peer device access, perform cross-device memcpy, and orchestrate multi-GPU workloads.
  - icon: 🏊
    title: Memory Pools & Pitched Memory
    details: Stream-ordered allocation via cudaMallocAsync/cudaFreeAsync and 2D pitched memory for maximum coalescing.
  - icon: 📊
    title: Occupancy & Profiler
    details: Query SM occupancy, stream priorities, and attach profiler session markers to measure GPU kernel efficiency.
  - icon: 🚀
    title: 40×+ GPU Speedup
    details: Verified N-body O(N²) benchmark shows 40× or greater parallel acceleration over sequential CPU on RTX hardware.
  - icon: 📦
    title: Related Zig Projects
    details: >
      Part of a growing Zig ecosystem — see the full list in our documentation.
      env.zig, tui.zig, zon.zig, loaders.zig, mcp.zig, args.zig, httpx.zig,
      api.zig, zix, archive.zig, zigx, downloader.zig, updater.zig, num.zig,
      logly.zig, zigantic, buildx.zig, cuda.zig.
---

---
layout: home
title: "NVIDIA CUDA Bindings & GPU Programming for Zig | CUDA.zig"
description: >
  Production-grade CUDA Runtime and Driver API for Zig. Zero link-time dependencies,
  automatic CPU fallback, Tensor operations, NVRTC, CUDA Graphs, and multi-GPU support.

hero:
  name: cuda.zig
  text: GPU Computing for Zig
  tagline: Production-grade CUDA Runtime API with zero link-time dependencies, automatic CPU fallback, and full Zig 0.16 support.
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
    details: Full async streams, event-based synchronisation, and elapsed-time measurement with a clean Zig API.
  - icon: 🌐
    title: Multi-GPU Peer Access
    details: Query and enable peer-to-peer device access, perform cross-device memcpy, and orchestrate multi-GPU workloads.
  - icon: 📦
    title: Related Zig Projects
    details: >
      Part of a growing Zig ecosystem: env.zig (env parsing), tui.zig (TUI support),
      zon.zig (ZON file format), loaders.zig (spinners/loading/progress), mcp.zig (MCP support),
      args.zig (args parsing), httpx.zig (HTTP client/server), api.zig (API framework),
      zix (web framework), archive.zig (archive/compression), zigx (compression file formats),
      downloader.zig (file downloading), updater.zig (auto-updater), num.zig (numerical computing),
      logly.zig (logging), zigantic (data validation/serialization), buildx.zig (build tooling),
      cuda.zig (CUDA/GPU computing).
---

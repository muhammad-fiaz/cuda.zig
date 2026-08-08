---
title: Version Compatibility
description: CUDA Toolkit version compatibility, minimum compute capabilities, and OS support for cuda.zig.
---

# Version Compatibility

## Zig Compiler

| cuda.zig | Minimum Zig |
|---|---|
| 0.0.x | 0.16.0 |

## CUDA Toolkit & Driver

cuda.zig dynamically probes and resolves CUDA symbols at runtime with zero link-time dependencies. The table below outlines supported toolkit and driver versions:

| cuda.zig | Supported CUDA Versions | Minimum Driver | cuDNN |
|---|---|---|---|
| 0.0.2 | CUDA 11.0 – 13.4 (Major 11, 12, 13; Minor 0–9 probe loop) | 525.60+ | Optional |

**Latest validated environment:** CUDA Toolkit 13.3 Update 1 (`nvidia/cuda:13.3.1-cudnn-devel`), Driver 572.16 (Windows 11 x86_64, RTX 4070 SUPER Compute Capability 8.9).

**Minimum validated environment:** CUDA 12.8.1 with cuDNN (`nvidia/cuda:12.8.1-cudnn-devel`).

## GPU Compute Capability Requirements

| Feature | Minimum Compute Cap | Required CUDA Version | Notes |
|---|---|---|---|
| Basic Memory, Streams, Events | 5.0 (Maxwell) | CUDA 11.0+ | H2D / D2H transfers, event timing |
| 2D Pitched Memory | 5.0 (Maxwell) | CUDA 11.0+ | `mallocPitch`, `memcpy2DAsync` |
| Multi-GPU Peer Access | 5.0 (Maxwell) | CUDA 11.0+ | `canAccessPeer`, `enablePeerAccess` |
| Unified/Managed Memory Prefetch | 6.0 (Pascal) | CUDA 11.0+ | `cudaMemPrefetchAsync`, `cudaMemAdvise` |
| Occupancy Calculator | 6.0 (Pascal) | CUDA 11.0+ | `cuOccupancyMaxActiveBlocksPerMultiprocessor` |
| Stream-Ordered Memory Pools | 7.0 (Volta) | CUDA 11.2+ | `cudaMallocAsync`, `cudaFreeAsync`, `PoolBuffer` |
| Tensor Cores (FP16 / BF16 / TF32) | 7.0 (Volta) | CUDA 11.0+ | Hardware acceleration via cuBLAS / PTX |
| N-Dimensional Tensors (up to 8D) | Any | Any | Pure Zig layout engine + CUDA / CPU fallback |
| NVRTC Runtime Compilation | 5.0 (Maxwell) | CUDA 11.0+ | Automatic probe across versioned DLLs/SOs |

## Operating System Support

CUDA is officially supported on Windows, Linux, and Windows Subsystem for Linux (WSL 2).

| OS Family | Distribution / Version | Status | Notes |
|---|---|---|---|
| **Windows** | Windows 10, 11, Server (64-bit) | ✅ Supported | Dynamic resolution of `nvcuda.dll` & `cudart64_*.dll` |
| **Linux** | Ubuntu, RHEL, CentOS, Fedora, Rocky, AlmaLinux, SUSE | ✅ Supported | Dynamic resolution of `libcuda.so.1` & `libcudart.so` |
| **WSL 2** | Ubuntu & supported distros under WSL 2 | ✅ Supported | Full GPU acceleration via Windows host driver bridge |
| **macOS** | macOS 10.15+ (Intel & Apple Silicon) | 🔶 CPU Fallback Only | Apple/NVIDIA dropped native macOS CUDA driver support |

> [!NOTE]
> On macOS or systems without NVIDIA hardware/drivers, `cuda.zig` transparently routes all memory allocations, tensor computations, and matrix operations through its pure Zig CPU fallback backend without crashing or throwing startup errors.

## Build Flags vs CUDA Libraries

| Build Flag | Default | Required CUDA Version | Target Dynamic Library |
|---|---|---|---|
| `-Denable_cublas=true` | `false` | 12.0+ | `libcublas.so` / `cublas64_12.dll` |
| `-Denable_curand=true` | `false` | 12.0+ | `libcurand.so` / `curand64_10.dll` |
| `-Denable_cusolver=true` | `false` | 12.0+ | `libcusolver.so` / `cusolver64_11.dll` |
| `-Denable_nvrtc=true` | `false` | 11.0+ | `libnvrtc.so` / `nvrtc64_*.dll` |

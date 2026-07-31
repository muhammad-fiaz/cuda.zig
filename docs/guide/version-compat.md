---
title: Version Compatibility
description: CUDA Toolkit version compatibility and minimum requirements for cuda.zig.
---

# Version Compatibility

## Zig

| cuda.zig | Minimum Zig |
|---|---|
| 0.0.x | 0.16.0 |

## CUDA Toolkit

cuda.zig dynamically loads CUDA symbols at runtime. The table below lists the minimum CUDA driver/runtime version required:

| cuda.zig | CUDA Runtime | CUDA Driver | cuDNN |
|---|---|---|---|
| 0.0.x | 12.x – 13.x | 525.60+ | Optional |

**Latest validated version:** CUDA Toolkit 13.3 Update 1 (`nvidia/cuda:13.3.1-cudnn-devel`)

**Minimum validated version:** CUDA 12.8.1 with cuDNN (`nvidia/cuda:12.8.1-cudnn-devel`)

## GPU Compute Capability

| Feature | Minimum compute cap |
|---|---|
| Basic memory, streams, events | 5.0 (Maxwell) |
| Unified/Managed memory prefetch | 6.0 (Pascal) |
| cuBLAS Tensor Cores (FP16/BF16) | 7.0 (Volta) |
| NVRTC all features | 5.0 |

## Operating System Support

CUDA is officially supported on Windows, Linux, and Windows Subsystem for Linux (WSL).

| OS Family | Distribution / Version | Status | Notes |
|---|---|---|---|
| **Windows** | Windows 10, 11, Server (64-bit) | ✅ Supported | Native `nvcuda.dll` dynamic resolution |
| **Linux** | Ubuntu, RHEL, CentOS, Fedora, Rocky, AlmaLinux, SUSE | ✅ Supported | Native `libcuda.so.1` & `libcudart.so` |
| **WSL 2** | Ubuntu & supported distros under WSL 2 | ✅ Supported | Full GPU acceleration via Windows host driver |
| **macOS** | macOS 10.15+ (Intel & Apple Silicon) | 🔶 CPU Fallback Only | Apple/NVIDIA dropped native macOS CUDA support |

> [!NOTE]
> On macOS or systems without NVIDIA hardware/drivers, `cuda.zig` transparently routes all operations through its CPU fallback backend without throwing runtime errors or failing binary startup.

## Feature Flags vs CUDA Versions

| Flag | Required CUDA | Library |
|---|---|---|
| `-Denable_cublas=true` | 12.0+ | `libcublas.so` / `cublas64_12.dll` |
| `-Denable_curand=true` | 12.0+ | `libcurand.so` / `curand64_10.dll` |
| `-Denable_cusolver=true` | 12.0+ | `libcusolver.so` / `cusolver64_11.dll` |
| `-Denable_nvrtc=true` | 11.0+ | `libnvrtc.so` / `nvrtc64_120_0.dll` |

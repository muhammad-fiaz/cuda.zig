---
title: Installation
description: All installation methods for cuda.zig — stable release, nightly, Docker, and manual build options.
---

# Installation

## Requirements

- **Zig 0.16.0** or newer ([download](https://ziglang.org/download/))
- **CUDA Toolkit 12.x – 13.x** *(optional)* — only needed at runtime on a machine with an NVIDIA GPU
- **NVIDIA driver** *(optional)* — for GPU execution

cuda.zig compiles without CUDA present. The dynamic loader resolves `libcuda.so` / `nvcuda.dll` / `libcudart.so` at runtime.

## Stable Release

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .cuda = .{
        .url = "https://github.com/muhammad-fiaz/cuda.zig/archive/refs/tags/v0.0.1.tar.gz",
        .hash = "<run zig fetch to get hash>",
    },
},
```

Or use `zig fetch` to populate the hash automatically:

```sh
zig fetch --save https://github.com/muhammad-fiaz/cuda.zig/archive/refs/tags/v0.0.1.tar.gz
```

## Development / Nightly

Track the `main` branch directly:

```sh
zig fetch --save https://github.com/muhammad-fiaz/cuda.zig/archive/refs/heads/main.tar.gz
```

> [!WARNING]
> The `main` branch may contain breaking changes between commits. Pin a specific commit SHA for reproducible builds.

## Manual Clone

```sh
git clone https://github.com/muhammad-fiaz/cuda.zig
```

Add a local path dependency to your `build.zig.zon`:

```zig
.cuda = .{ .path = "../cuda.zig" },
```

## Build Configuration

`build.zig` exposes optional feature flags:

| Flag | Default | Effect |
|---|---|---|
| `-Denable_cublas=true` | false | Link cuBLAS and use `cublasGemmEx` in `tensor/ops/blas.zig` |
| `-Denable_curand=true` | false | Enable `src/rand/` on-device random number generation |
| `-Denable_cusolver=true` | false | Enable `src/solver/` LU/QR/SVD solvers |
| `-Denable_nvrtc=true` | false | Enable `src/nvrtc/` runtime PTX compilation |
| `-Doptimize=ReleaseFast` | Debug | Standard Zig optimization modes |

Example:

```sh
zig build -Denable_nvrtc=true -Denable_cublas=true -Doptimize=ReleaseFast
```

## Docker (CUDA 12.8)

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y wget xz-utils
RUN wget -q https://ziglang.org/download/0.16.0/zig-linux-x86_64-0.16.0.tar.xz \
    && tar -xf zig-linux-x86_64-0.16.0.tar.xz -C /opt \
    && ln -s /opt/zig-linux-x86_64-0.16.0/zig /usr/local/bin/zig

WORKDIR /app
COPY . .
RUN zig build -Doptimize=ReleaseFast
```

## Docker (CUDA 13.3)

```dockerfile
FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y wget xz-utils
RUN wget -q https://ziglang.org/download/0.16.0/zig-linux-x86_64-0.16.0.tar.xz \
    && tar -xf zig-linux-x86_64-0.16.0.tar.xz -C /opt \
    && ln -s /opt/zig-linux-x86_64-0.16.0/zig /usr/local/bin/zig

WORKDIR /app
COPY . .
RUN zig build -Doptimize=ReleaseFast
```

## Verifying the Installation

```sh
zig build test
```

All tests pass on host without a GPU by running the CPU fallback path.

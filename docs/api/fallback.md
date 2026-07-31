---
title: Fallback API
description: CPU fallback backend and dispatch layer API in cuda.zig.
---

# Fallback API

## `cuda.fallback`

```zig
/// Force the CPU fallback backend regardless of GPU presence.
/// Must be called before cuda.init().
pub fn force(enabled: bool) void

/// Return true if the CPU fallback is currently active.
pub fn isActive() bool
```

## `fallback/dispatch.zig`

The dispatch layer sits between the public API and the CUDA runtime. On each call it checks whether the GPU backend is active:

```zig
pub fn memcpy(dst: anytype, src: anytype, bytes: usize, kind: MemcpyKind) !void {
    if (fallback.isActive()) {
        return cpu_backend.memcpy(dst, src, bytes);
    }
    return runtime.cudaMemcpy(dst, src, bytes, @intFromEnum(kind));
}
```

## CPU Backend Implementations

| GPU operation | CPU fallback |
|---|---|
| `cudaMalloc` | `allocator.alloc` |
| `cudaFree` | `allocator.free` |
| `cudaMemcpy` | `@memcpy` |
| `cudaMemset` | `@memset` |
| `cudaStreamCreate` | No-op handle |
| `cudaStreamSynchronize` | No-op |
| `cudaEventCreate` | Stores `std.time.Instant` |
| `cudaEventElapsedTime` | Monotonic clock delta |
| Tensor elementwise | Host scalar loop |
| Tensor reductions | Host scalar loop |
| Tensor matmul | Naïve O(n³) host loop |

## Environment Variable

```sh
CUDA_ZIG_FORCE_FALLBACK=1 ./your-binary
```

Setting this environment variable forces fallback mode without changing source code.

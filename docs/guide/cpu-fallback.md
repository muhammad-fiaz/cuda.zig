---
title: CPU Fallback Backend
description: How the transparent CPU fallback works in cuda.zig — no GPU required to run your code.
---

# CPU Fallback Backend

cuda.zig includes a transparent CPU fallback backend that mirrors the full GPU API surface. When no CUDA-capable device is detected at runtime, all operations automatically route through host-side implementations.

## How It Works

The entry point `cuda.init()` attempts to load `libcuda.so` (Linux), `nvcuda.dll` (Windows), or `libcuda.dylib` (macOS). If loading fails or no device is reported, the fallback dispatcher is activated:

```
cuda.init()
  └── loader.acquire()
        ├── (GPU present) → CUDA Runtime backend
        └── (no GPU)      → CPU Fallback backend
```

All subsequent calls go through `fallback/dispatch.zig`, which routes each operation to the matching host implementation in `fallback/cpu_backend.zig`.

## What Is Implemented

| API surface | CPU fallback |
|---|---|
| `DeviceBuffer(T)` | Backed by `std.mem.Allocator` heap allocation |
| `HostBuffer(T)` | Backed by `std.mem.Allocator` heap allocation |
| `ManagedBuffer(T)` | Backed by `std.mem.Allocator` heap allocation |
| `Stream.sync()` | No-op (host is always synchronised) |
| `Event.record()` | Captures `std.time.Instant` |
| `Event.elapsedMs()` | Uses monotonic clock delta |
| `Tensor(T)` elementwise | Single-threaded host loops |
| `Tensor(T)` matmul | Naïve O(n³) host matmul |
| `Tensor(T)` reductions | Single-threaded host reductions |

## Behaviour Differences

- Kernels cannot be launched in fallback mode. `func.launch()` returns `error.NoCudaDevice`.
- NVRTC compilation also returns `error.NoCudaDevice`.
- Peer access queries always return `false`.

## Testing Without a GPU

The test suite is designed to run entirely under the CPU fallback. Running:

```sh
zig build test
```

exercises every non-kernel API path on host. This is also what the CI pipeline runs.

## Forcing Fallback Mode

You can force fallback mode even on a machine with a GPU for testing:

```zig
cuda.fallback.force(true);
try cuda.init();
```

Or set the environment variable:

```sh
CUDA_ZIG_FORCE_FALLBACK=1 zig build run
```

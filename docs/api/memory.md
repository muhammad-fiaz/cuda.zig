---
title: Memory API Documentation
description: Reference for DeviceBuffer, PinnedBuffer, UnifiedBuffer, PoolBuffer, and memory management in cuda.zig.
---

# Memory API Reference

The `cuda.memory` namespace provides typed and raw memory management across GPU global memory, host pinned memory, unified/managed memory, and stream-ordered memory pools.

## Types & Modules

### `DeviceBuffer(T)`
Typed wrapper for CUDA device-side allocations (`cudaMalloc`/`cudaFree`).

```zig
var buf = try cuda.DeviceBuffer(f32).alloc(1024);
defer buf.deinit();

try buf.copyFromHost(&host_slice);
try buf.copyToHost(&out_slice);
```

### `PinnedBuffer(T)`
Host memory allocated in page-locked (pinned) RAM for maximum transfer bandwidth (`cudaHostAlloc`/`cudaFreeHost`).

```zig
var pinned = try cuda.PinnedBuffer(f32).alloc(1024);
defer pinned.deinit();
```

### `UnifiedBuffer(T)`
Managed memory accessible from both CPU and GPU transparently (`cudaMallocManaged`).

```zig
var unif = try cuda.UnifiedBuffer(f32).alloc(1024);
defer unif.deinit();

try unif.prefetchToDevice(0, stream);
try unif.prefetchToHost(stream);
```

### `PoolBuffer(T)`
Stream-ordered allocation backed by CUDA memory pools (`cudaMallocAsync`/`cudaFreeAsync`).

```zig
var pool_buf = try cuda.PoolBuffer(f32).alloc(1024, stream);
defer pool_buf.freeOnStream(stream) catch {};
```

## Low-Level Memory Functions (`cuda.runtime.memory`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `allocDevice` | `(size: usize) !*anyopaque` | Allocate raw device memory |
| `freeDevice` | `(ptr: *anyopaque) void` | Release device memory |
| `allocHost` | `(size: usize, flags: c_uint) !*anyopaque` | Allocate pinned host memory |
| `freeHost` | `(ptr: *anyopaque) void` | Release pinned host memory |
| `allocManaged` | `(size: usize, flags: c_uint) !*anyopaque` | Allocate unified memory |
| `allocAsync` | `(size: usize, stream: Stream) !*anyopaque` | Stream-ordered pool allocation |
| `freeAsync` | `(ptr: *anyopaque, stream: Stream) !void` | Stream-ordered pool deallocation |
| `mallocPitch` | `(width: usize, height: usize) !{ ptr, pitch }` | 2-D pitched allocation |
| `memcpy2D` | `(...) !void` | Synchronous 2-D copy |
| `memcpy2DAsync` | `(...) !void` | Asynchronous 2-D copy on stream |
| `ipcGetMemHandle` | `(ptr: *anyopaque) !IpcMemHandle` | Export allocation as IPC handle |
| `ipcOpenMemHandle` | `(handle: IpcMemHandle, flags) !*anyopaque` | Import IPC handle |
| `ipcCloseMemHandle` | `(ptr: *anyopaque) !void` | Close imported IPC handle |

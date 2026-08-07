---
title: Memory Buffers
description: Device memory, host-pinned memory, managed/unified memory, stream-ordered memory pools, and the CudaAllocator in cuda.zig.
---

# Memory Buffers

cuda.zig exposes four distinct memory spaces through a uniform, type-safe API:

| Type | Location | Accessible by | API |
|---|---|---|---|
| `DeviceBuffer(T)` | GPU VRAM | GPU only | `cuda.DeviceBuffer` |
| `PinnedBuffer(T)` | Pinned RAM | CPU + DMA | `cuda.PinnedBuffer` |
| `UnifiedBuffer(T)` | Unified / UM | CPU + GPU | `cuda.UnifiedBuffer` |
| `PoolBuffer(T)` | GPU VRAM (pool) | GPU only | `cuda.PoolBuffer` |

## Device Memory

```zig
var buf = try cuda.DeviceBuffer(f32).init(allocator, 1024);
defer buf.deinit();

// Zero-fill
try buf.memset(0);

// Copy host → device
const host: [1024]f32 = undefined;
try buf.copyFromHost(&host);

// Copy device → host
var out: [1024]f32 = undefined;
try buf.copyToHost(&out);

// Async copy (requires a stream)
try buf.copyFromHostAsync(&host, stream);
try buf.copyToHostAsync(&out, stream);
```

`DeviceBuffer` internally calls `cudaMalloc` / `cudaFree`. The allocator parameter is used for Zig-side bookkeeping only — no CPU memory is allocated for element data.

## Host-Pinned Memory

Pinned (page-locked) host memory enables DMA transfers and is required for the fastest H2D / D2H throughput:

```zig
var pinned = try cuda.PinnedBuffer(f32).alloc(1024);
defer pinned.deinit();

// Pointer is valid on host and can be directly read/written
pinned.slice()[0] = 42.0;

// Fast DMA copy to device
try device_buf.copyFromHostAsync(pinned.slice(), stream);
```

Backed by `cudaHostAlloc` / `cudaFreeHost` with `cudaHostAllocDefault`.

## Managed / Unified Memory

Unified Memory pages migrate between CPU and GPU automatically:

```zig
var unif = try cuda.UnifiedBuffer(f32).alloc(1024);
defer unif.deinit();

// Write from CPU
unif.slice()[0] = 1.0;

// Optional: prefetch to GPU before a kernel
try unif.prefetchToDevice(0, stream);

// Optional: prefetch back to CPU
try unif.prefetchToHost(stream);
```

Backed by `cudaMallocManaged` with `cudaMemAttachGlobal`.

## Memory Pools {#memory-pools}

Stream-ordered memory pools (`cudaMallocAsync` / `cudaFreeAsync`) allow the CUDA runtime to reuse allocations within the same stream, dramatically reducing allocation overhead in iterative workloads:

```zig
var stream = try cuda.Stream.init();
defer stream.deinit();

// Allocate from the stream-ordered pool
var pool_buf = try cuda.PoolBuffer(f32).alloc(1024, stream.handle);
defer pool_buf.freeOnStream(stream.handle) catch {};

std.debug.print("Pool size: {d} bytes\n", .{pool_buf.byteSize()});
```

> [!NOTE]
> `PoolBuffer` requires CUDA 11.2+ runtime symbols (`cudaMallocAsync`). On older runtimes cuda.zig returns an error gracefully — check availability with `cuda.isAvailable()`.

## 2D Pitched Memory

2D pitched memory aligns each row to the hardware's preferred pitch for coalesced access in 2D kernels (image processing, matrix operations):

```zig
// Allocate a 512×512 float matrix with hardware-optimal pitch
const pitch_res = try cuda.runtime.memory.mallocPitch(512 * @sizeOf(f32), 512);
defer cuda.runtime.memory.freeDevice(pitch_res.ptr);

std.debug.print("Pitch: {d} bytes\n", .{pitch_res.pitch});

// 2D async copy using the returned pitch
try cuda.runtime.memory.memcpy2DAsync(dst, pitch_res.pitch, src, src_pitch, width, height, stream);
```

## CudaAllocator

`CudaAllocator` implements `std.mem.Allocator`, letting you pass device memory to any allocator-aware Zig API:

```zig
var cuda_alloc = cuda.CudaAllocator.init();
const allocator = cuda_alloc.allocator();

const slice = try allocator.alloc(f32, 512);
defer allocator.free(slice);
```

> [!WARNING]
> Slices returned by `CudaAllocator` reside in device memory and cannot be dereferenced on the host. Pass them only to GPU-side code.

## Memset

```zig
try buf.memset(0);   // sets all bytes to 0
```

`memset` calls `cudaMemset` which zeroes a byte pattern across the buffer. For typed patterns (e.g. fill with `1.0f`), launch a custom kernel.

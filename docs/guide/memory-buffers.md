---
title: Memory Buffers
description: Device memory, host-pinned memory, managed/unified memory, and the CudaAllocator in cuda.zig.
---

# Memory Buffers

cuda.zig exposes three distinct memory spaces through a uniform, type-safe API:

| Type | Location | Accessible by | API |
|---|---|---|---|
| `DeviceBuffer(T)` | GPU VRAM | GPU only | `cuda.DeviceBuffer` |
| `HostBuffer(T)` | Pinned RAM | CPU + DMA | `cuda.HostBuffer` |
| `ManagedBuffer(T)` | Unified / UM | CPU + GPU | `cuda.ManagedBuffer` |

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
var pinned = try cuda.HostBuffer(f32).init(allocator, 1024);
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
var managed = try cuda.ManagedBuffer(f32).init(allocator, 1024);
defer managed.deinit();

// Write from CPU
managed.slice()[0] = 1.0;

// Optional: prefetch to GPU before a kernel
try managed.prefetchToDevice(0, stream);

// Optional: advise access patterns
try managed.advise(.PreferredLocation, 0);
try managed.advise(.AccessedBy, 0);
```

Backed by `cudaMallocManaged` with `cudaMemAttachGlobal`.

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

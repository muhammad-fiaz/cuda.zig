---
title: CUDA Allocator
description: Use CudaAllocator with any std.mem.Allocator-aware Zig code to allocate device memory.
---

# CUDA Allocator

`cuda.CudaAllocator` implements the `std.mem.Allocator` interface, enabling device memory to be used anywhere Zig expects a standard allocator.

## Initialisation

```zig
var cuda_alloc = cuda.CudaAllocator.init();
const allocator = cuda_alloc.allocator();
```

## Usage

```zig
const slice = try allocator.alloc(f32, 1024);
defer allocator.free(slice);

// Pass to any allocator-aware code
var list = try std.ArrayListUnmanaged(f32).initCapacity(allocator, 512);
defer list.deinit(allocator);
```

> [!WARNING]
> Slices allocated by `CudaAllocator` live in **device (GPU) memory**. Attempting to read or write them from the host causes undefined behaviour. Always use `cudaMemcpy` or the `DeviceBuffer` copy helpers to move data between host and device.

## Resize and Free

The allocator supports `resize` (no-op that always fails, forcing reallocation) and `free` via `cudaFree`.

## CPU Fallback Mode

In fallback mode `CudaAllocator` falls back to the system heap (`std.heap.c_allocator`), so code written against the allocator interface continues to work transparently.

## When to Use `CudaAllocator` vs `DeviceBuffer`

| Use `DeviceBuffer(T)` when… | Use `CudaAllocator` when… |
|---|---|
| You know the element type statically | You need a generic `Allocator` interface |
| You want copy helpers (H2D, D2H) | You're integrating with allocator-aware abstractions |
| You manage one homogeneous array | You need multiple typed allocations |

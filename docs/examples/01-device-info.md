---
title: Device Info Example
description: Query and display GPU device properties, compute capability, memory, and feature support using cuda.zig.
---

# Device Info Example

This example demonstrates how to enumerate CUDA devices and query their properties — name, compute capability, memory, multiprocessors, clock rate, and supported features.

## Full Source

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Device Info Example ===\n\n", .{});

    if (!cuda.isAvailable()) {
        std.debug.print("CUDA is NOT available. Operating in CPU Fallback mode.\n", .{});
        return;
    }

    // Display version info
    const caps = cuda.runtimeCapabilities();
    if (caps.driver_version.raw > 0) {
        std.debug.print("CUDA Driver Version:  {d}.{d}\n", .{ caps.driver_version.major(), caps.driver_version.minor() });
    }
    if (caps.runtime_version.raw > 0) {
        std.debug.print("CUDA Runtime Version: {d}.{d}\n", .{ caps.runtime_version.major(), caps.runtime_version.minor() });
    }

    const count = try cuda.deviceCount();
    std.debug.print("Found {d} CUDA device(s).\n\n", .{count});

    const allocator = std.heap.page_allocator;
    const devices = try cuda.allDevices(allocator);
    defer allocator.free(devices);

    for (devices, 0..) |dev, i| {
        std.debug.print("Device #{d}:\n", .{i});

        const dev_name = try dev.name();
        std.debug.print("  Name:               {s}\n", .{dev_name});

        const cap = try dev.computeCapability();
        std.debug.print("  Compute Capability: {d}.{d}\n", .{ cap.major, cap.minor });

        const total_mem = try dev.totalMemory();
        std.debug.print("  Total Memory:       {d} MB ({d} bytes)\n", .{ total_mem / (1024 * 1024), total_mem });

        if (dev.freeMemory()) |free_mem| {
            std.debug.print("  Free Memory:        {d} MB\n", .{free_mem / (1024 * 1024)});
        } else |_| {}

        const props = try dev.propertiesRaw();
        std.debug.print("  Multiprocessors:    {d}\n", .{props.multi_processor_count});
        std.debug.print("  Max Threads/Block:  {d}\n", .{props.max_threads_per_block});
        std.debug.print("  Warp Size:          {d}\n", .{props.warp_size});
        std.debug.print("  Clock Rate:         {d} MHz\n", .{props.clock_rate_khz / 1000});
        std.debug.print("  L2 Cache:           {d} KB\n", .{props.l2_cache_size / 1024});
        std.debug.print("  Unified Addressing: {}\n", .{props.unified_addressing});
        std.debug.print("  Managed Memory:     {}\n", .{props.managed_memory});
        std.debug.print("\n", .{});
    }

    std.debug.print("=== CUDA Feature Support ===\n", .{});
    std.debug.print("  CUDA 12+:             {}\n", .{caps.supports_cuda_12});
    std.debug.print("  CUDA Graphs:          {}\n", .{caps.supports_graphs});
    std.debug.print("  Cooperative Launch:   {}\n", .{caps.supports_cooperative_launch});
    std.debug.print("  Memory Pools:         {}\n", .{caps.supports_memory_pools});
    std.debug.print("  Tensor Cores:         {}\n", .{caps.supports_tensor_cores});
    std.debug.print("  Managed Prefetch:     {}\n", .{caps.supports_managed_prefetch});
}
```

## Example Output (NVIDIA GeForce RTX 4070 SUPER)

```
=== cuda.zig Device Info Example ===

CUDA Driver Version:  13.3
Found 1 CUDA device(s).

Device #0:
  Name:               NVIDIA GeForce RTX 4070 SUPER
  Compute Capability: 8.9
  Total Memory:       12281 MB (12878086144 bytes)
  Free Memory:        11069 MB
  Multiprocessors:    56
  Max Threads/Block:  1024
  Warp Size:          32
  Clock Rate:         2475 MHz
  L2 Cache:           10254 KB
  Unified Addressing: false
  Managed Memory:     false

=== CUDA Feature Support ===
  CUDA 12+:             true
  CUDA Graphs:          true
  Cooperative Launch:   true
  Memory Pools:         true
  Tensor Cores:         true
  Managed Prefetch:     true
```

## Key APIs Used

| API | Description |
|-----|-------------|
| `cuda.isAvailable()` | Check if CUDA driver is loaded |
| `cuda.runtimeCapabilities()` | Get driver/runtime version and feature flags |
| `cuda.deviceCount()` | Number of visible CUDA devices |
| `cuda.allDevices(allocator)` | Enumerate all `Device` handles |
| `dev.name()` | Device name string (e.g. "NVIDIA GeForce RTX 4070 SUPER") |
| `dev.computeCapability()` | Compute capability `{ major, minor }` |
| `dev.totalMemory()` | Total GPU VRAM in bytes |
| `dev.freeMemory()` | Currently free GPU VRAM |
| `dev.propertiesRaw()` | Full `DeviceProperties` struct |

## How it Works

cuda.zig uses dynamic symbol resolution — it loads `nvcuda.dll` (Driver API) at runtime and calls `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem_v2`, and `cuDeviceGetAttribute` to populate device information without requiring pre-linked CUDA SDK headers or DLLs.

## Running This Example

```sh
zig build example-device-info
```

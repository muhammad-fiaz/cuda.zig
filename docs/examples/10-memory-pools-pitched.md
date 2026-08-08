---
title: Memory Pools & 2-D Pitched Memory Example
description: Stream-ordered pool allocations and 2D pitched memory management in cuda.zig.
---

# Memory Pools & 2-D Pitched Memory Example

Demonstrates stream-ordered memory pool allocations (`cudaMallocAsync`) and 2-D pitched memory management (`cudaMallocPitch`) in `cuda.zig`.

## Full Source

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Memory Pools & 2D Pitched Memory Example ===\n\n", .{});

    if (!cuda.isAvailable()) {
        std.debug.print("CUDA is NOT available. Operating in CPU Fallback mode.\n", .{});
        return;
    }

    var stream = try cuda.Stream.init();
    defer stream.deinit();

    // 1. Stream-Ordered Memory Pool Allocation
    if (cuda.PoolBuffer(f32).alloc(1024, stream.handle)) |pool_buf| {
        defer pool_buf.freeOnStream(stream.handle) catch {};
        std.debug.print("Pool allocation successful. Size: {d} bytes.\n\n", .{pool_buf.byteSize()});
    } else |_| {
        std.debug.print("cudaMallocAsync requires CUDA 11.2+ runtime symbols; skipped gracefully.\n\n", .{});
    }

    // 2. 2D Pitched Memory Allocation
    std.debug.print("Allocating 512x512 2D Pitched Memory Matrix...\n", .{});
    if (cuda.runtime.memory.mallocPitch(512 * @sizeOf(f32), 512)) |pitch_res| {
        defer cuda.runtime.memory.freeDevice(pitch_res.ptr);
        std.debug.print("2D Pitched allocation successful. Pitch: {d} bytes (row width: {d} bytes).\n", .{
            pitch_res.pitch,
            512 * @sizeOf(f32),
        });
    } else |_| {
        std.debug.print("cudaMallocPitch skipped (Driver API fallback active).\n", .{});
    }
}
```

## Running This Example

```sh
zig build example-memory-pools-pitched
```

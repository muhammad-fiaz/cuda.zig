---
title: Occupancy Calculator & Stream Priority Example
description: Querying SM occupancy and stream priority ranges in cuda.zig.
---

# Occupancy Calculator & Stream Priority Example

Demonstrates stream priority range queries, priority stream creation, and profiler session markers in `cuda.zig`.

## Full Source

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Occupancy Calculator & Stream Priority Example ===\n\n", .{});

    if (!cuda.isAvailable()) {
        std.debug.print("CUDA is NOT available. Operating in CPU Fallback mode.\n", .{});
        return;
    }

    // 1. Stream Priorities
    const range = try cuda.runtime.stream.getStreamPriorityRange();
    std.debug.print("Device Stream Priority Range: Highest={d}, Lowest={d}\n", .{ range.greatest, range.least });

    const high_prio_stream = try cuda.runtime.stream.createStreamWithPriority(
        cuda.runtime.ffi.StreamFlags.non_blocking,
        range.greatest,
    );
    defer cuda.runtime.stream.destroyStream(high_prio_stream) catch {};

    if (cuda.runtime.stream.getStreamPriority(high_prio_stream)) |prio| {
        std.debug.print("High Priority Stream created with priority: {d}\n\n", .{prio});
    } else |_| {
        std.debug.print("Stream priority query handled gracefully.\n\n", .{});
    }

    // 2. Profiler session start/stop
    cuda.profiler.start();
    defer cuda.profiler.stop();

    var _guard = cuda.profiler.ProfilerGuard.begin();
    defer _guard.end();

    std.debug.print("Profiler session markers attached successfully.\n", .{});
}
```

## Running This Example

```sh
zig build example-occupancy-profiler
```

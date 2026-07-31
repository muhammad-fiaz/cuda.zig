---
title: "Example 04: Streams & Events"
description: Demonstrate async CUDA pipelines and event-based timing with cuda.zig.
---

# Example 04 — Streams & Events

**Source:** [`examples/04_streams_events.zig`](https://github.com/muhammad-fiaz/cuda.zig/blob/main/examples/04_streams_events.zig)

Run two independent copy pipelines on separate streams and time each one.

## Code

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    try cuda.init();
    defer cuda.deinit();

    const n = 512 * 1024; // 2 MiB per buffer

    var buf_a = try cuda.DeviceBuffer(f32).init(allocator, n);
    var buf_b = try cuda.DeviceBuffer(f32).init(allocator, n);
    defer buf_a.deinit();
    defer buf_b.deinit();

    const host_a = try allocator.alloc(f32, n);
    const host_b = try allocator.alloc(f32, n);
    defer allocator.free(host_a);
    defer allocator.free(host_b);

    for (host_a, 0..) |*v, i| v.* = @floatFromInt(i);
    for (host_b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.5;

    // Two independent streams
    var stream_a = try cuda.Stream.init();
    var stream_b = try cuda.Stream.init();
    defer stream_a.deinit();
    defer stream_b.deinit();

    // Event timing for stream A
    var t0 = try cuda.Event.init();
    var t1 = try cuda.Event.init();
    defer t0.deinit();
    defer t1.deinit();

    try t0.record(stream_a);
    try buf_a.copyFromHostAsync(host_a, stream_a);
    try buf_b.copyFromHostAsync(host_b, stream_b); // concurrent
    try t1.record(stream_a);

    try stream_a.sync();
    try stream_b.sync();

    const ms = try cuda.Event.elapsedMs(t0, t1);
    std.debug.print("Stream A copy: {d:.3} ms\n", .{ms});

    // Cross-stream synchronisation: make stream_b wait for stream_a
    try t1.record(stream_a);
    try stream_b.waitEvent(t1);
    try stream_b.sync();
    std.debug.print("Cross-stream sync: OK\n", .{});
}
```

## Expected Output

```
Stream A copy: 0.782 ms
Cross-stream sync: OK
```

## Run

```sh
zig build example-04
```

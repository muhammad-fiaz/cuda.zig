---
title: "Example 02: Memory Transfer"
description: Host-to-device and device-to-host memory transfers with timing using cuda.zig.
---

# Example 02 — Memory Transfer

**Source:** [`examples/02_memory_transfer.zig`](https://github.com/muhammad-fiaz/cuda.zig/blob/main/examples/02_memory_transfer.zig)

Allocate device memory, copy a host array to the GPU, and copy it back — measuring the round-trip time.

## Code

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    try cuda.init();
    defer cuda.deinit();

    const n = 1024 * 1024; // 4 MiB of f32
    const host_in = try allocator.alloc(f32, n);
    const host_out = try allocator.alloc(f32, n);
    defer allocator.free(host_in);
    defer allocator.free(host_out);

    for (host_in, 0..) |*v, i| v.* = @floatFromInt(i);

    var buf = try cuda.DeviceBuffer(f32).init(allocator, n);
    defer buf.deinit();

    var stream = try cuda.Stream.init();
    defer stream.deinit();

    var t0 = try cuda.Event.init();
    var t1 = try cuda.Event.init();
    defer t0.deinit();
    defer t1.deinit();

    // Timed async round-trip
    try t0.record(stream);
    try buf.copyFromHostAsync(host_in, stream);
    try buf.copyToHostAsync(host_out, stream);
    try t1.record(stream);
    try t1.sync();

    const ms = try cuda.Event.elapsedMs(t0, t1);
    const gb = @as(f64, @floatFromInt(n * @sizeOf(f32) * 2)) / 1e9;
    std.debug.print("Round-trip {d} MiB in {d:.2} ms ({d:.1} GB/s)\n", .{
        n * @sizeOf(f32) / (1024 * 1024), ms, gb / (ms / 1000.0),
    });

    // Verify data integrity
    var ok = true;
    for (host_in, host_out) |a, b| {
        if (a != b) { ok = false; break; }
    }
    std.debug.print("Data integrity: {s}\n", .{if (ok) "PASS" else "FAIL"});
}
```

## Expected Output

```
Round-trip 4 MiB in 1.43 ms (5.6 GB/s)
Data integrity: PASS
```

## Run

```sh
zig build example-02
```

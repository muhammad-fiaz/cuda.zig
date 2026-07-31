---
title: "Example 06: Multi-GPU"
description: Enable peer access and perform cross-device memory copies with cuda.zig.
---

# Example 06 — Multi-GPU

**Source:** [`examples/06_multi_gpu.zig`](https://github.com/muhammad-fiaz/cuda.zig/blob/main/examples/06_multi_gpu.zig)

Detect multiple GPUs, enable peer-to-peer access, and copy memory between devices.

## Code

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    try cuda.init();
    defer cuda.deinit();

    const n = try cuda.device.count();
    std.debug.print("Devices: {d}\n", .{n});

    if (n < 2) {
        std.debug.print("Need at least 2 GPUs for this example.\n", .{});
        return;
    }

    // Check and enable peer access 0 → 1
    const p2p = try cuda.peer.canAccess(0, 1);
    std.debug.print("P2P 0→1 capable: {}\n", .{p2p});

    if (p2p) {
        try cuda.device.set(0);
        try cuda.peer.enable(1);
        std.debug.print("Peer access enabled.\n", .{});
    }

    // Allocate on each device
    try cuda.device.set(0);
    var buf0 = try cuda.DeviceBuffer(f32).init(allocator, 1024);
    defer buf0.deinit();

    try cuda.device.set(1);
    var buf1 = try cuda.DeviceBuffer(f32).init(allocator, 1024);
    defer buf1.deinit();

    // Write known data on device 1
    var host: [1024]f32 = undefined;
    for (&host, 0..) |*v, i| v.* = @floatFromInt(i);
    try buf1.copyFromHost(&host);

    if (p2p) {
        // Cross-device copy: device 1 → device 0
        try cuda.memory.copyPeer(
            buf0.devicePtr(), 0,
            buf1.devicePtr(), 1,
            1024 * @sizeOf(f32),
        );
        std.debug.print("P2P copy complete.\n", .{});
    }

    try cuda.device.set(0);
    var result: [1024]f32 = undefined;
    try buf0.copyToHost(&result);
    std.debug.print("result[42] = {d}\n", .{result[42]}); // should be 42.0
}
```

## Expected Output (2 GPUs with P2P)

```
Devices: 2
P2P 0→1 capable: true
Peer access enabled.
P2P copy complete.
result[42] = 42.0
```

## Run

```sh
zig build example-06
```

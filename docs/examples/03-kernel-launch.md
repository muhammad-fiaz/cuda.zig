---
title: "Example 03: Kernel Launch"
description: Load a PTX kernel, configure grid and block dimensions, and launch it with cuda.zig.
---

# Example 03 — Kernel Launch

**Source:** [`examples/03_kernel_launch.zig`](https://github.com/muhammad-fiaz/cuda.zig/blob/main/examples/03_kernel_launch.zig)

Embed a PTX string (compiled from a simple CUDA C kernel), load it as a module, and launch it.

## CUDA C Kernel

```c
// kernel.cu  — compile with: nvcc -ptx kernel.cu
extern "C" __global__ void scale(float* data, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}
```

## Zig Launcher

```zig
const std = @import("std");
const cuda = @import("cuda");

// Embed the PTX at compile time
const ptx = @embedFile("scale.ptx");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    try cuda.init();
    defer cuda.deinit();

    _ = try cuda.Device.select(0);

    // Host data
    const n: i32 = 65536;
    const host_data = blk: {
        var arr: [65536]f32 = undefined;
        for (&arr, 0..) |*v, i| v.* = @floatFromInt(i);
        break :blk arr;
    };

    var buf = try cuda.DeviceBuffer(f32).init(allocator, @intCast(n));
    defer buf.deinit();
    try buf.copyFromHost(&host_data);

    // Load module and look up function
    var module = try cuda.Module.load(ptx);
    defer module.deinit();
    var kernel = try module.getFunction("scale");

    // Launch: 256 threads per block
    const threads: u32 = 256;
    const blocks: u32 = (@as(u32, @intCast(n)) + threads - 1) / threads;
    var stream = try cuda.Stream.init();
    defer stream.deinit();

    const factor: f32 = 2.0;
    try kernel.launch(
        .{ .x = blocks, .y = 1, .z = 1 },
        .{ .x = threads, .y = 1, .z = 1 },
        0, stream,
        .{ buf.devicePtr(), &factor, &n },
    );
    try stream.sync();

    // Verify
    var result: [65536]f32 = undefined;
    try buf.copyToHost(&result);
    std.debug.print("result[0]     = {d}\n", .{result[0]});   // 0.0 * 2 = 0
    std.debug.print("result[1]     = {d}\n", .{result[1]});   // 1.0 * 2 = 2
    std.debug.print("result[65535] = {d}\n", .{result[65535]}); // 65535 * 2
}
```

## Expected Output

```
result[0]     = 0
result[1]     = 2
result[65535] = 131070
```

## Run

```sh
nvcc -ptx examples/scale.cu -o examples/scale.ptx
zig build example-03
```

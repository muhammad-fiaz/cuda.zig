---
title: "Example 09: NVRTC Compilation"
description: Compile a CUDA C kernel string to PTX at runtime and launch it using cuda.zig NVRTC.
---

# Example 09 — NVRTC Compilation

**Source:** [`examples/09_nvrtc_compilation.zig`](https://github.com/muhammad-fiaz/cuda.zig/blob/main/examples/09_nvrtc_compilation.zig)

Compile a CUDA C/C++ kernel string to PTX at runtime — no `nvcc` or pre-compiled PTX file required.

## Code

```zig
const std = @import("std");
const cuda = @import("cuda");

const kernel_src =
    \\extern "C" __global__ void saxpy(float a, float* x, float* y, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) y[i] = a * x[i] + y[i];
    \\}
;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    try cuda.init();
    defer cuda.deinit();

    _ = try cuda.Device.select(0);

    // --- NVRTC Compilation ---
    var prog = try cuda.nvrtc.Program.create(kernel_src, "saxpy.cu");
    defer prog.deinit();

    prog.compile(&.{}) catch |err| {
        const log = try prog.getLog(allocator);
        defer allocator.free(log);
        std.debug.print("Compilation failed:\n{s}\n", .{log});
        return err;
    };
    std.debug.print("Compilation: OK\n", .{});

    const ptx = try prog.getPtx(allocator);
    defer allocator.free(ptx);

    // --- Module and kernel ---
    var module = try cuda.Module.load(ptx);
    defer module.deinit();
    var kernel = try module.getFunction("saxpy");

    // --- Buffers ---
    const n: i32 = 1024;
    const a: f32 = 3.0;

    var host_x: [1024]f32 = undefined;
    var host_y: [1024]f32 = undefined;
    for (&host_x, 0..) |*v, i| v.* = @floatFromInt(i);     // x[i] = i
    for (&host_y, 0..) |*v, _| v.* = 1.0;                  // y[i] = 1

    var d_x = try cuda.DeviceBuffer(f32).init(allocator, @intCast(n));
    var d_y = try cuda.DeviceBuffer(f32).init(allocator, @intCast(n));
    defer d_x.deinit();
    defer d_y.deinit();

    try d_x.copyFromHost(&host_x);
    try d_y.copyFromHost(&host_y);

    // --- Launch ---
    const threads: u32 = 256;
    const blocks: u32 = (@as(u32, @intCast(n)) + threads - 1) / threads;

    var stream = try cuda.Stream.init();
    defer stream.deinit();

    try kernel.launch(
        .{ .x = blocks }, .{ .x = threads }, 0, stream,
        .{ &a, d_x.devicePtr(), d_y.devicePtr(), &n },
    );
    try stream.sync();

    // --- Verify: y[i] = 3*i + 1 ---
    var result: [1024]f32 = undefined;
    try d_y.copyToHost(&result);
    std.debug.print("y[0]   = {d}  (expected 1.0)\n",    .{result[0]});
    std.debug.print("y[1]   = {d}  (expected 4.0)\n",    .{result[1]});
    std.debug.print("y[100] = {d}  (expected 301.0)\n",  .{result[100]});
}
```

## Expected Output

```
Compilation: OK
y[0]   = 1.0  (expected 1.0)
y[1]   = 4.0  (expected 4.0)
y[100] = 301.0  (expected 301.0)
```

## Run

```sh
zig build example-09
```

> [!NOTE]
> NVRTC requires the CUDA Toolkit to be installed (`libnvrtc.so` / `nvrtc64_*.dll`). The example fails gracefully in CPU fallback mode with `error.NoCudaDevice`.

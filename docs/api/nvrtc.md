---
title: NVRTC API
description: Runtime CUDA C/C++ compilation API in cuda.zig.
---

# NVRTC API

## `cuda.nvrtc.Program`

```zig
pub const Program = struct {
    handle: nvrtcProgram,

    /// Create a program from a CUDA C/C++ source string.
    /// `name` is used only in error messages (e.g. "kernel.cu").
    pub fn create(src: []const u8, name: [:0]const u8) !Program

    /// Compile the program with optional flags.
    /// Flags are passed directly to NVRTC (e.g. "--gpu-architecture=compute_80").
    pub fn compile(self: *Program, flags: []const [:0]const u8) !void

    /// Return the compiled PTX. Caller owns the returned slice.
    pub fn getPtx(self: Program, allocator: std.mem.Allocator) ![]const u8

    /// Return the compilation log. Caller owns the returned slice.
    pub fn getLog(self: Program, allocator: std.mem.Allocator) ![]const u8

    /// Destroy the NVRTC program handle.
    pub fn deinit(self: *Program) void
};
```

## Common Compiler Flags

| Flag | Effect |
|---|---|
| `--gpu-architecture=compute_XX` | Target compute capability (e.g. `compute_80` for A100) |
| `--std=c++17` | Enable C++17 features |
| `--use_fast_math` | Enable fast math approximations |
| `--extra-device-vectorization` | Increase vectorisation |
| `-default-device` | Compile __device__ functions without explicit annotation |

## Error Handling

```zig
program.compile(&.{}) catch |err| {
    const log = try program.getLog(allocator);
    defer allocator.free(log);
    std.debug.print("Compile error:\n{s}\n", .{log});
    return err;
};
```

## Full Example

```zig
const cuda = @import("cuda");
const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    try cuda.init();
    defer cuda.deinit();

    const src =
        \\extern "C" __global__ void increment(float* data, int n) {
        \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (i < n) data[i] += 1.0f;
        \\}
    ;

    var prog = try cuda.nvrtc.Program.create(src, "increment.cu");
    defer prog.deinit();

    try prog.compile(&.{});
    const ptx = try prog.getPtx(allocator);
    defer allocator.free(ptx);

    var module = try cuda.Module.load(ptx);
    defer module.deinit();

    var kernel = try module.getFunction("increment");

    const n: i32 = 1024;
    var buf = try cuda.DeviceBuffer(f32).init(allocator, @intCast(n));
    defer buf.deinit();
    try buf.memset(0);

    const threads: u32 = 256;
    const blocks: u32 = (@as(u32, @intCast(n)) + threads - 1) / threads;
    var stream = try cuda.Stream.init();
    defer stream.deinit();

    try kernel.launch(
        .{ .x = blocks }, .{ .x = threads }, 0, stream,
        .{ buf.devicePtr(), &n },
    );
    try stream.sync();
}
```

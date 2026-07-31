---
title: NVRTC Runtime Compilation
description: Compile CUDA C/C++ kernels to PTX at runtime using NVRTC in cuda.zig.
---

# NVRTC Runtime Compilation

NVRTC (NVIDIA Runtime Compilation) lets you compile CUDA C/C++ source code to PTX at application runtime — no `nvcc` binary required at compile time.

## Creating a Program

```zig
const cuda = @import("cuda");

const kernel_src =
    \\extern "C" __global__ void saxpy(float a, float* x, float* y, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) y[i] = a * x[i] + y[i];
    \\}
;

var prog = try cuda.nvrtc.Program.create(kernel_src, "saxpy.cu");
defer prog.deinit();
```

## Compiling

```zig
// Minimal: compile with no extra flags
try prog.compile(&.{});

// With architecture targeting
try prog.compile(&.{
    "--gpu-architecture=compute_80",
    "--std=c++17",
    "--extra-device-vectorization",
});
```

If compilation fails, the error log is captured and returned:

```zig
prog.compile(&.{}) catch |err| {
    const log = try prog.getLog(allocator);
    defer allocator.free(log);
    std.debug.print("NVRTC error:\n{s}\n", .{log});
    return err;
};
```

## Extracting PTX

```zig
const ptx = try prog.getPtx(allocator);
defer allocator.free(ptx);
```

## Loading and Launching

```zig
var module = try cuda.Module.load(ptx);
defer module.deinit();

var kernel = try module.getFunction("saxpy");

const n: i32 = 65536;
const a: f32 = 2.0;
var d_x = try cuda.DeviceBuffer(f32).init(allocator, @intCast(n));
var d_y = try cuda.DeviceBuffer(f32).init(allocator, @intCast(n));
defer d_x.deinit();
defer d_y.deinit();

const threads: u32 = 256;
const blocks: u32 = (@as(u32, @intCast(n)) + threads - 1) / threads;

try kernel.launch(
    .{ .x = blocks, .y = 1, .z = 1 },
    .{ .x = threads, .y = 1, .z = 1 },
    0, stream,
    .{ &a, d_x.devicePtr(), d_y.devicePtr(), &n },
);
```

## PTX Caching

For production use, cache the PTX string to disk to avoid recompilation on every run:

```zig
const cache_path = "cache/saxpy.ptx";
const ptx = blk: {
    // Try loading from cache
    const f = std.fs.cwd().openFile(cache_path, .{}) catch |e| {
        if (e != error.FileNotFound) return e;
        // Compile fresh
        var p = try cuda.nvrtc.Program.create(kernel_src, "saxpy.cu");
        defer p.deinit();
        try p.compile(&.{});
        const fresh_ptx = try p.getPtx(allocator);
        try std.fs.cwd().makePath("cache");
        try std.fs.cwd().writeFile(cache_path, fresh_ptx);
        break :blk fresh_ptx;
    };
    defer f.close();
    break :blk try f.readToEndAlloc(allocator, 4 * 1024 * 1024);
};
defer allocator.free(ptx);
```

## NVRTC API Functions

| Function | Description |
|---|---|
| `Program.create(src, name)` | Create a program from a source string |
| `Program.compile(options)` | Compile to PTX |
| `Program.getPtx(allocator)` | Return the compiled PTX |
| `Program.getLog(allocator)` | Return the compilation log |
| `Program.deinit()` | Free NVRTC program handle |

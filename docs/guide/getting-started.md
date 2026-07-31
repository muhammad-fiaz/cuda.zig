---
title: Getting Started
description: Install cuda.zig, configure your project, and run your first GPU kernel in under five minutes.
---

# Getting Started

cuda.zig is a production-grade Zig library that wraps the CUDA Runtime and Driver APIs with zero link-time dependencies. All CUDA symbols are resolved at runtime through dynamic loading, so your binary compiles on any machine — even without a GPU or CUDA Toolkit installed.

## Prerequisites

| Requirement | Version |
|---|---|
| [Zig](https://ziglang.org/download/) | 0.16.0 or newer |
| CUDA Toolkit *(optional)* | 12.x or 13.x (13.3 Update 1 recommended) |
| GPU *(optional)* | Any NVIDIA GPU with compute capability 5.0+ |

Without a GPU or CUDA Toolkit, cuda.zig automatically falls back to a transparent CPU backend that implements the same API surface.

## Installation

### Stable (via `zig fetch`)

```sh
zig fetch --save https://github.com/muhammad-fiaz/cuda.zig/archive/refs/tags/v0.0.1.tar.gz
```

### Development / Nightly (directly from GitHub)

```sh
zig fetch --save https://github.com/muhammad-fiaz/cuda.zig/archive/refs/heads/main.tar.gz
```

### `build.zig.zon`

After running `zig fetch`, your `build.zig.zon` will contain an entry like:

```zig
.dependencies = .{
    .cuda = .{
        .url = "https://github.com/muhammad-fiaz/cuda.zig/archive/refs/tags/v0.0.1.tar.gz",
        .hash = "<sha256 printed by zig fetch>",
    },
},
```

### `build.zig`

Add the module to your executable:

```zig
const cuda_dep = b.dependency("cuda", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("cuda", cuda_dep.module("cuda"));
```

## Your First Program

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    // Initialize CUDA (falls back silently to CPU if no GPU is available)
    try cuda.init();
    defer cuda.deinit();

    const dev = try cuda.Device.select(0);
    std.debug.print("Running on: {s}\n", .{dev.name()});

    // Allocate device memory and copy a slice to the GPU
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const host_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var buf = try cuda.DeviceBuffer(f32).init(allocator, host_data.len);
    defer buf.deinit();

    try buf.copyFromHost(&host_data);

    var result: [8]f32 = undefined;
    try buf.copyToHost(&result);
    std.debug.print("Round-trip OK: {any}\n", .{result});
}
```

Run with:

```sh
zig build run
```

## Next Steps

- [Installation guide](/guide/installation) — build options, feature flags, Docker
- [Device Management](/guide/device-management) — enumerate GPUs, read properties
- [Memory Buffers](/guide/memory-buffers) — device, host-pinned, and managed memory
- [Kernel Launch](/guide/kernel-launch) — PTX, dynamic shared memory, argument marshaling

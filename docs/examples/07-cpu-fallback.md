---
title: "Example 07: CPU Fallback"
description: Run cuda.zig programs without a GPU using the transparent CPU fallback backend.
---

# Example 07 — CPU Fallback

**Source:** [`examples/07_cpu_fallback.zig`](https://github.com/muhammad-fiaz/cuda.zig/blob/main/examples/07_cpu_fallback.zig)

Demonstrate that the full Tensor API works identically in CPU fallback mode — no code changes required.

## Code

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Force fallback mode for demonstration
    cuda.fallback.force(true);

    try cuda.init();
    defer cuda.deinit();

    std.debug.print("Fallback active: {}\n", .{cuda.fallback.isActive()});

    // Exactly the same code as the GPU path
    var t = try cuda.Tensor(f32).init(allocator, .{8});
    defer t.deinit();

    const data = [_]f32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    try t.copyFromHost(&data);
    try t.add_scalar(5.0);

    var out: [8]f32 = undefined;
    try t.copyToHost(&out);
    std.debug.print("Result: {any}\n", .{out});

    const total = try t.sum();
    std.debug.print("Sum: {d}\n", .{total});
}
```

## Expected Output

```
Fallback active: true
Result: { 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0 }
Sum: 400.0
```

## Run

```sh
zig build example-07
# Or force via environment variable:
CUDA_ZIG_FORCE_FALLBACK=1 zig build run
```

## Use Case

The CPU fallback makes cuda.zig useful for:

- **CI/CD pipelines** running on CPU-only workers
- **macOS development** where CUDA is unavailable
- **Unit testing** without a GPU requirement
- **Algorithm validation** before GPU optimisation

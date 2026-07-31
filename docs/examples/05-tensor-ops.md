---
title: "Example 05: Tensor Operations"
description: Elementwise operations, reductions, and matrix multiplication with cuda.zig Tensor(T).
---

# Example 05 — Tensor Operations

**Source:** [`examples/05_tensor_ops.zig`](https://github.com/muhammad-fiaz/cuda.zig/blob/main/examples/05_tensor_ops.zig)

Use the high-level `Tensor(f32)` API for elementwise ops, reductions, and matrix multiplication.

## Code

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    try cuda.init();
    defer cuda.deinit();

    // --- 1D elementwise ---
    var t = try cuda.Tensor(f32).init(allocator, .{8});
    defer t.deinit();

    const data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try t.copyFromHost(&data);
    try t.scale(2.0);

    var out: [8]f32 = undefined;
    try t.copyToHost(&out);
    std.debug.print("After scale×2: {any}\n", .{out});

    // --- Reductions ---
    const s = try t.sum();
    const mx = try t.max();
    std.debug.print("Sum={d}  Max={d}\n", .{ s, mx });

    // --- 2D matmul: [4,4] × [4,4] → [4,4] ---
    var a = try cuda.Tensor(f32).init(allocator, .{ 4, 4 });
    var b = try cuda.Tensor(f32).init(allocator, .{ 4, 4 });
    var c = try cuda.Tensor(f32).init(allocator, .{ 4, 4 });
    defer a.deinit();
    defer b.deinit();
    defer c.deinit();

    // Identity matrix A
    const ident = [_]f32{
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1,
    };
    // Arbitrary B
    const bdata = [_]f32{
        1,2,3,4,
        5,6,7,8,
        9,10,11,12,
        13,14,15,16,
    };
    try a.copyFromHost(&ident);
    try b.copyFromHost(&bdata);

    try cuda.Tensor(f32).matmul(a, b, c);

    var cdata: [16]f32 = undefined;
    try c.copyToHost(&cdata);
    std.debug.print("I × B = {any}\n", .{cdata});
}
```

## Expected Output

```
After scale×2: { 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0 }
Sum=72  Max=16
I × B = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 }
```

## Run

```sh
zig build example-05
```

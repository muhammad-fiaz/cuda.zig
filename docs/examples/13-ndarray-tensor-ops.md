---
title: "Example 13: N-Dimensional Tensor Operations"
description: N-D Tensor manipulation, broadcasting, axis reductions, batched matmul, reshape, transpose, and slice in cuda.zig.
---

# Example 13: N-Dimensional Tensor Operations

This example demonstrates the complete set of N-Dimensional tensor operations in `cuda.zig` up to **8 dimensions**.

## Source Code

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    // 1. Create N-D Tensor
    var t = try cuda.Tensor(f32).zeros(&.{ 2, 3, 4 });
    defer t.deinit();

    // 2. Reshape & Flatten
    var reshaped = try t.reshape(&.{ 4, 6 });
    defer reshaped.deinit();

    // 3. Transpose
    var transposed = try reshaped.T2();
    defer transposed.deinit();

    // 4. Broadcast Addition [3,4] + [4]
    var a = try cuda.Tensor(f32).fromSlice(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &.{ 3, 4 });
    defer a.deinit();
    var bias = try cuda.Tensor(f32).fromSlice(&.{ 100, 200, 300, 400 }, &.{4});
    defer bias.deinit();

    var result = try a.broadcastAdd(bias);
    defer result.deinit();

    // 5. Batched MatMul [2,2,2] @ [2,2,2]
    var b1 = try cuda.Tensor(f32).zeros(&.{ 2, 2, 2 }); defer b1.deinit();
    var b2 = try cuda.Tensor(f32).zeros(&.{ 2, 2, 2 }); defer b2.deinit();
    var b_out = try b1.batchedMatmul(b2); defer b_out.deinit();
}
```

## Running the Example

```sh
zig build example-ndarray-tensor-ops
```

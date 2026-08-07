---
title: Tensor API
description: High-level N-Dimensional Tensor(T) API — elementwise, broadcast, reduction, batched matmul, reshape, transpose, and slice operations in cuda.zig.
---

# Tensor API

`cuda.Tensor(T)` is an N-Dimensional tensor supporting up to **8 dimensions** (`MAX_DIMS = 8`), row-major C-contiguous memory layout, and dynamic CPU/GPU dispatch.

## `cuda.Tensor(T)`

```zig
pub fn Tensor(comptime T: type) type {
    return struct {
        buffer: DeviceBuffer(T),
        shape: Shape,

        // ---- Lifecycle ----
        pub fn zeros(shape_slice: []const usize) !Self
        pub fn fromSlice(data: []const T, shape_slice: []const usize) !Self
        pub fn clone(self: Self) !Self
        pub fn deinit(self: *Self) void

        // ---- Introspection ----
        pub fn numel(self: Self) usize
        pub fn ndim(self: Self) usize
        pub fn size(self: Self, dim: usize) usize
        pub fn toHost(self: Self, allocator: std.mem.Allocator) ![]T
        pub fn print(self: Self) void

        // ---- Shape Transforms ----
        pub fn reshape(self: Self, new_shape: []const usize) !Self
        pub fn flatten(self: Self) !Self
        pub fn squeeze(self: Self, axis: ?usize) !Self
        pub fn unsqueeze(self: Self, axis: usize) !Self
        pub fn transpose(self: Self, perm: []const usize) !Self
        pub fn T2(self: Self) !Self

        // ---- Same-shape Elementwise ----
        pub fn add(self: Self, other: Self) !Self
        pub fn sub(self: Self, other: Self) !Self
        pub fn mul(self: Self, other: Self) !Self
        pub fn div(self: Self, other: Self) !Self
        pub fn relu(self: Self) !Self
        pub fn neg(self: Self) !Self
        pub fn fill(self: Self, value: T) !Self

        // ---- NumPy-style Broadcast Elementwise ----
        pub fn broadcastAdd(self: Self, other: Self) !Self
        pub fn broadcastSub(self: Self, other: Self) !Self
        pub fn broadcastMul(self: Self, other: Self) !Self
        pub fn broadcastDiv(self: Self, other: Self) !Self

        // ---- Reductions ----
        pub fn sum(self: Self) !T
        pub fn mean(self: Self) !T
        pub fn max(self: Self) !T
        pub fn min(self: Self) !T
        pub fn sumAxis(self: Self, axis: usize) !Self
        pub fn maxAxis(self: Self, axis: usize) !Self

        // ---- Matrix Multiplication ----
        pub fn matmul(self: Self, other: Self) !Self           // 2-D: [M,K] @ [K,N] -> [M,N]
        pub fn batchedMatmul(self: Self, other: Self) !Self    // 3-D/4-D: [B,M,K] @ [B,K,N] -> [B,M,N]

        // ---- Selection & Structural ----
        pub fn slice(self: Self, starts: []const usize, ends: []const usize) !Self
        pub fn concat(self: Self, other: Self, axis: usize) !Self
    };
}
```

## `cuda.Shape`

```zig
pub const MAX_DIMS = 8;

pub const Shape = struct {
    dims: [MAX_DIMS]usize,
    ndim: usize,

    pub fn init(shape_slice: []const usize) !Shape
    pub fn totalElements(self: Shape) usize
    pub fn computeContiguousStrides(self: Shape, strides_out: []usize) void
    pub fn eq(self: Shape, other: Shape) bool
    pub fn broadcastWith(self: Shape, other: Shape) !Shape
    pub fn permute(self: Shape, perm: []const usize) !Shape
};
```

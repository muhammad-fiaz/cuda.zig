---
title: Tensor API
description: High-level Tensor(T) API — elementwise, reduction, matmul, and shape operations in cuda.zig.
---

# Tensor API

## `cuda.Tensor(T)`

```zig
pub fn Tensor(comptime T: type) type {
    return struct {
        /// Shape: up to 4 dimensions. Unused dimensions are 1.
        shape: [4]usize,
        buf: DeviceBuffer(T),

        // ---- Lifecycle ----

        /// Allocate a tensor with the given shape (1–4 dimensions).
        pub fn init(allocator: std.mem.Allocator, shape: anytype) !@This()

        /// Free device memory.
        pub fn deinit(self: *@This()) void

        // ---- Data transfer ----

        /// Copy a flat host slice into the tensor.
        pub fn copyFromHost(self: @This(), src: []const T) !void

        /// Copy the tensor into a flat host slice.
        pub fn copyToHost(self: @This(), dst: []T) !void

        // ---- Elementwise (in-place) ----

        pub fn add_scalar(self: @This(), val: T) !void
        pub fn sub_scalar(self: @This(), val: T) !void
        pub fn mul_scalar(self: @This(), val: T) !void
        pub fn scale(self: @This(), factor: T) !void

        // ---- Elementwise (binary, out-of-place) ----

        pub fn add(a: @This(), b: @This(), c: @This()) !void
        pub fn sub(a: @This(), b: @This(), c: @This()) !void
        pub fn mul(a: @This(), b: @This(), c: @This()) !void
        pub fn div(a: @This(), b: @This(), c: @This()) !void

        // ---- Reductions ----

        pub fn sum(self: @This()) !T
        pub fn max(self: @This()) !T
        pub fn min(self: @This()) !T
        pub fn mean(self: @This()) !T

        // ---- Matrix multiplication ----

        /// C = A @ B  (A: [M,K], B: [K,N], C: [M,N])
        pub fn matmul(a: @This(), b: @This(), c: @This()) !void

        // ---- Async variants ----

        pub fn add_async(a: @This(), b: @This(), c: @This(), stream: Stream) !void
        pub fn matmul_async(a: @This(), b: @This(), c: @This(), stream: Stream) !void

        // ---- Utility ----

        /// Total number of elements.
        pub fn numel(self: @This()) usize

        /// Raw device pointer.
        pub fn devicePtr(self: @This()) CUdeviceptr
    };
}
```

## Supported Types

| Type | Elementwise | Matmul (Zig) | Matmul (cuBLAS) |
|---|---|---|---|
| `f32` | ✅ | ✅ | ✅ |
| `f64` | ✅ | ✅ | ✅ |
| `i32` | ✅ | ❌ | ❌ |
| `i64` | ✅ | ❌ | ❌ |

## Shape Conventions

```zig
// 1D
try cuda.Tensor(f32).init(allocator, .{1024})

// 2D
try cuda.Tensor(f32).init(allocator, .{ 128, 64 })

// 3D batch
try cuda.Tensor(f32).init(allocator, .{ 8, 128, 64 })
```

Internally shapes are stored as `[4]usize` (trailing dimensions default to 1).

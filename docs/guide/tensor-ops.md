---
title: Tensor Operations
description: High-level Tensor(T) API with elementwise ops, reductions, matmul, and optional cuBLAS in cuda.zig.
---

# Tensor Operations

`cuda.Tensor(T)` is a high-level, typed multi-dimensional array that runs on either the GPU or the CPU fallback backend.

## Creating a Tensor

```zig
// 1D tensor of 1024 f32 elements
var t = try cuda.Tensor(f32).init(allocator, .{1024});
defer t.deinit();

// 2D tensor: 128 rows × 64 columns
var mat = try cuda.Tensor(f32).init(allocator, .{ 128, 64 });
defer mat.deinit();
```

Shape is a comptime-known or runtime tuple of `usize`.

## Filling from Host

```zig
const data = [_]f32{1.0} ** 1024;
try t.copyFromHost(&data);
```

## Reading Back

```zig
var out: [1024]f32 = undefined;
try t.copyToHost(&out);
```

## Elementwise Operations

```zig
// in-place: t[i] += 1.0
try t.add_scalar(1.0);

// in-place: t[i] *= scale
try t.scale(2.0);

// element-wise add: c[i] = a[i] + b[i]
try cuda.Tensor(f32).add(a, b, c);

// element-wise multiply
try cuda.Tensor(f32).mul(a, b, c);
```

## Reductions

```zig
const sum = try t.sum();
const max = try t.max();
const min = try t.min();
```

## Matrix Multiplication

```zig
// C = A @ B   (shape: [M, K] × [K, N] → [M, N])
var a = try cuda.Tensor(f32).init(allocator, .{ 128, 256 });
var b = try cuda.Tensor(f32).init(allocator, .{ 256, 64 });
var c = try cuda.Tensor(f32).init(allocator, .{ 128, 64 });
defer a.deinit();
defer b.deinit();
defer c.deinit();

try cuda.Tensor(f32).matmul(a, b, c);
```

On GPU: uses the custom Zig kernel in `tensor/ops/matmul.zig`.  
With `-Denable_cublas=true`: delegates to `cublasGemmEx` for maximum throughput.

## Data Types

`Tensor(T)` is generic over any scalar type. Supported:

| Type | Notes |
|---|---|
| `f32` | Single precision — primary |
| `f64` | Double precision |
| `i32` | 32-bit signed integer |
| `i64` | 64-bit signed integer |

> [!NOTE]
> cuBLAS support is currently limited to `f32` and `f64`.

## Async Operations

All tensor ops accept an optional `stream` argument to run asynchronously:

```zig
try t.add_scalar_async(1.0, stream);
try cuda.Tensor(f32).matmul_async(a, b, c, stream);
try stream.sync();
```

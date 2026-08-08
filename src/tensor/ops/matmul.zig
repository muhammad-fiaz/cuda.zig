//! Matrix multiplication ops for Tensor — 2-D and batched N-D.

const std = @import("std");
const cpu_backend = @import("../../fallback/cpu_backend.zig");

/// 2-D matrix multiply: C[M,N] = A[M,K] @ B[K,N].
pub fn matmul(
    comptime T: type,
    c: []T,
    a: []const T,
    b: []const T,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (c.len != m * n or a.len != m * k or b.len != k * n) return error.InvalidValue;
    cpu_backend.matmulNaive(T, c, a, b, m, n, k);
}

/// Batched matrix multiply: C[B,M,N] = A[B,M,K] @ B[B,K,N].
pub fn batchedMatmul(
    comptime T: type,
    c: []T,
    a: []const T,
    b: []const T,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (c.len != batch * m * n or a.len != batch * m * k or b.len != batch * k * n) return error.InvalidValue;
    cpu_backend.batchedMatmulNaive(T, c, a, b, batch, m, n, k);
}

test "matmul 2D" {
    // 2x3 @ 3x2 -> 2x2
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 1, 2, 3 };
    var c = [_]f32{ 0, 0, 0, 0 };
    try matmul(f32, &c, &a, &b, 2, 2, 3);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 31, 19, 85, 55 }, &c);
}

test "batchedMatmul 3D" {
    // batch=1, 2x2 @ 2x2
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 0, 0, 1 }; // identity
    var c = [_]f32{ 0, 0, 0, 0 };
    try batchedMatmul(f32, &c, &a, &b, 1, 2, 2, 2);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &c);
}

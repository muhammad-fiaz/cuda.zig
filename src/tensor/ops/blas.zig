//! Optional cuBLAS accelerated matmul backend.

const std = @import("std");
const err = @import("../../core/error.zig");

pub fn isAvailable() bool {
    return false; // cuBLAS feature-gated build option
}

pub fn gemm(
    comptime T: type,
    c: []T,
    a: []const T,
    b: []const T,
    m: usize,
    n: usize,
    k: usize,
) err.CudaError!void {
    _ = c;
    _ = a;
    _ = b;
    _ = m;
    _ = n;
    _ = k;
    return error.NotYetImplemented;
}

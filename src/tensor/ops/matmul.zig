//! Matrix multiplication for Tensor.

const std = @import("std");
const cpu_backend = @import("../../fallback/cpu_backend.zig");
const err = @import("../../core/error.zig");

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

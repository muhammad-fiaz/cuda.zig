//! CPU Backend implementations for numerical tensor ops (elementwise, reduction, matmul).
//!
//! Provides pure Zig CPU implementations matching the GPU op signatures.
//! Runs unconditionally when no CUDA GPU or driver is available.

const std = @import("std");

pub fn elementwiseAdd(comptime T: type, dst: []T, a: []const T, b: []const T) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, 0..) |*out, i| {
        out.* = a[i] + b[i];
    }
}

pub fn elementwiseSub(comptime T: type, dst: []T, a: []const T, b: []const T) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, 0..) |*out, i| {
        out.* = a[i] - b[i];
    }
}

pub fn elementwiseMul(comptime T: type, dst: []T, a: []const T, b: []const T) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, 0..) |*out, i| {
        out.* = a[i] * b[i];
    }
}

pub fn elementwiseDiv(comptime T: type, dst: []T, a: []const T, b: []const T) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, 0..) |*out, i| {
        out.* = a[i] / b[i];
    }
}

pub fn elementwiseRelu(comptime T: type, dst: []T, a: []const T) void {
    std.debug.assert(dst.len == a.len);
    const zero: T = 0;
    for (dst, 0..) |*out, i| {
        out.* = @max(zero, a[i]);
    }
}

pub fn reduceSum(comptime T: type, data: []const T) T {
    var acc: T = 0;
    for (data) |v| acc += v;
    return acc;
}

pub fn reduceMean(comptime T: type, data: []const T) T {
    if (data.len == 0) return 0;
    const s = reduceSum(T, data);
    return switch (@typeInfo(T)) {
        .float => s / @as(T, @floatFromInt(data.len)),
        .int => @divTrunc(s, @as(T, @intCast(data.len))),
        else => s,
    };
}

pub fn matmulNaive(
    comptime T: type,
    c: []T,
    a: []const T,
    b: []const T,
    m: usize,
    n: usize,
    k: usize,
) void {
    std.debug.assert(c.len == m * n);
    std.debug.assert(a.len == m * k);
    std.debug.assert(b.len == k * n);

    @memset(c, 0);
    for (0..m) |i| {
        for (0..k) |p| {
            const a_val = a[i * k + p];
            for (0..n) |j| {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
}

test "CPU backend elementwise ops" {
    const a = [_]f32{ 1.0, 2.0, 3.0, -4.0 };
    const b = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    var dst = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    elementwiseAdd(f32, &dst, &a, &b);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 11.0, 22.0, 33.0, 36.0 }, &dst);

    elementwiseSub(f32, &dst, &b, &a);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9.0, 18.0, 27.0, 44.0 }, &dst);

    elementwiseRelu(f32, &dst, &a);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 0.0 }, &dst);
}

test "CPU backend matmul" {
    // 2x3 * 3x2 -> 2x2
    const a = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const b = [_]f32{
        7, 8,
        9, 1,
        2, 3,
    };
    var c = [_]f32{ 0, 0, 0, 0 };

    matmulNaive(f32, &c, &a, &b, 2, 2, 3);
    // [1*7+2*9+3*2, 1*8+2*1+3*3] = [31, 19]
    // [4*7+5*9+6*2, 4*8+5*1+6*3] = [85, 55]
    try std.testing.expectEqualSlices(f32, &[_]f32{ 31, 19, 85, 55 }, &c);
}

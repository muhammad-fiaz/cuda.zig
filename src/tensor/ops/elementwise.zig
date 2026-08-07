//! Elementwise ops for Tensor — wraps cpu_backend dispatch.

const std = @import("std");
const cpu_backend = @import("../../fallback/cpu_backend.zig");

pub fn add(comptime T: type, dst: []T, a: []const T, b: []const T) !void {
    if (dst.len != a.len or a.len != b.len) return error.InvalidValue;
    cpu_backend.elementwiseAdd(T, dst, a, b);
}

pub fn sub(comptime T: type, dst: []T, a: []const T, b: []const T) !void {
    if (dst.len != a.len or a.len != b.len) return error.InvalidValue;
    cpu_backend.elementwiseSub(T, dst, a, b);
}

pub fn mul(comptime T: type, dst: []T, a: []const T, b: []const T) !void {
    if (dst.len != a.len or a.len != b.len) return error.InvalidValue;
    cpu_backend.elementwiseMul(T, dst, a, b);
}

pub fn div(comptime T: type, dst: []T, a: []const T, b: []const T) !void {
    if (dst.len != a.len or a.len != b.len) return error.InvalidValue;
    cpu_backend.elementwiseDiv(T, dst, a, b);
}

pub fn relu(comptime T: type, dst: []T, a: []const T) !void {
    if (dst.len != a.len) return error.InvalidValue;
    cpu_backend.elementwiseRelu(T, dst, a);
}

pub fn neg(comptime T: type, dst: []T, a: []const T) !void {
    if (dst.len != a.len) return error.InvalidValue;
    cpu_backend.elementwiseNeg(T, dst, a);
}

test "elementwise add, sub, mul, div, relu, neg" {
    const a = [_]f32{ 2.0, -4.0, 6.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    var dst: [3]f32 = undefined;

    try add(f32, &dst, &a, &b);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3.0, -2.0, 9.0 }, &dst);

    try sub(f32, &dst, &a, &b);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, -6.0, 3.0 }, &dst);

    try mul(f32, &dst, &a, &b);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2.0, -8.0, 18.0 }, &dst);

    try div(f32, &dst, &a, &b);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2.0, -2.0, 2.0 }, &dst);

    try relu(f32, &dst, &a);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2.0, 0.0, 6.0 }, &dst);

    try neg(f32, &dst, &a);
    try std.testing.expectEqualSlices(f32, &[_]f32{ -2.0, 4.0, -6.0 }, &dst);
}

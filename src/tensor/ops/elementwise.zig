//! Elementwise ops for Tensor.

const std = @import("std");
const cpu_backend = @import("../../fallback/cpu_backend.zig");
const err = @import("../../core/error.zig");

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

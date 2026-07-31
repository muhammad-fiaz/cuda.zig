//! Reduction ops for Tensor.

const std = @import("std");
const cpu_backend = @import("../../fallback/cpu_backend.zig");

pub fn sum(comptime T: type, data: []const T) T {
    return cpu_backend.reduceSum(T, data);
}

pub fn mean(comptime T: type, data: []const T) T {
    return cpu_backend.reduceMean(T, data);
}

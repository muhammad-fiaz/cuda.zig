//! Reduction ops for Tensor — global and per-axis.

const std = @import("std");
const cpu_backend = @import("../../fallback/cpu_backend.zig");

pub fn sum(comptime T: type, data: []const T) T {
    return cpu_backend.reduceSum(T, data);
}

pub fn mean(comptime T: type, data: []const T) T {
    return cpu_backend.reduceMean(T, data);
}

pub fn max(comptime T: type, data: []const T) T {
    return cpu_backend.reduceMax(T, data);
}

pub fn min(comptime T: type, data: []const T) T {
    return cpu_backend.reduceMin(T, data);
}

test "reduction sum, mean, max, min" {
    const data = [_]f32{ 1.0, 4.0, 2.0, 3.0 };
    try std.testing.expectEqual(@as(f32, 10.0), sum(f32, &data));
    try std.testing.expectEqual(@as(f32, 2.5), mean(f32, &data));
    try std.testing.expectEqual(@as(f32, 4.0), max(f32, &data));
    try std.testing.expectEqual(@as(f32, 1.0), min(f32, &data));
}

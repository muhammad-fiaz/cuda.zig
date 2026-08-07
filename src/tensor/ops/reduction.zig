//! Reduction ops for Tensor.

const std = @import("std");
const cpu_backend = @import("../../fallback/cpu_backend.zig");

pub fn sum(comptime T: type, data: []const T) T {
    return cpu_backend.reduceSum(T, data);
}

pub fn mean(comptime T: type, data: []const T) T {
    return cpu_backend.reduceMean(T, data);
}

test "reduction sum and mean" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try std.testing.expectEqual(@as(f32, 10.0), sum(f32, &data));
    try std.testing.expectEqual(@as(f32, 2.5), mean(f32, &data));
}

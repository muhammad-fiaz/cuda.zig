const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig CPU Fallback Example ===\n", .{});

    std.debug.print("GPU is available: {}\n", .{cuda.isAvailable()});

    const allocator = std.heap.page_allocator;

    var buf = try cuda.DeviceBuffer(f32).alloc(8);
    defer buf.free();

    const input = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try buf.copyFromHost(&input);

    const output = try allocator.alloc(f32, 8);
    defer allocator.free(output);

    try buf.copyToHost(output);

    std.debug.print("Successfully executed operations via CPU fallback path:\n{any}\n", .{output});
}

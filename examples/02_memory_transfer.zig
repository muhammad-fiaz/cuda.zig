const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Memory Transfer Example ===\n", .{});

    const count = 1024 * 1024; // 1 million f32 elements (4 MB)
    std.debug.print("Allocating host and device buffers for {d} elements (4 MB)...\n", .{count});

    const host_src = try std.heap.page_allocator.alloc(f32, count);
    defer std.heap.page_allocator.free(host_src);
    for (host_src, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    var dev_buf = try cuda.DeviceBuffer(f32).alloc(count);
    defer dev_buf.free();

    std.debug.print("Copying Host -> Device...\n", .{});
    try dev_buf.copyFromHost(host_src);

    const host_dst = try std.heap.page_allocator.alloc(f32, count);
    defer std.heap.page_allocator.free(host_dst);

    std.debug.print("Copying Device -> Host...\n", .{});
    try dev_buf.copyToHost(host_dst);

    std.debug.print("Verifying data...\n", .{});
    for (host_dst, 0..) |val, i| {
        if (val != host_src[i]) {
            std.debug.print("Mismatch at index {d}: expected {d}, got {d}\n", .{ i, host_src[i], val });
            return;
        }
    }
    std.debug.print("Memory transfer verification successful!\n", .{});
}

const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Advanced Managed Memory & Prefetch Example ===\n", .{});

    var buf = try cuda.UnifiedBuffer(f32).alloc(2048, 0x01); // global attach
    defer buf.free();

    const data = buf.slice();
    for (data, 0..) |*item, idx| {
        item.* = @floatFromInt(idx);
    }

    if (cuda.isAvailable()) {
        var stream = try cuda.Stream.init();
        defer stream.deinit();

        try buf.prefetchToDevice(0, stream);
        try stream.synchronize();
        std.debug.print("Prefetched 2048 elements of managed memory to GPU Device 0.\n", .{});

        try buf.prefetchToHost(stream);
        try stream.synchronize();
        std.debug.print("Prefetched managed memory back to Host CPU.\n", .{});
    } else {
        std.debug.print("Running in CPU fallback mode; unified memory accessed seamlessly on CPU.\n", .{});
    }

    std.debug.print("First element: {d:.1}, Last element: {d:.1}\n", .{ data[0], data[data.len - 1] });
}

const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Multi-GPU Example ===\n", .{});

    if (!cuda.isAvailable()) {
        std.debug.print("No CUDA available.\n", .{});
        return;
    }

    const count = try cuda.deviceCount();
    std.debug.print("Available devices: {d}\n", .{count});

    if (count < 2) {
        std.debug.print("Multi-GPU operations require at least 2 devices.\n", .{});
        return;
    }

    const can_access = try cuda.transfer.canAccessPeer(0, 1);
    std.debug.print("Device 0 can access Device 1 peer memory: {}\n", .{can_access});
}

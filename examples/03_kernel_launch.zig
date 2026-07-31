const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Kernel Launch Example ===\n", .{});

    if (!cuda.isAvailable()) {
        std.debug.print("No CUDA GPU available; skipping real kernel launch.\n", .{});
        return;
    }

    std.debug.print("Kernel launch requires a compiled PTX module.\n", .{});
}

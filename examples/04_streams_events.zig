const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Streams and Events Example ===\n", .{});

    if (!cuda.isAvailable()) {
        std.debug.print("CUDA is not available; streams & events fall back gracefully.\n", .{});
        return;
    }

    var stream = try cuda.Stream.init();
    defer stream.deinit();

    var start_evt = try cuda.Event.init();
    defer start_evt.deinit();

    var stop_evt = try cuda.Event.init();
    defer stop_evt.deinit();

    try start_evt.record(stream);
    try stream.synchronize();
    try stop_evt.record(stream);
    try stop_evt.synchronize();

    const elapsed_ms = try cuda.Event.elapsed(start_evt, stop_evt);
    std.debug.print("Elapsed stream execution time: {d:.4} ms\n", .{elapsed_ms});
}

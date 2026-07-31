//! CUDA Stream and Event high-level wrappers.
//!
//! Provides `Stream` and `Event` struct abstractions for stream-ordered
//! operations and GPU timing.

const std = @import("std");
const loader = @import("../core/loader.zig");
const stream_rt = @import("../runtime/stream.zig");
const event_rt = @import("../runtime/event.zig");
const ffi = @import("../runtime/ffi.zig");
const err = @import("../core/error.zig");

/// High-level wrapper around `cudaStream_t`.
pub const Stream = struct {
    handle: ffi.Stream,

    /// Creates a new CUDA stream.
    pub fn init() err.CudaError!Stream {
        if (!loader.isAvailable()) return error.NotInitialized;
        const h = try stream_rt.createStream();
        return .{ .handle = h };
    }

    /// Creates a new non-blocking CUDA stream.
    pub fn initNonBlocking() err.CudaError!Stream {
        if (!loader.isAvailable()) return error.NotInitialized;
        const h = try stream_rt.createStreamWithFlags(ffi.StreamFlags.non_blocking);
        return .{ .handle = h };
    }

    /// Destroys the stream. Safe to call multiple times (idempotent).
    pub fn deinit(self: *Stream) void {
        if (self.handle == null) return;
        _ = stream_rt.destroyStream(self.handle) catch {};
        self.handle = null;
    }

    /// Blocks host until all work in this stream completes.
    pub fn synchronize(self: Stream) err.CudaError!void {
        if (self.handle == null) return;
        try stream_rt.synchronizeStream(self.handle);
    }

    /// Queries whether all operations in this stream have completed.
    pub fn query(self: Stream) err.CudaError!void {
        if (self.handle == null) return;
        try stream_rt.queryStream(self.handle);
    }
};

test "Stream init without CUDA" {
    if (!loader.isAvailable()) {
        const r = Stream.init();
        try std.testing.expectError(error.NotInitialized, r);
        return;
    }
    var s = try Stream.init();
    defer s.deinit();
    try s.synchronize();
}

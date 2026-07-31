//! High-level Event wrapper.

const std = @import("std");
const loader = @import("../core/loader.zig");
const event_rt = @import("../runtime/event.zig");
const stream_mod = @import("stream.zig");
const ffi = @import("../runtime/ffi.zig");
const err = @import("../core/error.zig");

/// High-level wrapper around `cudaEvent_t`.
pub const Event = struct {
    handle: ffi.Event,

    /// Creates a new CUDA event.
    pub fn init() err.CudaError!Event {
        if (!loader.isAvailable()) return error.NotInitialized;
        const h = try event_rt.createEvent();
        return .{ .handle = h };
    }

    /// Destroys the event. Safe to call multiple times (idempotent).
    pub fn deinit(self: *Event) void {
        if (self.handle == null) return;
        event_rt.destroyEvent(self.handle);
        self.handle = null;
    }

    /// Records this event in `stream`.
    pub fn record(self: Event, stream: stream_mod.Stream) err.CudaError!void {
        try event_rt.recordEvent(self.handle, stream.handle);
    }

    /// Synchronizes host until event is recorded.
    pub fn synchronize(self: Event) err.CudaError!void {
        try event_rt.synchronizeEvent(self.handle);
    }

    /// Returns elapsed time in milliseconds between `start` and `end`.
    pub fn elapsed(start_evt: Event, end_evt: Event) err.CudaError!f32 {
        return event_rt.elapsedTime(start_evt.handle, end_evt.handle);
    }
};

test "Event init without CUDA" {
    if (!loader.isAvailable()) {
        const r = Event.init();
        try std.testing.expectError(error.NotInitialized, r);
        return;
    }
    var e = try Event.init();
    defer e.deinit();
}

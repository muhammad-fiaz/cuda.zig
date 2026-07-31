//! CUDA event lifecycle bindings.
//!
//! Events mark a point in a stream's execution timeline and can be used to:
//!   measure elapsed time between two points (GPU-side timing),
//!   synchronize the host with device progress without blocking the entire device,
//!   synchronize across streams (stream-wait-event).
//!
//! All timing events carry a hardware timestamp that is set when the GPU
//! reaches the recorded position in the stream. The `elapsed` function
//! computes the time in milliseconds between two such recorded points.

const std = @import("std");
const loader = @import("../core/loader.zig");
const ffi = @import("ffi.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");

/// Creates an event with default flags (timing enabled, non-blocking query).
///
/// The event is associated with the current device of the calling thread.
pub fn createEvent() err.CudaError!ffi.Event {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.EventCreateFn, "cudaEventCreate")) |f| {
        var event: ffi.Event = null;
        try result.checkRuntime(f(&event));
        return event;
    }
    if (ldr.getDriverSymbol(*const fn (*?*anyopaque, c_uint) callconv(.c) c_int, "cuEventCreate")) |f| {
        var event: ?*anyopaque = null;
        try result.checkDriver(f(&event, 0));
        return event;
    }
    return error.NotInitialized;
}

pub fn createEventWithFlags(flags: c_uint) err.CudaError!ffi.Event {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.EventCreateWithFlagsFn, "cudaEventCreateWithFlags")) |f| {
        var event: ffi.Event = null;
        try result.checkRuntime(f(&event, flags));
        return event;
    }
    if (ldr.getDriverSymbol(*const fn (*?*anyopaque, c_uint) callconv(.c) c_int, "cuEventCreate")) |f| {
        var event: ?*anyopaque = null;
        try result.checkDriver(f(&event, flags));
        return event;
    }
    return error.NotInitialized;
}

/// Destroys an event and releases its resources.
///
/// Destroying an event that has been recorded but not yet reached by the GPU
/// is valid; the driver waits until the event is no longer needed.
pub fn destroyEvent(event: ffi.Event) void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.EventDestroyFn, "cudaEventDestroy")) |f| {
        _ = f(event);
        return;
    }
    if (ldr.getDriverSymbol(*const fn (ffi.Event) callconv(.c) c_int, "cuEventDestroy_v2")) |f| {
        _ = f(event);
    } else if (ldr.getDriverSymbol(*const fn (ffi.Event) callconv(.c) c_int, "cuEventDestroy")) |f| {
        _ = f(event);
    }
}

pub fn recordEvent(event: ffi.Event, stream: ffi.Stream) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.EventRecordFn, "cudaEventRecord")) |f| {
        try result.checkRuntime(f(event, stream));
        return;
    }
    if (ldr.getDriverSymbol(*const fn (ffi.Event, ffi.Stream) callconv(.c) c_int, "cuEventRecord")) |f| {
        try result.checkDriver(f(event, stream));
        return;
    }
    return error.NotInitialized;
}

pub fn synchronizeEvent(event: ffi.Event) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.EventSynchronizeFn, "cudaEventSynchronize")) |f| {
        try result.checkRuntime(f(event));
        return;
    }
    if (ldr.getDriverSymbol(*const fn (ffi.Event) callconv(.c) c_int, "cuEventSynchronize")) |f| {
        try result.checkDriver(f(event));
        return;
    }
    return error.NotInitialized;
}

/// Returns the elapsed time in milliseconds between the recording of `start`
/// and `end`.
///
/// Both events must have been recorded on the same device. Both must have
/// completed (i.e., the GPU has reached their recorded positions in the stream).
/// If either event has not completed, this function blocks until both are done.
///
/// The resolution is approximately 0.5 microseconds on most NVIDIA hardware.
/// Do not use timing events (those created with `ffi.EventFlags.disable_timing`)
/// with this function; doing so returns `error.InvalidHandle`.
pub fn elapsedTime(start: ffi.Event, end: ffi.Event) err.CudaError!f32 {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.EventElapsedTimeFn, "cudaEventElapsedTime")) |f| {
        var ms: f32 = 0;
        try result.checkRuntime(f(&ms, start, end));
        return ms;
    }
    if (ldr.getDriverSymbol(*const fn (*f32, ffi.Event, ffi.Event) callconv(.c) c_int, "cuEventElapsedTime")) |f| {
        var ms: f32 = 0;
        try result.checkDriver(f(&ms, start, end));
        return ms;
    }
    return error.NotInitialized;
}

/// Queries whether `event` has been recorded without blocking.
///
/// Returns `void` if the event has been recorded, or `error.NotReady` if the
/// GPU has not yet reached the event's recorded position.
pub fn queryEvent(event: ffi.Event) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.EventQueryFn, "cudaEventQuery") orelse
        return error.NotInitialized;
    try result.checkRuntime(f(event));
}

test "event create/destroy skips without CUDA" {
    if (!loader.isAvailable()) return error.SkipZigTest;
    const e = try createEvent();
    destroyEvent(e);
}

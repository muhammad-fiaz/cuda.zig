//! CUDA stream lifecycle bindings.
//!
//! Streams provide a mechanism for ordering device operations and enabling
//! asynchronous execution. Operations submitted to the same stream execute
//! in issue order. Operations on different streams may overlap.
//!
//! The default stream (`null`) is a special implicit stream that synchronizes
//! with all other streams. For production use, always create explicit streams.

const std = @import("std");
const loader = @import("../core/loader.zig");
const ffi = @import("ffi.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");

/// Creates a new CUDA stream with default flags.
///
/// The new stream is associated with the current device of the calling thread.
/// Returns a `cudaStream_t` handle (`ffi.Stream`). Destroy with `destroyStream`.
pub fn createStream() err.CudaError!ffi.Stream {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.StreamCreateFn, "cudaStreamCreate")) |f| {
        var stream: ffi.Stream = null;
        try result.checkRuntime(f(&stream));
        return stream;
    }
    if (ldr.getDriverSymbol(*const fn (*?*anyopaque, c_uint) callconv(.c) c_int, "cuStreamCreate")) |f| {
        var stream: ?*anyopaque = null;
        try result.checkDriver(f(&stream, 0));
        return stream;
    }
    return error.NotInitialized;
}

pub fn createStreamWithFlags(flags: c_uint) err.CudaError!ffi.Stream {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.StreamCreateWithFlagsFn, "cudaStreamCreateWithFlags")) |f| {
        var stream: ffi.Stream = null;
        try result.checkRuntime(f(&stream, flags));
        return stream;
    }
    if (ldr.getDriverSymbol(*const fn (*?*anyopaque, c_uint) callconv(.c) c_int, "cuStreamCreate")) |f| {
        var stream: ?*anyopaque = null;
        try result.checkDriver(f(&stream, flags));
        return stream;
    }
    return error.NotInitialized;
}

/// Destroys a stream and releases its resources.
///
/// The function returns once the stream has been destroyed on the host side.
/// Any device operations already submitted to the stream will still complete.
pub fn destroyStream(stream: ffi.Stream) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.StreamDestroyFn, "cudaStreamDestroy")) |f| {
        try result.checkRuntime(f(stream));
        return;
    }
    if (ldr.getDriverSymbol(*const fn (ffi.Stream) callconv(.c) c_int, "cuStreamDestroy_v2")) |f| {
        try result.checkDriver(f(stream));
        return;
    }
    if (ldr.getDriverSymbol(*const fn (ffi.Stream) callconv(.c) c_int, "cuStreamDestroy")) |f| {
        try result.checkDriver(f(stream));
        return;
    }
    return error.NotInitialized;
}

/// Blocks the calling CPU thread until all operations in `stream` have completed.
///
/// Operations submitted after this call returns may still be pending.
pub fn synchronizeStream(stream: ffi.Stream) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.StreamSynchronizeFn, "cudaStreamSynchronize")) |f| {
        try result.checkRuntime(f(stream));
        return;
    }
    if (ldr.getDriverSymbol(*const fn (ffi.Stream) callconv(.c) c_int, "cuStreamSynchronize")) |f| {
        try result.checkDriver(f(stream));
        return;
    }
    return error.NotInitialized;
}

/// Queries whether all operations in `stream` have completed without blocking.
///
/// Returns `void` if all operations are done, or `error.NotReady` if work is
/// still in progress. Any other error indicates a failure.
pub fn queryStream(stream: ffi.Stream) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.StreamQueryFn, "cudaStreamQuery") orelse
        return error.NotInitialized;
    try result.checkRuntime(f(stream));
}

/// Makes `stream` wait until `event` has been recorded.
///
/// The stream will not begin executing any operations submitted after this
/// call until the event has been recorded. `flags` must be `0` (reserved for
/// future use). This call does not block the host.
pub fn streamWaitEvent(stream: ffi.Stream, event: ffi.Event, flags: c_uint) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.StreamWaitEventFn, "cudaStreamWaitEvent") orelse
        return error.NotInitialized;
    try result.checkRuntime(f(stream, event, flags));
}

test "stream create/destroy skips without CUDA" {
    if (!loader.isAvailable()) return error.SkipZigTest;
    const s = try createStream();
    try destroyStream(s);
}

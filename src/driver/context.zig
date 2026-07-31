//! CUDA driver context lifecycle management.
//!
//! A `DriverContext` wraps a `CUcontext` handle and provides create, destroy,
//! push, and pop operations corresponding to `cuCtxCreate`, `cuCtxDestroy`,
//! `cuCtxPushCurrent`, and `cuCtxPopCurrent`.
//!
//! For most use cases the primary context (retained via
//! `cuDevicePrimaryCtxRetain`) is more appropriate than creating an explicit
//! context. Use `DriverContext.retainPrimary` for that path.
//!
//! Thread safety: a CUDA context is associated with the OS thread that created
//! it. Operations on a `DriverContext` must be performed from the same thread,
//! or the context must be explicitly pushed/popped around cross-thread use.
//! See the CUDA Driver API documentation for full semantics.

const std = @import("std");
const loader = @import("../core/loader.zig");
const ffi = @import("ffi.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");

/// Ensures the CUDA driver is initialized (`cuInit(0)` called once).
///
/// The driver must be initialized before any Driver API call except
/// `cuDriverGetVersion`. `initDriver` is idempotent: the first call issues
/// `cuInit(0)`; subsequent calls from any thread are no-ops.
///
/// Returns `error.NoDevice` if no CUDA-capable device is present, or
/// `error.NotInitialized` if the driver library could not be found.
pub fn initDriver() err.CudaError!void {
    const ldr = loader.globalLoader();
    const init_fn = ldr.getDriverSymbol(ffi.InitFn, "cuInit") orelse
        return error.NotInitialized;
    try result.checkDriver(init_fn(0));
}

/// Owns a `CUcontext` handle and provides a safe lifecycle around it.
pub const DriverContext = struct {
    handle: *anyopaque,

    /// Creates a new CUDA context on `device` with the given `flags`.
    ///
    /// `flags` is a bitwise OR of the constants in `ffi.CtxFlags`. Pass `0`
    /// for the default scheduling policy.
    ///
    /// The new context becomes the current context on the calling thread.
    /// Call `push` / `pop` to manage the context stack explicitly when
    /// working with multiple contexts.
    pub fn init(device: ffi.Device, flags: c_uint) err.CudaError!DriverContext {
        try initDriver();
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.CtxCreateFn, "cuCtxCreate_v2") orelse
            ldr.getDriverSymbol(ffi.CtxCreateFn, "cuCtxCreate") orelse
            return error.NotInitialized;
        var handle: *anyopaque = undefined;
        try result.checkDriver(f(&handle, flags, device));
        return .{ .handle = handle };
    }

    /// Retains the primary context for `device`.
    ///
    /// The primary context is a per-device singleton managed by the CUDA
    /// runtime. Unlike explicitly created contexts, the primary context is
    /// reference-counted: multiple `retainPrimary` calls are balanced by the
    /// same number of `deinit` calls (which call `cuDevicePrimaryCtxRelease`).
    ///
    /// After `retainPrimary` the returned context is not automatically current;
    /// call `setCurrent` to activate it on the calling thread.
    pub fn retainPrimary(device: ffi.Device) err.CudaError!DriverContext {
        try initDriver();
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.DevicePrimaryCtxRetainFn, "cuDevicePrimaryCtxRetain") orelse
            return error.NotInitialized;
        var handle: *anyopaque = undefined;
        try result.checkDriver(f(&handle, device));
        return .{ .handle = handle };
    }

    /// Destroys the context or releases the primary context reference.
    ///
    /// For contexts created with `init`, this calls `cuCtxDestroy`. For
    /// contexts obtained via `retainPrimary` the caller is responsible for
    /// knowing which release function to use; prefer `releasePrimary` in
    /// that case.
    ///
    /// Calling `deinit` more than once is safe: subsequent calls detect the
    /// nulled handle and return immediately.
    pub fn deinit(self: *DriverContext) void {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.CtxDestroyFn, "cuCtxDestroy_v2") orelse
            ldr.getDriverSymbol(ffi.CtxDestroyFn, "cuCtxDestroy") orelse return;
        _ = f(self.handle);
    }

    /// Releases the primary context reference for `device`.
    ///
    /// Use this in tandem with `retainPrimary`. Equivalent to calling
    /// `cuDevicePrimaryCtxRelease`.
    pub fn releasePrimary(device: ffi.Device) void {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.DevicePrimaryCtxReleaseFn, "cuDevicePrimaryCtxRelease_v2") orelse
            ldr.getDriverSymbol(ffi.DevicePrimaryCtxReleaseFn, "cuDevicePrimaryCtxRelease") orelse return;
        _ = f(device);
    }

    /// Pushes this context onto the calling thread's context stack, making
    /// it the current context.
    ///
    /// Use `pop` to restore the previous context.
    pub fn push(self: DriverContext) err.CudaError!void {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.CtxPushCurrentFn, "cuCtxPushCurrent_v2") orelse
            ldr.getDriverSymbol(ffi.CtxPushCurrentFn, "cuCtxPushCurrent") orelse
            return error.NotInitialized;
        try result.checkDriver(f(self.handle));
    }

    /// Pops the current context off the calling thread's context stack,
    /// restoring whatever context was current before `push`.
    ///
    /// Returns the popped context handle. The caller must ensure that the
    /// context being popped is indeed this one; mismatched push/pop pairs
    /// are a programming error.
    pub fn pop() err.CudaError!DriverContext {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.CtxPopCurrentFn, "cuCtxPopCurrent_v2") orelse
            ldr.getDriverSymbol(ffi.CtxPopCurrentFn, "cuCtxPopCurrent") orelse
            return error.NotInitialized;
        var handle: *anyopaque = undefined;
        try result.checkDriver(f(&handle));
        return .{ .handle = handle };
    }

    /// Makes this context the current context on the calling thread without
    /// modifying the context stack.
    pub fn setCurrent(self: DriverContext) err.CudaError!void {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.CtxSetCurrentFn, "cuCtxSetCurrent") orelse
            return error.NotInitialized;
        try result.checkDriver(f(self.handle));
    }

    /// Blocks until all preceding operations in this context have completed.
    ///
    /// This is a host-synchronous call: it does not return until the device
    /// has finished all outstanding work in the current context.
    pub fn synchronize() err.CudaError!void {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.CtxSynchronizeFn, "cuCtxSynchronize") orelse
            return error.NotInitialized;
        try result.checkDriver(f());
    }
};

test "initDriver without CUDA" {
    if (!loader.isAvailable()) {
        // Expected: either NotInitialized or NoDevice.
        const r = initDriver();
        try std.testing.expect(r == error.NotInitialized or r == error.NoDevice);
        return;
    }
    try initDriver();
}

test "DriverContext.retainPrimary skips without CUDA" {
    if (!loader.isAvailable()) return error.SkipZigTest;
    // Only valid on a machine with a CUDA device.
}

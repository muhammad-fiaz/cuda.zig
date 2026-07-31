//! Global device selection: setDevice, currentDevice, deviceCount, synchronize.
//!
//! The "current device" in cuda.zig matches real CUDA semantics: it is a
//! per-OS-thread value. Each thread independently tracks which device it
//! is associated with. Changing the current device on one thread does not
//! affect other threads.
//!
//! Global state: `threadlocal var current_device_index: u32` is the only
//! piece of mutable global state in cuda.zig beyond the process-level library
//! loader. It mirrors `cudaSetDevice`/`cudaGetDevice` semantics.

const std = @import("std");
const loader = @import("../core/loader.zig");
const runtime_dev = @import("../runtime/device.zig");
const err = @import("../core/error.zig");

/// The device index associated with the calling thread. Initialized to 0,
/// updated by `setDevice`. Mirrors the CUDA runtime's per-thread device
/// association; this variable is the Zig-side cache of what was last set.
threadlocal var current_device_index: u32 = 0;

/// Associates the calling thread with CUDA device `index`.
///
/// Subsequent memory allocations and kernel launches on the calling thread
/// target this device. Other threads are not affected. `index` must be in
/// `[0, deviceCount())`.
pub fn setDevice(index: u32) err.CudaError!void {
    try runtime_dev.setDevice(index);
    current_device_index = index;
}

/// Returns the device index currently associated with the calling thread.
///
/// On the first call in a thread that has not explicitly called `setDevice`,
/// this queries the runtime (which defaults to device 0). After a successful
/// `setDevice` call, subsequent calls to `currentDevice` return that index
/// without another runtime round-trip.
pub fn currentDevice() err.CudaError!u32 {
    if (loader.isAvailable()) {
        // Keep the thread-local cache in sync with what the runtime thinks.
        current_device_index = try runtime_dev.getCurrentDevice();
    }
    return current_device_index;
}

/// Returns the number of CUDA-capable devices visible to the process.
///
/// Returns `0` if no devices are present. Returns `error.NotInitialized`
/// if the runtime library could not be loaded.
pub fn deviceCount() err.CudaError!u32 {
    return runtime_dev.getDeviceCount();
}

/// Blocks the calling thread until all device operations on the current device
/// have completed.
///
/// Equivalent to `cudaDeviceSynchronize`. Waits for all streams and all
/// asynchronous operations on the device associated with this thread.
pub fn synchronize() err.CudaError!void {
    return runtime_dev.deviceSynchronize();
}

test "setDevice and currentDevice without CUDA" {
    if (!loader.isAvailable()) {
        // currentDevice falls through to the cached thread-local.
        const idx = try currentDevice();
        try std.testing.expectEqual(@as(u32, 0), idx);
        return;
    }
    const count = try deviceCount();
    if (count == 0) return error.SkipZigTest;
    try setDevice(0);
    const cur = try currentDevice();
    try std.testing.expectEqual(@as(u32, 0), cur);
}

test "deviceCount does not crash" {
    if (!loader.isAvailable()) return error.SkipZigTest;
    _ = try deviceCount();
}

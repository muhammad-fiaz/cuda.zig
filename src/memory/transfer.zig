//! Transfer direction helpers and peer-access-enabled D2D transfers.
//!
//! This module complements the per-call functions in `runtime/memory.zig` with
//! direction-typed wrappers and multi-GPU peer transfer support. Callers that
//! want the typed `DeviceBuffer` copy helpers should use `memory/device_memory.zig`
//! instead; this module is the lower layer.

const std = @import("std");
const loader = @import("../core/loader.zig");
const mem_rt = @import("../runtime/memory.zig");
const ffi_rt = @import("../runtime/ffi.zig");
const stream_mod = @import("../stream/stream.zig");
const err = @import("../core/error.zig");

/// Copy direction tag.
pub const Direction = enum { host_to_device, device_to_host, device_to_device };

/// Synchronously copies `count` bytes in the given `direction`.
///
/// `dst` and `src` must be appropriate for the direction:
///   `.host_to_device` — `src` is host memory, `dst` is device memory.
///   `.device_to_host` — `src` is device memory, `dst` is host memory.
///   `.device_to_device` — both are device memory.
pub fn copy(dst: *anyopaque, src: *const anyopaque, count: usize, direction: Direction) err.CudaError!void {
    return switch (direction) {
        .host_to_device => mem_rt.memcpyHostToDevice(dst, src, count),
        .device_to_host => mem_rt.memcpyDeviceToHost(dst, src, count),
        .device_to_device => mem_rt.memcpyDeviceToDevice(dst, src, count),
    };
}

/// Asynchronously copies `count` bytes in the given `direction` on `stream`.
pub fn copyAsync(
    dst: *anyopaque,
    src: *const anyopaque,
    count: usize,
    direction: Direction,
    stream: stream_mod.Stream,
) err.CudaError!void {
    return switch (direction) {
        .host_to_device => mem_rt.memcpyHostToDeviceAsync(dst, src, count, stream.handle),
        .device_to_host => mem_rt.memcpyDeviceToHostAsync(dst, src, count, stream.handle),
        .device_to_device => mem_rt.memcpyDeviceToDeviceAsync(dst, src, count, stream.handle),
    };
}

/// Queries whether device `src_device` can directly access memory on `dst_device`.
///
/// Returns `true` if peer access is supported and `false` if the devices cannot
/// communicate directly (requiring a host-mediated transfer).
pub fn canAccessPeer(src_device: u32, dst_device: u32) err.CudaError!bool {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi_rt.DeviceCanAccessPeerFn, "cudaDeviceCanAccessPeer") orelse
        return error.NotInitialized;
    var can: c_int = 0;
    try @import("../core/result.zig").checkRuntime(f(&can, @intCast(src_device), @intCast(dst_device)));
    return can != 0;
}

/// Enables peer access from the calling thread's current device to `peer_device`.
///
/// Once enabled, device-to-device transfers between the two devices can proceed
/// via `copy(.device_to_device, ...)` or `copyAsync(.device_to_device, ...)`.
/// `flags` must be `0` (reserved).
pub fn enablePeerAccess(peer_device: u32, flags: c_uint) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi_rt.DeviceEnablePeerAccessFn, "cudaDeviceEnablePeerAccess") orelse
        return error.NotInitialized;
    try @import("../core/result.zig").checkRuntime(f(@intCast(peer_device), flags));
}

/// Disables peer access from the calling thread's current device to `peer_device`.
pub fn disablePeerAccess(peer_device: u32) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi_rt.DeviceDisablePeerAccessFn, "cudaDeviceDisablePeerAccess") orelse
        return error.NotInitialized;
    try @import("../core/result.zig").checkRuntime(f(@intCast(peer_device)));
}

test "canAccessPeer skips without CUDA" {
    if (!loader.isAvailable()) return error.SkipZigTest;
}

test "copy direction enum values" {
    // Ensure the enum is defined correctly.
    try std.testing.expect(Direction.host_to_device != Direction.device_to_host);
    try std.testing.expect(Direction.device_to_host != Direction.device_to_device);
}

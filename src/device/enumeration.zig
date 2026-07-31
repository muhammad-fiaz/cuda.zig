//! Device enumeration: iterate all visible CUDA devices.
//!
//! Provides `allDevices`, which returns a caller-owned slice of `Device`
//! handles for every device visible to the current process.

const std = @import("std");
const loader = @import("../core/loader.zig");
const runtime_dev = @import("../runtime/device.zig");
const device_mod = @import("device.zig");
const err = @import("../core/error.zig");

/// Returns a slice of `Device` handles for all CUDA-capable devices visible
/// to the current process, allocated with `allocator`.
///
/// The caller owns the returned slice and must free it via `allocator.free`.
/// If no devices are present the returned slice is empty (length 0) but still
/// valid and must be freed. Returns `error.NotInitialized` if the runtime
/// library is unavailable.
pub fn allDevices(allocator: std.mem.Allocator) err.CudaError![]device_mod.Device {
    if (!loader.isAvailable()) return error.NotInitialized;
    const count = try runtime_dev.getDeviceCount();
    const devices = try allocator.alloc(device_mod.Device, count);
    for (0..count) |i| {
        devices[i] = device_mod.Device{ .index = @intCast(i) };
    }
    return devices;
}

test "allDevices without CUDA" {
    if (!loader.isAvailable()) {
        const r = allDevices(std.testing.allocator);
        try std.testing.expectError(error.NotInitialized, r);
        return;
    }
    const devs = try allDevices(std.testing.allocator);
    defer std.testing.allocator.free(devs);
    // Count may be 0 on a headless machine; just ensure it doesn't crash.
    for (devs) |d| {
        try std.testing.expect(d.index < devs.len);
    }
}

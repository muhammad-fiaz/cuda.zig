//! Runtime dispatch: chooses between the GPU path and the CPU fallback.
//!
//! This module is the gatekeeper consulted by `memory/device_memory.zig`,
//! `tensor/tensor.zig`, and the ops layer before any CUDA call is made.
//! It provides:
//!   `isGpuAvailable()` — returns true if CUDA libraries and a device are present.
//!   `cpuAlloc(bytes)` — allocates on the process heap (fallback path).
//!   `cpuFree(ptr, bytes)` — releases a heap allocation.
//!
//! The GPU availability check is cached after the first call.

const std = @import("std");
const loader = @import("../core/loader.zig");

// We use the page allocator for fallback allocations so that there is no
// dependency on a caller-provided allocator, keeping the DeviceBuffer API
// stateless.
const fallback_allocator = std.heap.page_allocator;

/// Returns `true` if the CUDA driver and runtime libraries were loaded AND
/// at least one CUDA-capable device is present.
///
/// This is the primary gate consulted before any GPU API call. The result is
/// cached after the first successful check to avoid repeated runtime queries.
pub fn isGpuAvailable() bool {
    // Static check: are the libraries present?
    if (!loader.isAvailable()) return false;
    // Dynamic check: is there at least one device?
    return deviceCheckCached();
}

var device_check_inited = false;
var device_available: bool = false;

fn deviceCheckCached() bool {
    if (!@atomicLoad(bool, &device_check_inited, .acquire)) {
        const rt = @import("../runtime/device.zig");
        const count = rt.getDeviceCount() catch 0;
        device_available = count > 0;
        @atomicStore(bool, &device_check_inited, true, .release);
    }
    return device_available;
}

/// Allocates `size` bytes from the process heap for the CPU fallback path.
///
/// Uses `std.heap.page_allocator` so there is no dependency on a caller-
/// provided allocator. The returned pointer must be freed via `cpuFree`.
pub fn cpuAlloc(size: usize) !*anyopaque {
    const buf = try fallback_allocator.alloc(u8, size);
    return @ptrCast(buf.ptr);
}

/// Releases a fallback allocation returned by `cpuAlloc`.
pub fn cpuFree(ptr: *anyopaque, size: usize) void {
    const p: [*]u8 = @ptrCast(ptr);
    fallback_allocator.free(p[0..size]);
}

test "isGpuAvailable returns a bool" {
    _ = isGpuAvailable();
}

test "cpuAlloc and cpuFree round-trip" {
    const ptr = try cpuAlloc(256);
    const bytes: [*]u8 = @ptrCast(ptr);
    bytes[0] = 0xFF;
    try std.testing.expect(bytes[0] == 0xFF);
    cpuFree(ptr, 256);
}

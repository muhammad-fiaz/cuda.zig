//! The Device struct: a handle to a specific CUDA device by index.
//!
//! `Device` is a lightweight value type — it holds only the device index.
//! All operations on a `Device` query the CUDA runtime lazily; properties are
//! not cached. If caching is needed, callers should store the result of
//! `Device.properties()` themselves.

const std = @import("std");
const loader = @import("../core/loader.zig");
const runtime_dev = @import("../runtime/device.zig");
const ffi = @import("../runtime/ffi.zig");
const props_mod = @import("properties.zig");
const err = @import("../core/error.zig");

pub const DeviceProperties = props_mod.DeviceProperties;

/// Thread-local storage for the device name string. This lets `name()` return
/// a slice that remains valid across the function boundary.
var tls_name_buf: [256]u8 = [_]u8{0} ** 256;

/// A handle to a specific CUDA device identified by its ordinal index.
///
/// Obtaining a `Device` does not affect the "current device" for any thread;
/// use `selection.setDevice` for that. All property queries on a `Device`
/// communicate with the runtime regardless of which device is currently active.
pub const Device = struct {
    /// Zero-based ordinal index of the device.
    index: u32,

    /// Constructs a `Device` handle for the device at `index`.
    ///
    /// Returns `error.InvalidDevice` if `index` is >= `deviceCount()`, or
    /// `error.NotInitialized` if the runtime library is unavailable.
    pub fn init(index: u32) err.CudaError!Device {
        if (!loader.isAvailable()) return error.NotInitialized;
        const count = try runtime_dev.getDeviceCount();
        if (index >= count) return error.InvalidDevice;
        return .{ .index = index };
    }

    /// Returns the full set of device properties.
    ///
    /// This call communicates with the CUDA runtime and involves a kernel
    /// round-trip on the first invocation for a given device. Consider caching
    /// the result if it will be queried frequently.
    pub fn propertiesRaw(self: Device) err.CudaError!DeviceProperties {
        var raw: ffi.DeviceProp = std.mem.zeroes(ffi.DeviceProp);
        try runtime_dev.getDeviceProperties(&raw, self.index);
        return rawToProperties(&raw);
    }

    /// Returns the device name as a slice backed by a global buffer.
    ///
    /// The returned slice is valid until the next call to `name()` on any
    /// `Device`. Copy it if you need a longer lifetime.
    pub fn name(self: Device) err.CudaError![]const u8 {
        var raw: ffi.DeviceProp = std.mem.zeroes(ffi.DeviceProp);
        try runtime_dev.getDeviceProperties(&raw, self.index);
        // Copy into module-level buffer so the slice survives the return.
        const len = std.mem.indexOfScalar(u8, &raw.name, 0) orelse raw.name.len;
        @memset(&tls_name_buf, 0);
        @memcpy(tls_name_buf[0..len], raw.name[0..len]);
        return tls_name_buf[0..len];
    }

    /// Returns the total global memory of the device in bytes.
    pub fn totalMemory(self: Device) err.CudaError!usize {
        var raw: ffi.DeviceProp = std.mem.zeroes(ffi.DeviceProp);
        try runtime_dev.getDeviceProperties(&raw, self.index);
        return raw.total_global_mem;
    }

    /// Returns the free and total global memory of the current device in bytes.
    ///
    /// Note: this queries the device associated with the calling thread, not
    /// necessarily `self.index`. Call `selection.setDevice(self.index)` first
    /// if you need accurate free memory for a specific device.
    pub fn freeMemory(_: Device) err.CudaError!usize {
        const info = try runtime_dev.getMemInfo();
        return info.free;
    }

    /// Returns the compute capability as `{ .major, .minor }`.
    pub fn computeCapability(self: Device) err.CudaError!struct { major: u32, minor: u32 } {
        var raw: ffi.DeviceProp = std.mem.zeroes(ffi.DeviceProp);
        try runtime_dev.getDeviceProperties(&raw, self.index);
        return .{
            .major = @intCast(raw.major),
            .minor = @intCast(raw.minor),
        };
    }
};

/// Converts a raw `ffi.DeviceProp` ABI struct into a `DeviceProperties`.
fn rawToProperties(raw: *const ffi.DeviceProp) DeviceProperties {
    var p: DeviceProperties = undefined;
    @memset(&p.name, 0);
    const len = std.mem.indexOfScalar(u8, &raw.name, 0) orelse raw.name.len;
    @memcpy(p.name[0..len], raw.name[0..len]);
    p.name[len] = 0;
    p.total_global_mem = raw.total_global_mem;
    p.shared_mem_per_block = raw.shared_mem_per_block;
    p.regs_per_block = @intCast(raw.regs_per_block);
    p.warp_size = @intCast(raw.warp_size);
    p.max_threads_per_block = @intCast(raw.max_threads_per_block);
    for (0..3) |i| {
        p.max_threads_dim[i] = @intCast(raw.max_threads_dim[i]);
        p.max_grid_size[i] = @intCast(raw.max_grid_size[i]);
    }
    p.clock_rate_khz = @intCast(raw.clock_rate);
    p.total_const_mem = raw.total_const_mem;
    p.major = @intCast(raw.major);
    p.minor = @intCast(raw.minor);
    p.multi_processor_count = @intCast(raw.multi_processor_count);
    p.kernel_exec_timeout_enabled = raw.kernel_exec_timeout_enabled != 0;
    p.integrated = raw.integrated != 0;
    p.can_map_host_memory = raw.can_map_host_memory != 0;
    p.async_engine_count = @intCast(raw.async_engine_count);
    p.unified_addressing = raw.unified_addressing != 0;
    p.memory_clock_rate_khz = @intCast(raw.memory_clock_rate);
    p.memory_bus_width = @intCast(raw.memory_bus_width);
    p.l2_cache_size = @intCast(raw.l2_cache_size);
    p.max_threads_per_multi_processor = @intCast(raw.max_threads_per_multi_processor);
    p.cooperative_launch = raw.cooperative_launch != 0;
    p.managed_memory = raw.managed_memory != 0;
    p.concurrent_managed_access = raw.concurrent_managed_access != 0;
    p.memory_pools_supported = raw.memory_pools_supported != 0;
    p.pci_bus_id = @intCast(raw.pci_bus_id);
    p.pci_device_id = @intCast(raw.pci_device_id);
    p.pci_domain_id = @intCast(raw.pci_domain_id);
    return p;
}

test "Device.init requires CUDA" {
    if (!loader.isAvailable()) {
        const r = Device.init(0);
        try std.testing.expectError(error.NotInitialized, r);
        return;
    }
    // With CUDA: device 0 must succeed if any device is present.
    const count = try runtime_dev.getDeviceCount();
    if (count == 0) return error.SkipZigTest;
    const d = try Device.init(0);
    try std.testing.expectEqual(@as(u32, 0), d.index);
}

test "Device.init invalid index" {
    if (!loader.isAvailable()) return error.SkipZigTest;
    const count = try runtime_dev.getDeviceCount();
    const r = Device.init(count + 9999);
    try std.testing.expectError(error.InvalidDevice, r);
}

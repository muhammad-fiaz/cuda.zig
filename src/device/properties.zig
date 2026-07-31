//! DeviceProperties: structured representation of cudaDeviceProp.
//!
//! This struct is a Zig-idiomatic view of the fields returned by
//! `cudaGetDeviceProperties`. It is separate from the raw `ffi.DeviceProp`
//! ABI struct so that the public API surface is not tied to the CUDA ABI
//! layout, which has changed between toolkit versions.

const std = @import("std");

/// Properties of a CUDA-capable device, as returned by
/// `runtime/device.zig::getDeviceProperties`.
///
/// All memory sizes are in bytes. Frequencies are in kHz.
pub const DeviceProperties = struct {
    /// ASCII name of the device (e.g. "NVIDIA GeForce RTX 4090").
    /// Null-terminated; the slice does not include the terminator.
    name: [256:0]u8,

    /// Total global memory available on the device, in bytes.
    total_global_mem: usize,

    /// Maximum amount of shared memory available per block, in bytes.
    shared_mem_per_block: usize,

    /// Maximum number of 32-bit registers available per block.
    regs_per_block: u32,

    /// Warp size (number of threads per warp); always 32 on NVIDIA hardware.
    warp_size: u32,

    /// Maximum number of threads per block.
    max_threads_per_block: u32,

    /// Maximum number of threads in each dimension of a block.
    max_threads_dim: [3]u32,

    /// Maximum number of blocks in each dimension of a grid.
    max_grid_size: [3]u32,

    /// Clock frequency of the device, in kHz.
    clock_rate_khz: u32,

    /// Total amount of constant memory available, in bytes.
    total_const_mem: usize,

    /// Major revision of the device compute capability.
    major: u32,

    /// Minor revision of the device compute capability.
    minor: u32,

    /// Number of multiprocessors on the device.
    multi_processor_count: u32,

    /// Kernel execution timeout: `true` if there is a watchdog timer
    /// (device is attached to a display).
    kernel_exec_timeout_enabled: bool,

    /// Whether the device is integrated (e.g. Jetson) rather than discrete.
    integrated: bool,

    /// Whether the device supports mapping host memory into the device's
    /// virtual address space.
    can_map_host_memory: bool,

    /// Number of asynchronous engines (copy engines).
    async_engine_count: u32,

    /// Whether unified addressing is supported.
    unified_addressing: bool,

    /// Clock rate of the memory bus, in kHz.
    memory_clock_rate_khz: u32,

    /// Width of the memory bus, in bits.
    memory_bus_width: u32,

    /// L2 cache size, in bytes.
    l2_cache_size: u32,

    /// Maximum number of threads per multiprocessor.
    max_threads_per_multi_processor: u32,

    /// Whether cooperative kernel launches are supported.
    cooperative_launch: bool,

    /// Whether managed memory is supported.
    managed_memory: bool,

    /// Whether concurrent managed memory access is supported.
    concurrent_managed_access: bool,

    /// Whether memory pools (`cudaMallocAsync`) are supported.
    memory_pools_supported: bool,

    /// PCI bus identifier.
    pci_bus_id: u32,
    /// PCI device identifier.
    pci_device_id: u32,
    /// PCI domain identifier.
    pci_domain_id: u32,

    /// Returns a null-terminated slice view of the device name.
    pub fn nameSlice(self: *const DeviceProperties) [:0]const u8 {
        return std.mem.sliceTo(&self.name, 0);
    }
};

test "DeviceProperties.nameSlice" {
    var p: DeviceProperties = std.mem.zeroes(DeviceProperties);
    @memcpy(p.name[0..10], "TestDevice");
    p.name[10] = 0;
    const s = p.nameSlice();
    try std.testing.expectEqualStrings("TestDevice", s);
}

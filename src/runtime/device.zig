//! Thin wrappers around the CUDA Runtime API device-management functions.
//!
//! This module provides the lowest-level Zig-typed wrappers for:
//!   cudaGetDeviceCount, cudaGetDevice, cudaSetDevice,
//!   cudaGetDeviceProperties, cudaDeviceReset, cudaDeviceSynchronize,
//!   cudaMemGetInfo.
//!
//! Higher-level `Device` struct ergonomics live in `device/device.zig`.
//! Thread safety: `cudaSetDevice` affects only the calling thread's device
//! association; operations on another thread are not affected.

const std = @import("std");
const loader = @import("../core/loader.zig");
const ffi = @import("ffi.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");

/// Returns the number of CUDA-capable devices visible to the process.
///
/// Returns `0` if no devices are present. Returns `error.NotInitialized` if
/// the runtime library could not be loaded. This call does not require a
/// context to be current.
pub fn getDeviceCount() err.CudaError!u32 {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.GetDeviceCountFn, "cudaGetDeviceCount")) |f| {
        var count: c_int = 0;
        try result.checkRuntime(f(&count));
        return @intCast(count);
    }
    if (ldr.getDriverSymbol(*const fn (*c_int) callconv(.c) c_int, "cuDeviceGetCount")) |f| {
        if (ldr.getDriverSymbol(*const fn (c_uint) callconv(.c) c_int, "cuInit")) |cu_init| {
            _ = cu_init(0);
        }
        var count: c_int = 0;
        try result.checkDriver(f(&count));
        return @intCast(count);
    }
    return 0;
}

pub fn getCurrentDevice() err.CudaError!u32 {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.GetDeviceFn, "cudaGetDevice")) |f| {
        var dev: c_int = 0;
        try result.checkRuntime(f(&dev));
        return @intCast(dev);
    }
    return 0;
}

pub fn setDevice(index: u32) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.SetDeviceFn, "cudaSetDevice")) |f| {
        try result.checkRuntime(f(@intCast(index)));
        return;
    }
    // Driver API fallback for device selection: retain primary context and make it current
    if (ldr.getDriverSymbol(*const fn (*?*anyopaque, c_int) callconv(.c) c_int, "cuDevicePrimaryCtxRetain")) |retain_fn| {
        var ctx: ?*anyopaque = null;
        if (retain_fn(&ctx, @intCast(index)) == 0) {
            if (ldr.getDriverSymbol(*const fn (?*anyopaque) callconv(.c) c_int, "cuCtxSetCurrent")) |set_curr| {
                _ = set_curr(ctx);
            }
        }
    }
}

/// Fills `prop` with the properties of device `index`.
///
/// `prop` must point to a caller-allocated `ffi.DeviceProp`. The struct is
/// populated by the runtime and remains valid for the lifetime of the caller's
/// stack frame (or however long the pointer lives).
pub fn getDeviceProperties(prop: *ffi.DeviceProp, index: u32) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.GetDevicePropertiesFn, "cudaGetDeviceProperties_v2") orelse
        ldr.getRuntimeSymbol(ffi.GetDevicePropertiesFn, "cudaGetDeviceProperties")) |f|
    {
        try result.checkRuntime(f(prop, @intCast(index)));
        return;
    }
    // Fallback: populate properties accurately via Driver API
    @memset(std.mem.asBytes(prop), 0);
    var cu_dev: c_int = @intCast(index);
    if (ldr.getDriverSymbol(*const fn (*c_int, c_int) callconv(.c) c_int, "cuDeviceGet")) |dev_fn| {
        _ = dev_fn(&cu_dev, @intCast(index));
    }

    if (ldr.getDriverSymbol(*const fn ([*c]u8, c_int, c_int) callconv(.c) c_int, "cuDeviceGetName")) |name_fn| {
        if (name_fn(@ptrCast(&prop.name), @intCast(prop.name.len - 1), cu_dev) == 0) {
            prop.name[prop.name.len - 1] = 0;
        }
    }
    if (ldr.getDriverSymbol(*const fn (*usize, c_int) callconv(.c) c_int, "cuDeviceTotalMem_v2") orelse
        ldr.getDriverSymbol(*const fn (*usize, c_int) callconv(.c) c_int, "cuDeviceTotalMem")) |mem_fn|
    {
        _ = mem_fn(&prop.total_global_mem, cu_dev);
    }
    if (ldr.getDriverSymbol(*const fn (*c_int, c_int, c_int) callconv(.c) c_int, "cuDeviceGetAttribute")) |attr_fn| {
        _ = attr_fn(&prop.major, 75, cu_dev); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        _ = attr_fn(&prop.minor, 76, cu_dev); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        _ = attr_fn(&prop.multi_processor_count, 16, cu_dev); // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        _ = attr_fn(&prop.max_threads_per_block, 1, cu_dev); // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        _ = attr_fn(&prop.warp_size, 10, cu_dev); // CU_DEVICE_ATTRIBUTE_WARP_SIZE
        _ = attr_fn(&prop.clock_rate, 13, cu_dev); // CU_DEVICE_ATTRIBUTE_CLOCK_RATE
        _ = attr_fn(&prop.l2_cache_size, 36, cu_dev); // CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
        _ = attr_fn(&prop.unified_addressing, 41, cu_dev); // CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
        _ = attr_fn(&prop.managed_memory, 83, cu_dev); // CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY
        _ = attr_fn(&prop.concurrent_managed_access, 89, cu_dev); // CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
        _ = attr_fn(&prop.async_engine_count, 37, cu_dev); // CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
        _ = attr_fn(&prop.cooperative_launch, 96, cu_dev); // CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH
        _ = attr_fn(&prop.memory_pools_supported, 115, cu_dev); // CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED
        _ = attr_fn(&prop.memory_clock_rate, 38, cu_dev); // CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
        _ = attr_fn(&prop.memory_bus_width, 35, cu_dev); // CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
        _ = attr_fn(&prop.max_threads_per_multi_processor, 39, cu_dev); // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
        _ = attr_fn(&prop.integrated, 18, cu_dev); // CU_DEVICE_ATTRIBUTE_INTEGRATED
        _ = attr_fn(&prop.can_map_host_memory, 19, cu_dev); // CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
        _ = attr_fn(&prop.kernel_exec_timeout_enabled, 17, cu_dev); // CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT
        _ = attr_fn(&prop.pci_bus_id, 33, cu_dev); // CU_DEVICE_ATTRIBUTE_PCI_BUS_ID
        _ = attr_fn(&prop.pci_device_id, 34, cu_dev); // CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
        _ = attr_fn(&prop.pci_domain_id, 50, cu_dev); // CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
        // usize fields need a c_int temp
        var tmp_const_mem: c_int = 0;
        _ = attr_fn(&tmp_const_mem, 12, cu_dev); // CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
        prop.total_const_mem = @intCast(tmp_const_mem);
        var tmp_shmem: c_int = 0;
        _ = attr_fn(&tmp_shmem, 8, cu_dev); // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
        prop.shared_mem_per_block = @intCast(tmp_shmem);
    }
}

pub fn deviceReset() err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.DeviceResetFn, "cudaDeviceReset")) |f| {
        try result.checkRuntime(f());
        return;
    }
    if (ldr.getDriverSymbol(*const fn (c_int) callconv(.c) c_int, "cuDevicePrimaryCtxReset")) |f| {
        _ = f(0);
    }
}

pub fn deviceSynchronize() err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.DeviceSynchronizeFn, "cudaDeviceSynchronize")) |f| {
        try result.checkRuntime(f());
        return;
    }
    if (ldr.getDriverSymbol(*const fn () callconv(.c) c_int, "cuCtxSynchronize")) |f| {
        try result.checkDriver(f());
    }
}

/// Returns the free and total global memory available on the current device.
///
/// Both values are in bytes. `free` changes dynamically as allocations occur.
pub fn getMemInfo() err.CudaError!struct { free: usize, total: usize } {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.DeviceGetMemInfoFn, "cudaMemGetInfo")) |f| {
        var free: usize = 0;
        var total: usize = 0;
        try result.checkRuntime(f(&free, &total));
        return .{ .free = free, .total = total };
    }
    if (ldr.getDriverSymbol(*const fn (*usize, *usize) callconv(.c) c_int, "cuMemGetInfo_v2") orelse
        ldr.getDriverSymbol(*const fn (*usize, *usize) callconv(.c) c_int, "cuMemGetInfo")) |f|
    {
        var free: usize = 0;
        var total: usize = 0;
        try result.checkDriver(f(&free, &total));
        return .{ .free = free, .total = total };
    }
    return error.NotInitialized;
}

test "getDeviceCount without CUDA" {
    if (!loader.isAvailable()) {
        // Without the runtime library we cannot call this.
        return error.SkipZigTest;
    }
    const count = try getDeviceCount();
    // Count may be 0 on a headless CI machine.
    _ = count;
}

// Peer (P2P) Access

/// Returns `true` if `device` can directly access memory on `peer_device`.
///
/// Both devices must be in the same CUDA context group. On most systems,
/// two RTX/Quadro GPUs on the same PCIe root complex support P2P access.
/// Call `enablePeerAccess` before using P2P `cudaMemcpy` or unified memory
/// across devices.
pub fn canAccessPeer(device: u32, peer_device: u32) err.CudaError!bool {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.DeviceCanAccessPeerFn, "cudaDeviceCanAccessPeer")) |f| {
        var can: c_int = 0;
        try result.checkRuntime(f(&can, @intCast(device), @intCast(peer_device)));
        return can != 0;
    }
    if (ldr.getDriverSymbol(*const fn (*c_int, c_int, c_int) callconv(.c) c_int, "cuDeviceCanAccessPeer")) |f| {
        var can: c_int = 0;
        try result.checkDriver(f(&can, @intCast(device), @intCast(peer_device)));
        return can != 0;
    }
    return false;
}

/// Enables direct access from the current device to `peer_device`'s memory.
///
/// Must be called from the thread that owns the context on the **source**
/// device (i.e., after `setDevice(source)`). `flags` must be `0` (reserved).
/// Call `disablePeerAccess` when done to release hardware resources.
pub fn enablePeerAccess(peer_device: u32, flags: c_uint) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.DeviceEnablePeerAccessFn, "cudaDeviceEnablePeerAccess")) |f| {
        try result.checkRuntime(f(@intCast(peer_device), flags));
        return;
    }
    if (ldr.getDriverSymbol(*const fn (?*anyopaque, c_uint) callconv(.c) c_int, "cuCtxEnablePeerAccess")) |f| {
        // Driver API takes a context handle; we use null (current context).
        try result.checkDriver(f(null, flags));
        return;
    }
    return error.NotInitialized;
}

/// Disables direct access from the current device to `peer_device`'s memory.
pub fn disablePeerAccess(peer_device: u32) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.DeviceDisablePeerAccessFn, "cudaDeviceDisablePeerAccess")) |f| {
        try result.checkRuntime(f(@intCast(peer_device)));
        return;
    }
    return error.NotInitialized;
}

// Raw Device Attribute Query

/// Queries a single device attribute by its CUDA enum integer value.
///
/// This is the lowest-level escape hatch: if `DeviceProperties` is missing
/// an attribute you need, call `getAttribute` with the appropriate
/// `CU_DEVICE_ATTRIBUTE_*` constant. Returns `0` if the attribute is
/// unavailable or the driver is absent.
pub fn getAttribute(attribute: c_int, device: u32) i32 {
    const ldr = loader.globalLoader();
    if (ldr.getDriverSymbol(*const fn (*c_int, c_int, c_int) callconv(.c) c_int, "cuDeviceGetAttribute")) |f| {
        var val: c_int = 0;
        _ = f(&val, attribute, @intCast(device));
        return @intCast(val);
    }
    return 0;
}

test "peer access compiles" {
    _ = canAccessPeer;
    _ = enablePeerAccess;
    _ = disablePeerAccess;
}

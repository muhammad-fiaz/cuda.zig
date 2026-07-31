//! Raw extern function declarations for the CUDA Driver API.
//!
//! This file contains only `extern fn` declarations and opaque handle type
//! definitions. No logic lives here; all logic is in the sibling modules
//! `context.zig`, `module.zig`, and `function.zig`.
//!
//! Symbols are NOT linked at compile time. The actual function pointers are
//! resolved dynamically from `nvcuda.dll`/`libcuda.so` by `core/loader.zig`.
//! The declarations here are used only for their type signatures, which are
//! passed as the `T` parameter to `Loader.getDriverSymbol`.
//!
//! Naming convention: Zig type aliases strip the `CU` prefix and use
//! PascalCase. Raw function pointer types are suffixed `Fn`.

const std = @import("std");

/// Opaque handle to a CUDA driver context (CUcontext).
pub const Context = *anyopaque;
/// Opaque handle to a CUDA driver module (CUmodule).
pub const Module = *anyopaque;
/// Opaque handle to a CUDA driver function/kernel (CUfunction).
pub const Function = *anyopaque;
/// Opaque handle to a CUDA driver stream (CUstream).
pub const Stream = *anyopaque;
/// Opaque handle to a CUDA driver event (CUevent).
pub const Event = *anyopaque;
/// Opaque handle to a CUDA device (CUdevice is actually c_int).
pub const Device = c_int;
/// Device pointer type as returned by driver-API allocations.
pub const DevicePtr = u64;

/// Context creation flags (cuCtxCreate flags parameter).
pub const CtxFlags = struct {
    pub const sched_auto: c_uint = 0x00;
    pub const sched_spin: c_uint = 0x01;
    pub const sched_yield: c_uint = 0x02;
    pub const sched_blocking_sync: c_uint = 0x04;
    pub const map_host: c_uint = 0x08;
    pub const lmem_resize_to_max: c_uint = 0x10;
};

/// JIT option keys for `cuModuleLoadDataEx`.
pub const JitOption = enum(c_uint) {
    max_registers = 0,
    threads_per_block = 1,
    wall_time = 2,
    info_log_buffer = 3,
    info_log_buffer_size_bytes = 4,
    error_log_buffer = 5,
    error_log_buffer_size_bytes = 6,
    optimization_level = 7,
    target_from_cucontext = 8,
    target = 9,
    fallback_strategy = 10,
    generate_debug_info = 11,
    log_verbose = 12,
    generate_line_info = 13,
    cache_mode = 14,
    new_sm3x_opt = 15,
    fast_compile = 16,
    global_symbol_names = 17,
    global_symbol_addresses = 18,
    global_symbol_count = 19,
    lto = 22,
    ftz = 23,
    prec_div = 24,
    prec_sqrt = 25,
    fma = 26,
    referenced_kernel_names = 27,
    referenced_kernel_count = 28,
    referenced_variable_names = 29,
    referenced_variable_count = 30,
    optimize_unused_device_variables = 31,
    position_independent_code = 32,
};

/// Function pointer types for all Driver API functions used by cuda.zig.
/// Each is passed to `Loader.getDriverSymbol` to obtain a callable pointer.
pub const InitFn = *const fn (flags: c_uint) callconv(.c) c_int;
pub const DriverGetVersionFn = *const fn (driver_version: *c_int) callconv(.c) c_int;

pub const DeviceGetFn = *const fn (device: *Device, ordinal: c_int) callconv(.c) c_int;
pub const DeviceGetCountFn = *const fn (count: *c_int) callconv(.c) c_int;
pub const DeviceGetNameFn = *const fn (name: [*c]u8, len: c_int, dev: Device) callconv(.c) c_int;
pub const DeviceGetAttributeFn = *const fn (pi: *c_int, attrib: c_int, dev: Device) callconv(.c) c_int;
pub const DeviceTotalMemFn = *const fn (bytes: *usize, dev: Device) callconv(.c) c_int;

pub const CtxCreateFn = *const fn (ctx: **anyopaque, flags: c_uint, dev: Device) callconv(.c) c_int;
pub const CtxDestroyFn = *const fn (ctx: *anyopaque) callconv(.c) c_int;
pub const CtxPushCurrentFn = *const fn (ctx: *anyopaque) callconv(.c) c_int;
pub const CtxPopCurrentFn = *const fn (ctx: **anyopaque) callconv(.c) c_int;
pub const CtxGetCurrentFn = *const fn (ctx: **anyopaque) callconv(.c) c_int;
pub const CtxSetCurrentFn = *const fn (ctx: *anyopaque) callconv(.c) c_int;
pub const CtxSynchronizeFn = *const fn () callconv(.c) c_int;

pub const DevicePrimaryCtxRetainFn = *const fn (ctx: **anyopaque, dev: Device) callconv(.c) c_int;
pub const DevicePrimaryCtxReleaseFn = *const fn (dev: Device) callconv(.c) c_int;
pub const DevicePrimaryCtxResetFn = *const fn (dev: Device) callconv(.c) c_int;

pub const ModuleLoadFn = *const fn (module: **anyopaque, fname: [*:0]const u8) callconv(.c) c_int;
pub const ModuleLoadDataFn = *const fn (module: **anyopaque, image: *const anyopaque) callconv(.c) c_int;
pub const ModuleLoadDataExFn = *const fn (
    module: **anyopaque,
    image: *const anyopaque,
    num_options: c_uint,
    options: [*]const c_uint,
    option_values: [*]*anyopaque,
) callconv(.c) c_int;
pub const ModuleLoadFatBinaryFn = *const fn (module: **anyopaque, fatCubin: *const anyopaque) callconv(.c) c_int;
pub const ModuleUnloadFn = *const fn (module: *anyopaque) callconv(.c) c_int;
pub const ModuleGetFunctionFn = *const fn (func: **anyopaque, module: *anyopaque, name: [*:0]const u8) callconv(.c) c_int;
pub const ModuleGetGlobalFn = *const fn (dptr: *DevicePtr, bytes: *usize, module: *anyopaque, name: [*:0]const u8) callconv(.c) c_int;

pub const LaunchKernelFn = *const fn (
    f: *anyopaque,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    stream: ?*anyopaque,
    kernel_params: [*]?*anyopaque,
    extra: ?[*]?*anyopaque,
) callconv(.c) c_int;

pub const MemAllocFn = *const fn (dptr: *DevicePtr, byte_size: usize) callconv(.c) c_int;
pub const MemFreeFn = *const fn (dptr: DevicePtr) callconv(.c) c_int;
pub const MemcpyHtoDFn = *const fn (dst: DevicePtr, src_host: *const anyopaque, byte_count: usize) callconv(.c) c_int;
pub const MemcpyDtoHFn = *const fn (dst_host: *anyopaque, src: DevicePtr, byte_count: usize) callconv(.c) c_int;
pub const MemcpyDtoDFn = *const fn (dst: DevicePtr, src: DevicePtr, byte_count: usize) callconv(.c) c_int;
pub const MemcpyHtoDAsyncFn = *const fn (dst: DevicePtr, src_host: *const anyopaque, byte_count: usize, stream: ?*anyopaque) callconv(.c) c_int;
pub const MemcpyDtoHAsyncFn = *const fn (dst_host: *anyopaque, src: DevicePtr, byte_count: usize, stream: ?*anyopaque) callconv(.c) c_int;
pub const MemsetD8Fn = *const fn (dst: DevicePtr, value: u8, count: usize) callconv(.c) c_int;
pub const MemsetD32Fn = *const fn (dst: DevicePtr, value: c_uint, count: usize) callconv(.c) c_int;

pub const StreamCreateFn = *const fn (stream: **anyopaque) callconv(.c) c_int;
pub const StreamCreateWithFlagsFn = *const fn (stream: **anyopaque, flags: c_uint) callconv(.c) c_int;
pub const StreamDestroyFn = *const fn (stream: *anyopaque) callconv(.c) c_int;
pub const StreamSynchronizeFn = *const fn (stream: *anyopaque) callconv(.c) c_int;
pub const StreamQueryFn = *const fn (stream: *anyopaque) callconv(.c) c_int;
pub const StreamWaitEventFn = *const fn (stream: *anyopaque, event: *anyopaque, flags: c_uint) callconv(.c) c_int;

pub const EventCreateFn = *const fn (event: **anyopaque) callconv(.c) c_int;
pub const EventCreateWithFlagsFn = *const fn (event: **anyopaque, flags: c_uint) callconv(.c) c_int;
pub const EventDestroyFn = *const fn (event: *anyopaque) callconv(.c) c_int;
pub const EventRecordFn = *const fn (event: *anyopaque, stream: ?*anyopaque) callconv(.c) c_int;
pub const EventSynchronizeFn = *const fn (event: *anyopaque) callconv(.c) c_int;
pub const EventElapsedTimeFn = *const fn (ms: *f32, start: *anyopaque, end: *anyopaque) callconv(.c) c_int;
pub const EventQueryFn = *const fn (event: *anyopaque) callconv(.c) c_int;

pub const DeviceCanAccessPeerFn = *const fn (can_access: *c_int, dev: Device, peer_dev: Device) callconv(.c) c_int;
pub const CtxEnablePeerAccessFn = *const fn (peer_ctx: *anyopaque, flags: c_uint) callconv(.c) c_int;
pub const CtxDisablePeerAccessFn = *const fn (peer_ctx: *anyopaque) callconv(.c) c_int;

test "driver ffi type sizes are plausible" {
    // Opaque pointers must be pointer-sized.
    try std.testing.expect(@sizeOf(*anyopaque) == @sizeOf(usize));
    // DevicePtr is a 64-bit integer on all supported platforms.
    try std.testing.expect(@sizeOf(DevicePtr) == 8);
    // Device is a c_int.
    try std.testing.expect(@sizeOf(Device) == @sizeOf(c_int));
}

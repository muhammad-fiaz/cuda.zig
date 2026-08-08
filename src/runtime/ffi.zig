//! Raw function pointer type declarations for the CUDA Runtime API.
//!
//! Parallel to `driver/ffi.zig` but covering `cudaXxx` Runtime API symbols.
//! All symbols are resolved dynamically via `core/loader.zig`; nothing here
//! generates a link-time dependency on the runtime library.

const std = @import("std");

/// Opaque handle to a CUDA runtime stream (cudaStream_t).
pub const Stream = ?*anyopaque;
/// Opaque handle to a CUDA runtime event (cudaEvent_t).
pub const Event = ?*anyopaque;

/// cudaMemcpyKind: direction constants for cudaMemcpy.
pub const MemcpyKind = enum(c_int) {
    host_to_host = 0,
    host_to_device = 1,
    device_to_host = 2,
    device_to_device = 3,
    default = 4,
};

/// cudaMemAttach flags for cudaMallocManaged.
pub const MemAttachFlags = struct {
    pub const global: c_uint = 0x01;
    pub const host: c_uint = 0x02;
    pub const single: c_uint = 0x04;
};

/// cudaMemAdvise values for cudaMemAdvise.
pub const MemAdvise = enum(c_int) {
    set_read_mostly = 1,
    unset_read_mostly = 2,
    set_preferred_location = 3,
    unset_preferred_location = 4,
    set_accessed_by = 5,
    unset_accessed_by = 6,
};

/// cudaHostAllocFlags for cudaHostAlloc.
pub const HostAllocFlags = struct {
    pub const default: c_uint = 0x00;
    pub const portable: c_uint = 0x01;
    pub const mapped: c_uint = 0x02;
    pub const write_combined: c_uint = 0x04;
};

/// cudaEventFlags for cudaEventCreateWithFlags.
pub const EventFlags = struct {
    pub const default: c_uint = 0x00;
    pub const blocking_sync: c_uint = 0x01;
    pub const disable_timing: c_uint = 0x02;
    pub const interprocess: c_uint = 0x04;
};

/// cudaStreamFlags for cudaStreamCreateWithFlags.
pub const StreamFlags = struct {
    pub const default: c_uint = 0x00;
    pub const non_blocking: c_uint = 0x01;
};

// Simplified cudaDeviceProp layout covering fields we surface.
// The full struct in the header is ~4KB; we pull only the fields we use.
// Fields are positioned at their actual byte offsets to match the ABI.
// Padded with anonymous arrays to reach the correct total size (1072 bytes
// for CUDA 12). If the struct size ever becomes version-sensitive, adjust
// in core/version.zig.
pub const DeviceProp = extern struct {
    name: [256]u8,
    uuid: [16]u8,
    luid: [8]u8,
    luid_device_node_mask: c_uint,
    total_global_mem: usize,
    shared_mem_per_block: usize,
    regs_per_block: c_int,
    warp_size: c_int,
    mem_pitch: usize,
    max_threads_per_block: c_int,
    max_threads_dim: [3]c_int,
    max_grid_size: [3]c_int,
    clock_rate: c_int,
    total_const_mem: usize,
    major: c_int,
    minor: c_int,
    texture_alignment: usize,
    texture_pitch_alignment: usize,
    device_overlap: c_int,
    multi_processor_count: c_int,
    kernel_exec_timeout_enabled: c_int,
    integrated: c_int,
    can_map_host_memory: c_int,
    compute_mode: c_int,
    max_texture_1d: c_int,
    max_texture_1d_mipmap: c_int,
    max_texture_1d_linear: c_int,
    max_texture_2d: [2]c_int,
    max_texture_2d_mipmap: [2]c_int,
    max_texture_2d_linear: [3]c_int,
    max_texture_2d_gather: [2]c_int,
    max_texture_3d: [3]c_int,
    max_texture_3d_alt: [3]c_int,
    max_texture_cubemap: c_int,
    max_texture_1d_layered: [2]c_int,
    max_texture_2d_layered: [3]c_int,
    max_texture_cubemap_layered: [2]c_int,
    max_surface_1d: c_int,
    max_surface_2d: [2]c_int,
    max_surface_3d: [3]c_int,
    max_surface_1d_layered: [2]c_int,
    max_surface_2d_layered: [3]c_int,
    max_surface_cubemap: c_int,
    max_surface_cubemap_layered: [2]c_int,
    surface_alignment: usize,
    concurrent_kernels: c_int,
    ecc_enabled: c_int,
    pci_bus_id: c_int,
    pci_device_id: c_int,
    pci_domain_id: c_int,
    tcc_driver: c_int,
    async_engine_count: c_int,
    unified_addressing: c_int,
    memory_clock_rate: c_int,
    memory_bus_width: c_int,
    l2_cache_size: c_int,
    persisting_l2_cache_max_size: c_int,
    max_threads_per_multi_processor: c_int,
    stream_priorities_supported: c_int,
    global_l1_cache_supported: c_int,
    local_l1_cache_supported: c_int,
    shared_mem_per_multiprocessor: usize,
    regs_per_multiprocessor: c_int,
    managed_memory: c_int,
    is_multi_gpu_board: c_int,
    multi_gpu_board_group_id: c_int,
    host_native_atomic_supported: c_int,
    single_to_double_precision_perf_ratio: c_int,
    pageable_memory_access: c_int,
    concurrent_managed_access: c_int,
    compute_preemption_supported: c_int,
    can_use_host_pointer_for_registered_mem: c_int,
    cooperative_launch: c_int,
    cooperative_multi_device_launch: c_int,
    shared_mem_per_block_optin: usize,
    pageable_memory_access_uses_host_page_tables: c_int,
    direct_managed_mem_access_from_host: c_int,
    max_blocks_per_multi_processor: c_int,
    access_policy_max_window_size: c_int,
    reserved_shared_mem_per_block: usize,
    host_register_supported: c_int,
    sparse_cuda_array_supported: c_int,
    host_register_read_only_supported: c_int,
    timeline_semaphore_interop_supported: c_int,
    memory_pools_supported: c_int,
    gpu_direct_rdma_supported: c_int,
    gpu_direct_rdma_flush_writes_options: c_uint,
    gpu_direct_rdma_writes_ordering: c_int,
    memory_pool_supported_handle_types: c_uint,
    deferred_mapping_cuda_array_supported: c_int,
    ipc_event_supported: c_int,
    cluster_launch: c_int,
    unified_function_pointers: c_int,
    _reserved: [2]c_int,
    _reserved2: [1]c_int,
    num_async_engines: c_int,
};

// Function pointer types for Runtime API symbols.
pub const RuntimeGetVersionFn = *const fn (version: *c_int) callconv(.c) c_int;
pub const GetDeviceCountFn = *const fn (count: *c_int) callconv(.c) c_int;
pub const GetDeviceFn = *const fn (device: *c_int) callconv(.c) c_int;
pub const SetDeviceFn = *const fn (device: c_int) callconv(.c) c_int;
pub const GetDevicePropertiesFn = *const fn (prop: *DeviceProp, device: c_int) callconv(.c) c_int;
pub const DeviceResetFn = *const fn () callconv(.c) c_int;
pub const DeviceSynchronizeFn = *const fn () callconv(.c) c_int;
pub const DeviceGetMemInfoFn = *const fn (free: *usize, total: *usize) callconv(.c) c_int;

pub const MallocFn = *const fn (devPtr: *?*anyopaque, size: usize) callconv(.c) c_int;
pub const FreeFn = *const fn (devPtr: ?*anyopaque) callconv(.c) c_int;
pub const MemcpyFn = *const fn (dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_int) callconv(.c) c_int;
pub const MemcpyAsyncFn = *const fn (dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_int, stream: Stream) callconv(.c) c_int;
pub const MemsetFn = *const fn (devPtr: ?*anyopaque, value: c_int, count: usize) callconv(.c) c_int;
pub const MemsetAsyncFn = *const fn (devPtr: ?*anyopaque, value: c_int, count: usize, stream: Stream) callconv(.c) c_int;
pub const HostAllocFn = *const fn (pHost: *?*anyopaque, size: usize, flags: c_uint) callconv(.c) c_int;
pub const FreeHostFn = *const fn (ptr: ?*anyopaque) callconv(.c) c_int;
pub const MallocManagedFn = *const fn (devPtr: *?*anyopaque, size: usize, flags: c_uint) callconv(.c) c_int;
pub const MemPrefetchAsyncFn = *const fn (devPtr: ?*const anyopaque, count: usize, dst_device: c_int, stream: Stream) callconv(.c) c_int;
pub const MemAdviseFn = *const fn (devPtr: ?*const anyopaque, count: usize, advice: c_int, device: c_int) callconv(.c) c_int;

pub const StreamCreateFn = *const fn (stream: *Stream) callconv(.c) c_int;
pub const StreamCreateWithFlagsFn = *const fn (stream: *Stream, flags: c_uint) callconv(.c) c_int;
pub const StreamDestroyFn = *const fn (stream: Stream) callconv(.c) c_int;
pub const StreamSynchronizeFn = *const fn (stream: Stream) callconv(.c) c_int;
pub const StreamQueryFn = *const fn (stream: Stream) callconv(.c) c_int;
pub const StreamWaitEventFn = *const fn (stream: Stream, event: Event, flags: c_uint) callconv(.c) c_int;

pub const EventCreateFn = *const fn (event: *Event) callconv(.c) c_int;
pub const EventCreateWithFlagsFn = *const fn (event: *Event, flags: c_uint) callconv(.c) c_int;
pub const EventDestroyFn = *const fn (event: Event) callconv(.c) c_int;
pub const EventRecordFn = *const fn (event: Event, stream: Stream) callconv(.c) c_int;
pub const EventSynchronizeFn = *const fn (event: Event) callconv(.c) c_int;
pub const EventElapsedTimeFn = *const fn (ms: *f32, start: Event, end: Event) callconv(.c) c_int;
pub const EventQueryFn = *const fn (event: Event) callconv(.c) c_int;

pub const DeviceCanAccessPeerFn = *const fn (can_access: *c_int, device: c_int, peer_device: c_int) callconv(.c) c_int;
pub const DeviceEnablePeerAccessFn = *const fn (peer_device: c_int, flags: c_uint) callconv(.c) c_int;
pub const DeviceDisablePeerAccessFn = *const fn (peer_device: c_int) callconv(.c) c_int;

pub const LaunchKernelFn = *const fn (
    func: ?*const anyopaque,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: usize,
    stream: Stream,
    args: [*]?*anyopaque,
) callconv(.c) c_int;

// Async Memory Pool APIs
pub const MemPool = ?*anyopaque;
pub const MallocAsyncFn = *const fn (devPtr: *?*anyopaque, size: usize, stream: Stream) callconv(.c) c_int;
pub const FreeAsyncFn = *const fn (devPtr: ?*anyopaque, stream: Stream) callconv(.c) c_int;
pub const MemPoolCreateFn = *const fn (pool: *MemPool, props: *const anyopaque) callconv(.c) c_int;
pub const MemPoolDestroyFn = *const fn (pool: MemPool) callconv(.c) c_int;
pub const MallocFromPoolAsyncFn = *const fn (devPtr: *?*anyopaque, size: usize, pool: MemPool, stream: Stream) callconv(.c) c_int;
pub const MemPoolSetAttributeFn = *const fn (pool: MemPool, attr: c_int, value: *anyopaque) callconv(.c) c_int;
pub const MemPoolGetAttributeFn = *const fn (pool: MemPool, attr: c_int, value: *anyopaque) callconv(.c) c_int;

// Pitched 2-D Memory
pub const MallocPitchFn = *const fn (devPtr: *?*anyopaque, pitch: *usize, width_bytes: usize, height: usize) callconv(.c) c_int;
pub const Memcpy2DFn = *const fn (dst: ?*anyopaque, dpitch: usize, src: ?*const anyopaque, spitch: usize, width: usize, height: usize, kind: c_int) callconv(.c) c_int;
pub const Memcpy2DAsyncFn = *const fn (dst: ?*anyopaque, dpitch: usize, src: ?*const anyopaque, spitch: usize, width: usize, height: usize, kind: c_int, stream: Stream) callconv(.c) c_int;

// Occupancy
pub const OccupancyMaxActiveBlocksPerMultiprocessorFn = *const fn (
    num_blocks: *c_int,
    func: ?*const anyopaque,
    block_size: c_int,
    dynamic_smem_size: usize,
) callconv(.c) c_int;
pub const OccupancyMaxPotentialBlockSizeFn = *const fn (
    min_grid_size: *c_int,
    block_size: *c_int,
    func: ?*const anyopaque,
    block_size_to_dynamic_smem_size: ?*const anyopaque,
    dynamic_smem_size: usize,
    block_size_limit: c_int,
) callconv(.c) c_int;

// Stream Priority
pub const StreamGetPriorityFn = *const fn (stream: Stream, priority: *c_int) callconv(.c) c_int;
pub const StreamCreateWithPriorityFn = *const fn (stream: *Stream, flags: c_uint, priority: c_int) callconv(.c) c_int;
pub const DeviceGetStreamPriorityRangeFn = *const fn (least_priority: *c_int, greatest_priority: *c_int) callconv(.c) c_int;

// Cooperative Launch
pub const LaunchCooperativeKernelFn = *const fn (
    func: ?*const anyopaque,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: usize,
    stream: Stream,
    args: [*]?*anyopaque,
) callconv(.c) c_int;

// IPC
pub const IpcMemHandle = [64]u8;
pub const IpcGetMemHandleFn = *const fn (handle: *IpcMemHandle, devPtr: ?*anyopaque) callconv(.c) c_int;
pub const IpcOpenMemHandleFn = *const fn (devPtr: *?*anyopaque, handle: IpcMemHandle, flags: c_uint) callconv(.c) c_int;
pub const IpcCloseMemHandleFn = *const fn (devPtr: ?*anyopaque) callconv(.c) c_int;

test "runtime ffi types compile" {
    // Verify that pointer and struct sizes are plausible.
    try std.testing.expect(@sizeOf(Stream) == @sizeOf(usize));
    try std.testing.expect(@sizeOf(Event) == @sizeOf(usize));
    try std.testing.expect(@sizeOf(DeviceProp) > 256); // name field alone is 256
}

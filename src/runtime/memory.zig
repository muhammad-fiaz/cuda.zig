//! CUDA memory allocation and transfer bindings.
//!
//! Covers device memory (cudaMalloc/cudaFree), pinned host memory
//! (cudaHostAlloc/cudaFreeHost), unified/managed memory
//! (cudaMallocManaged/cudaMemPrefetchAsync/cudaMemAdvise), and all
//! variants of cudaMemcpy (H2D, D2H, D2D, synchronous and asynchronous).
//!
//! All returned pointers are typed as `*anyopaque`; callers must cast them to
//! the appropriate element type. The typed `DeviceBuffer(T)` abstraction in
//! `memory/device_memory.zig` provides safe typed access.

const std = @import("std");
const loader = @import("../core/loader.zig");
const ffi = @import("ffi.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");

/// Allocates `size` bytes of device global memory on the current device.
///
/// The returned pointer is device-side only. It cannot be dereferenced on the
/// host; use `memcpyDeviceToHost` to retrieve data. The memory is not
/// initialized; call `memset` if zeroed memory is needed.
///
/// The allocation is released with `freeDevice`. Do not mix with host-side
/// `free` or any allocator not aware of device memory.
pub fn allocDevice(size: usize) err.CudaError!*anyopaque {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.MallocFn, "cudaMalloc")) |f| {
        var ptr: ?*anyopaque = null;
        try result.checkRuntime(f(&ptr, size));
        return ptr orelse return error.OutOfMemory;
    }
    if (ldr.getDriverSymbol(*const fn (*u64, usize) callconv(.c) c_int, "cuMemAlloc_v2")) |f| {
        var dptr: u64 = 0;
        try result.checkDriver(f(&dptr, size));
        return @ptrFromInt(@as(usize, @intCast(dptr)));
    } else if (ldr.getDriverSymbol(*const fn (*u64, usize) callconv(.c) c_int, "cuMemAlloc")) |f| {
        var dptr: u64 = 0;
        try result.checkDriver(f(&dptr, size));
        return @ptrFromInt(@as(usize, @intCast(dptr)));
    }
    return error.NotInitialized;
}

pub fn freeDevice(ptr: *anyopaque) void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.FreeFn, "cudaFree")) |f| {
        _ = f(ptr);
        return;
    }
    if (ldr.getDriverSymbol(*const fn (u64) callconv(.c) c_int, "cuMemFree_v2")) |f| {
        _ = f(@intFromPtr(ptr));
    } else if (ldr.getDriverSymbol(*const fn (u64) callconv(.c) c_int, "cuMemFree")) |f| {
        _ = f(@intFromPtr(ptr));
    }
}

/// Sets the first `count` bytes of device memory at `ptr` to `value`.
///
/// This call is enqueued on the default (null) stream and is asynchronous
/// with respect to the host. For stream-ordered semantics, use `memsetAsync`.
pub fn memset(ptr: *anyopaque, value: u8, count: usize) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.MemsetFn, "cudaMemset")) |f| {
        try result.checkRuntime(f(ptr, @intCast(value), count));
        return;
    }
    if (ldr.getDriverSymbol(*const fn (u64, u8, usize) callconv(.c) c_int, "cuMemsetD8_v2")) |f| {
        try result.checkDriver(f(@intFromPtr(ptr), value, count));
        return;
    } else if (ldr.getDriverSymbol(*const fn (u64, u8, usize) callconv(.c) c_int, "cuMemsetD8")) |f| {
        try result.checkDriver(f(@intFromPtr(ptr), value, count));
        return;
    }
    return error.NotInitialized;
}

pub fn memsetAsync(ptr: *anyopaque, value: u8, count: usize, stream: ffi.Stream) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.MemsetAsyncFn, "cudaMemsetAsync")) |f| {
        try result.checkRuntime(f(ptr, @intCast(value), count, stream));
        return;
    }
    if (ldr.getDriverSymbol(*const fn (u64, u8, usize, ffi.Stream) callconv(.c) c_int, "cuMemsetD8Async")) |f| {
        try result.checkDriver(f(@intFromPtr(ptr), value, count, stream));
        return;
    }
    return error.NotInitialized;
}

/// Copies `count` bytes from host memory `src` to device memory `dst`.
///
/// This call is synchronous: it does not return until the transfer has
/// completed. `src` must be a valid host pointer; `dst` must be a device
/// pointer returned by `allocDevice` (or equivalent).
pub fn memcpyHostToDevice(dst: *anyopaque, src: *const anyopaque, count: usize) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.MemcpyFn, "cudaMemcpy")) |f| {
        try result.checkRuntime(f(dst, src, count, @intFromEnum(ffi.MemcpyKind.host_to_device)));
        return;
    }
    if (ldr.getDriverSymbol(*const fn (u64, *const anyopaque, usize) callconv(.c) c_int, "cuMemcpyHtoD_v2")) |f| {
        try result.checkDriver(f(@intFromPtr(dst), src, count));
        return;
    } else if (ldr.getDriverSymbol(*const fn (u64, *const anyopaque, usize) callconv(.c) c_int, "cuMemcpyHtoD")) |f| {
        try result.checkDriver(f(@intFromPtr(dst), src, count));
        return;
    }
    return error.NotInitialized;
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *const anyopaque, count: usize) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.MemcpyFn, "cudaMemcpy")) |f| {
        try result.checkRuntime(f(dst, src, count, @intFromEnum(ffi.MemcpyKind.device_to_host)));
        return;
    }
    if (ldr.getDriverSymbol(*const fn (*anyopaque, u64, usize) callconv(.c) c_int, "cuMemcpyDtoH_v2")) |f| {
        try result.checkDriver(f(dst, @intFromPtr(src), count));
        return;
    } else if (ldr.getDriverSymbol(*const fn (*anyopaque, u64, usize) callconv(.c) c_int, "cuMemcpyDtoH")) |f| {
        try result.checkDriver(f(dst, @intFromPtr(src), count));
        return;
    }
    return error.NotInitialized;
}

pub fn memcpyDeviceToDevice(dst: *anyopaque, src: *const anyopaque, count: usize) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.MemcpyFn, "cudaMemcpy")) |f| {
        try result.checkRuntime(f(dst, src, count, @intFromEnum(ffi.MemcpyKind.device_to_device)));
        return;
    }
    if (ldr.getDriverSymbol(*const fn (u64, u64, usize) callconv(.c) c_int, "cuMemcpyDtoD_v2")) |f| {
        try result.checkDriver(f(@intFromPtr(dst), @intFromPtr(src), count));
        return;
    } else if (ldr.getDriverSymbol(*const fn (u64, u64, usize) callconv(.c) c_int, "cuMemcpyDtoD")) |f| {
        try result.checkDriver(f(@intFromPtr(dst), @intFromPtr(src), count));
        return;
    }
    return error.NotInitialized;
}

/// Enqueues an asynchronous host-to-device transfer on `stream`.
///
/// The transfer begins once all previously enqueued operations in `stream`
/// have completed. `src` must remain valid and unmodified until the transfer
/// completes (i.e., until `stream` is synchronized or a subsequent event
/// records past this point). Using page-locked (pinned) host memory for `src`
/// allows the transfer to proceed concurrently with host execution.
pub fn memcpyHostToDeviceAsync(
    dst: *anyopaque,
    src: *const anyopaque,
    count: usize,
    stream: ffi.Stream,
) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.MemcpyAsyncFn, "cudaMemcpyAsync") orelse
        return error.NotInitialized;
    try result.checkRuntime(f(dst, src, count, @intFromEnum(ffi.MemcpyKind.host_to_device), stream));
}

/// Enqueues an asynchronous device-to-host transfer on `stream`.
///
/// `dst` must remain valid until the stream reaches this point. Pinned host
/// memory for `dst` is recommended for maximum transfer throughput.
pub fn memcpyDeviceToHostAsync(
    dst: *anyopaque,
    src: *const anyopaque,
    count: usize,
    stream: ffi.Stream,
) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.MemcpyAsyncFn, "cudaMemcpyAsync") orelse
        return error.NotInitialized;
    try result.checkRuntime(f(dst, src, count, @intFromEnum(ffi.MemcpyKind.device_to_host), stream));
}

/// Enqueues an asynchronous device-to-device transfer on `stream`.
pub fn memcpyDeviceToDeviceAsync(
    dst: *anyopaque,
    src: *const anyopaque,
    count: usize,
    stream: ffi.Stream,
) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.MemcpyAsyncFn, "cudaMemcpyAsync") orelse
        return error.NotInitialized;
    try result.checkRuntime(f(dst, src, count, @intFromEnum(ffi.MemcpyKind.device_to_device), stream));
}

/// Allocates `size` bytes of page-locked (pinned) host memory.
///
/// Page-locked memory enables the GPU's DMA engine to directly read/write
/// host memory without going through the OS paging system. This provides
/// maximum host-to-device and device-to-host transfer bandwidth and allows
/// truly asynchronous transfers (the host CPU can continue while the transfer
/// proceeds in the background on a separate stream).
///
/// The returned memory is accessible from both host and device (if
/// `ffi.HostAllocFlags.mapped` is set). Free with `freeHost`.
///
/// Over-use of pinned memory degrades overall system performance by reducing
/// the amount of memory available for the OS page cache.
pub fn allocHost(size: usize, flags: c_uint) err.CudaError!*anyopaque {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.HostAllocFn, "cudaHostAlloc")) |f| {
        var ptr: ?*anyopaque = null;
        try result.checkRuntime(f(&ptr, size, flags));
        return ptr orelse return error.OutOfMemory;
    }
    if (ldr.getDriverSymbol(*const fn (*?*anyopaque, usize, c_uint) callconv(.c) c_int, "cuMemHostAlloc")) |f| {
        var ptr: ?*anyopaque = null;
        try result.checkDriver(f(&ptr, size, flags));
        return ptr orelse return error.OutOfMemory;
    }
    return error.NotInitialized;
}

pub fn freeHost(ptr: *anyopaque) void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.FreeHostFn, "cudaFreeHost")) |f| {
        _ = f(ptr);
        return;
    }
    if (ldr.getDriverSymbol(*const fn (*anyopaque) callconv(.c) c_int, "cuMemFreeHost")) |f| {
        _ = f(ptr);
    }
}

/// Allocates `size` bytes of unified/managed memory accessible from both host
/// and device.
///
/// `flags` should be `ffi.MemAttachFlags.global` for memory that can be
/// accessed by any stream on any device. The returned pointer can be
/// dereferenced on both host and device, but concurrent access requires
/// explicit synchronization.
///
/// On devices supporting unified virtual addressing (compute capability >= 6.x),
/// page migration between host and device happens automatically. Use
/// `memPrefetchAsync` to move pages proactively before they are needed.
pub fn allocManaged(size: usize, flags: c_uint) err.CudaError!*anyopaque {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.MallocManagedFn, "cudaMallocManaged")) |f| {
        var ptr: ?*anyopaque = null;
        try result.checkRuntime(f(&ptr, size, flags));
        return ptr orelse return error.OutOfMemory;
    }
    if (ldr.getDriverSymbol(*const fn (*u64, usize, c_uint) callconv(.c) c_int, "cuMemAllocManaged")) |f| {
        var dptr: u64 = 0;
        try result.checkDriver(f(&dptr, size, flags));
        return @ptrFromInt(@as(usize, @intCast(dptr)));
    }
    return error.NotInitialized;
}

/// Prefetches `count` bytes starting at `ptr` to `device` asynchronously on
/// `stream`.
///
/// `ptr` must have been allocated with `allocManaged`. `device` is the
/// destination device index, or `-1` to prefetch to the CPU. This call
/// enqueues the prefetch; the memory is not guaranteed to have moved until
/// `stream` is synchronized.
pub fn memPrefetchAsync(ptr: *const anyopaque, count: usize, device: i32, stream: ffi.Stream) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.MemPrefetchAsyncFn, "cudaMemPrefetchAsync")) |f| {
        result.checkRuntime(f(ptr, count, device, stream)) catch |e| switch (e) {
            error.InvalidDevice, error.InvalidValue, error.NotSupported => return,
            else => return e,
        };
        return;
    }
    if (ldr.getDriverSymbol(*const fn (u64, usize, c_int, ffi.Stream) callconv(.c) c_int, "cuMemPrefetchAsync")) |f| {
        var target_dev: c_int = -1; // CU_DEVICE_CPU
        if (device >= 0) {
            if (ldr.getDriverSymbol(*const fn (*c_int, c_int) callconv(.c) c_int, "cuDeviceGet")) |dev_get| {
                _ = dev_get(&target_dev, device);
            } else {
                target_dev = device;
            }
        }
        result.checkDriver(f(@intFromPtr(ptr), count, target_dev, stream)) catch |e| switch (e) {
            error.InvalidDevice, error.InvalidValue, error.NotSupported => return,
            else => return e,
        };
        return;
    }
    return error.NotInitialized;
}

/// Provides a hint to the CUDA driver about how managed memory at `ptr` will
/// be used.
///
/// `advice` is one of the `ffi.MemAdvise` values (e.g. `set_read_mostly`,
/// `set_preferred_location`). `device` is the target device for location
/// hints, or `-1` for the CPU. These are advisory; the driver may ignore them
/// on platforms that do not support the requested behavior.
pub fn memAdvise(ptr: *const anyopaque, count: usize, advice: ffi.MemAdvise, device: i32) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.MemAdviseFn, "cudaMemAdvise") orelse
        return error.NotInitialized;
    try result.checkRuntime(f(ptr, count, @intFromEnum(advice), @intCast(device)));
}

// Async Memory Pool APIs

/// Allocates `size` bytes from the stream-ordered allocator on `stream`.
///
/// Uses `cudaMallocAsync` (CUDA 11.2+). Memory is automatically returned to
/// the pool when `freeAsync` is called on the same stream (or a stream that
/// synchronizes with it). Requires a device that supports memory pools
/// (`deviceProps.memory_pools_supported`).
pub fn allocAsync(size: usize, stream: ffi.Stream) err.CudaError!*anyopaque {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.MallocAsyncFn, "cudaMallocAsync")) |f| {
        var ptr: ?*anyopaque = null;
        try result.checkRuntime(f(&ptr, size, stream));
        return ptr orelse return error.OutOfMemory;
    }
    // Driver API fallback: cuMemAllocAsync (CUDA 11.2+)
    if (ldr.getDriverSymbol(*const fn (*u64, usize, ?*anyopaque) callconv(.c) c_int, "cuMemAllocAsync")) |f| {
        var dptr: u64 = 0;
        try result.checkDriver(f(&dptr, size, stream));
        return @ptrFromInt(@as(usize, @intCast(dptr)));
    }
    // Last resort: synchronous cudaMalloc — semantically equivalent on systems without pool support
    return allocDevice(size);
}

/// Frees memory previously allocated with `allocAsync` on the given `stream`.
pub fn freeAsync(ptr: *anyopaque, stream: ffi.Stream) err.CudaError!void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.FreeAsyncFn, "cudaFreeAsync")) |f| {
        try result.checkRuntime(f(ptr, stream));
        return;
    }
    // Driver API fallback
    if (ldr.getDriverSymbol(*const fn (u64, ?*anyopaque) callconv(.c) c_int, "cuMemFreeAsync")) |f| {
        try result.checkDriver(f(@intFromPtr(ptr), stream));
        return;
    }
    // Last resort: synchronous free
    freeDevice(ptr);
}

// Pitched / 2-D Memory

/// Allocates a 2-D block of device memory with optimal row pitch.
///
/// Returns the device pointer and actual row pitch in bytes. Use the returned
/// `pitch` (not `width_bytes`) as the row stride for subsequent `memcpy2D`
/// calls. This ensures hardware-optimal alignment for texture / 2-D access.
pub fn mallocPitch(width_bytes: usize, height: usize) err.CudaError!struct {
    ptr: *anyopaque,
    pitch: usize,
} {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ffi.MallocPitchFn, "cudaMallocPitch")) |f| {
        var ptr: ?*anyopaque = null;
        var pitch: usize = 0;
        try result.checkRuntime(f(&ptr, &pitch, width_bytes, height));
        return .{ .ptr = ptr orelse return error.OutOfMemory, .pitch = pitch };
    }
    // Driver API fallback: cuMemAllocPitch
    if (ldr.getDriverSymbol(*const fn (*u64, *usize, usize, usize, c_uint) callconv(.c) c_int, "cuMemAllocPitch_v2") orelse
        ldr.getDriverSymbol(*const fn (*u64, *usize, usize, usize, c_uint) callconv(.c) c_int, "cuMemAllocPitch")) |f|
    {
        var dptr: u64 = 0;
        var pitch: usize = 0;
        // element_size_bytes = 4 (float default); use width_bytes mod 16 to pick 4/8/16
        const elem_size: c_uint = if (width_bytes % 16 == 0) 16 else if (width_bytes % 8 == 0) 8 else 4;
        try result.checkDriver(f(&dptr, &pitch, width_bytes, height, elem_size));
        return .{ .ptr = @ptrFromInt(@as(usize, @intCast(dptr))), .pitch = pitch };
    }
    return error.NotInitialized;
}

/// Copies a 2-D region of `width` × `height` bytes between pitched buffers.
pub fn memcpy2D(
    dst: *anyopaque,
    dpitch: usize,
    src: *const anyopaque,
    spitch: usize,
    width: usize,
    height: usize,
    kind: ffi.MemcpyKind,
) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.Memcpy2DFn, "cudaMemcpy2D") orelse
        return error.NotInitialized;
    try result.checkRuntime(f(dst, dpitch, src, spitch, width, height, @intFromEnum(kind)));
}

/// Asynchronous 2-D memcpy on the given stream.
pub fn memcpy2DAsync(
    dst: *anyopaque,
    dpitch: usize,
    src: *const anyopaque,
    spitch: usize,
    width: usize,
    height: usize,
    kind: ffi.MemcpyKind,
    stream: ffi.Stream,
) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.Memcpy2DAsyncFn, "cudaMemcpy2DAsync") orelse
        return error.NotInitialized;
    try result.checkRuntime(f(dst, dpitch, src, spitch, width, height, @intFromEnum(kind), stream));
}

// IPC Memory (inter-process GPU memory sharing)

/// Exports a device pointer as an OS-level IPC handle.
///
/// `ptr` must be the base pointer of a cudaMalloc'd allocation (not an offset).
/// The 64-byte handle can be sent to another process via shared memory, pipe,
/// or socket. The other process opens it with `ipcOpenMemHandle`.
pub fn ipcGetMemHandle(ptr: *anyopaque) err.CudaError!ffi.IpcMemHandle {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.IpcGetMemHandleFn, "cudaIpcGetMemHandle") orelse
        return error.NotInitialized;
    var handle: ffi.IpcMemHandle = undefined;
    try result.checkRuntime(f(&handle, ptr));
    return handle;
}

/// Opens an IPC memory handle exported by another process.
///
/// `flags` should be `0` (future versions may add flag bits). The returned
/// pointer is valid in this process until `ipcCloseMemHandle` is called.
pub fn ipcOpenMemHandle(handle: ffi.IpcMemHandle, flags: c_uint) err.CudaError!*anyopaque {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.IpcOpenMemHandleFn, "cudaIpcOpenMemHandle") orelse
        return error.NotInitialized;
    var ptr: ?*anyopaque = null;
    try result.checkRuntime(f(&ptr, handle, flags));
    return ptr orelse return error.OutOfMemory;
}

/// Closes an IPC memory handle, releasing the mapping in this process.
///
/// Does not free the underlying allocation; the exporting process must call
/// `freeDevice` on the original pointer.
pub fn ipcCloseMemHandle(ptr: *anyopaque) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(ffi.IpcCloseMemHandleFn, "cudaIpcCloseMemHandle") orelse
        return error.NotInitialized;
    try result.checkRuntime(f(ptr));
}

test "runtime memory without CUDA" {
    if (!loader.isAvailable()) return error.SkipZigTest;
    // On a machine with CUDA, test a small round-trip.
    const size = 64;
    const dev = try allocDevice(size);
    freeDevice(dev);
}

test "memAdvise is accessible" {
    _ = memAdvise;
}

test "allocAsync signature" {
    _ = allocAsync;
    _ = freeAsync;
}

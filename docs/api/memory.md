---
title: Memory API
description: Device memory, host-pinned memory, managed memory, transfer helpers, and CudaAllocator in cuda.zig.
---

# Memory API

## `cuda.DeviceBuffer(T)`

```zig
pub fn DeviceBuffer(comptime T: type) type {
    return struct {
        /// Allocate `len` elements in device memory.
        pub fn init(allocator: std.mem.Allocator, len: usize) !@This()

        /// Free device memory.
        pub fn deinit(self: *@This()) void

        /// Return the raw device pointer (CUdeviceptr).
        pub fn devicePtr(self: @This()) CUdeviceptr

        /// Return the element count.
        pub fn len(self: @This()) usize

        /// Fill all bytes with `value`.
        pub fn memset(self: @This(), value: u8) !void

        /// Synchronous host → device copy.
        pub fn copyFromHost(self: @This(), src: []const T) !void

        /// Synchronous device → host copy.
        pub fn copyToHost(self: @This(), dst: []T) !void

        /// Asynchronous host → device copy.
        pub fn copyFromHostAsync(self: @This(), src: []const T, stream: Stream) !void

        /// Asynchronous device → host copy.
        pub fn copyToHostAsync(self: @This(), dst: []T, stream: Stream) !void

        /// Device → device copy (same device).
        pub fn copyFromDevice(self: @This(), src: @This()) !void
    };
}
```

## `cuda.HostBuffer(T)`

```zig
pub fn HostBuffer(comptime T: type) type {
    return struct {
        /// Allocate `len` elements in page-locked host memory (cudaHostAlloc).
        pub fn init(allocator: std.mem.Allocator, len: usize) !@This()
        pub fn deinit(self: *@This()) void

        /// Access the host-side slice directly.
        pub fn slice(self: @This()) []T

        /// Raw pinned pointer (for passing to cudaMemcpyAsync).
        pub fn ptr(self: @This()) *T
    };
}
```

## `cuda.ManagedBuffer(T)`

```zig
pub fn ManagedBuffer(comptime T: type) type {
    return struct {
        /// Allocate `len` elements in unified memory (cudaMallocManaged).
        pub fn init(allocator: std.mem.Allocator, len: usize) !@This()
        pub fn deinit(self: *@This()) void

        /// Access as a host-side slice (caution: synchronise first).
        pub fn slice(self: @This()) []T

        /// Prefetch data to the specified device (cudaMemPrefetchAsync).
        pub fn prefetchToDevice(self: @This(), device: u32, stream: Stream) !void

        /// Prefetch data back to CPU (device = cudaCpuDeviceId).
        pub fn prefetchToHost(self: @This(), stream: Stream) !void

        /// Set memory access advice (cudaMemAdvise).
        pub fn advise(self: @This(), advice: MemAdvise, device: u32) !void
    };
}
```

### `MemAdvise`

```zig
pub const MemAdvise = enum {
    SetReadMostly,
    UnsetReadMostly,
    PreferredLocation,
    UnsetPreferredLocation,
    AccessedBy,
    UnsetAccessedBy,
};
```

## `cuda.memory` — Transfer Helpers

```zig
/// Peer-to-peer device copy.
pub fn copyPeer(
    dst: CUdeviceptr, dst_device: u32,
    src: CUdeviceptr, src_device: u32,
    bytes: usize,
) !void
```

## `cuda.CudaAllocator`

```zig
pub const CudaAllocator = struct {
    /// Create a CudaAllocator backed by cudaMalloc / cudaFree.
    pub fn init() CudaAllocator

    /// Return a std.mem.Allocator interface.
    pub fn allocator(self: *CudaAllocator) std.mem.Allocator
};
```

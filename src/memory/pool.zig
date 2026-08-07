//! High-level CUDA Memory Pool abstraction
//!
//! Wraps `cudaMallocAsync` / `cudaFreeAsync` behind a typed `PoolBuffer(T)`
//! that mirrors the `DeviceBuffer(T)` API so the two can be used
//! interchangeably where the backing allocator does not matter.
//!
//! Memory pools reduce allocation latency by reusing previously freed memory
//! without returning it to the OS between kernels.  They require:
//!   - CUDA 11.2 or later (driver version >= 11.2)
//!   - `deviceProps.memory_pools_supported == true`
//!
//! Usage:
//! ```zig
//! const cuda = @import("cuda");
//! const stream = try cuda.Stream.create();
//! var buf = try cuda.PoolBuffer(f32).alloc(1024, stream.handle);
//! defer buf.freeOnStream(stream.handle) catch {};
//! ```

const std = @import("std");
const loader = @import("../core/loader.zig");
const rt_memory = @import("../runtime/memory.zig");
const ffi = @import("../runtime/ffi.zig");
const err = @import("../core/error.zig");

/// A stream-ordered device memory buffer backed by the CUDA memory pool.
///
/// All operations are tied to the stream provided at construction time.
/// You must free this buffer on a stream that synchronizes before the
/// next allocation reuses the same region.
pub fn PoolBuffer(comptime T: type) type {
    return struct {
        ptr: [*]T,
        len: usize,

        const Self = @This();

        /// Allocates `count` elements of type `T` from the stream-ordered pool.
        pub fn alloc(count: usize, stream: ffi.Stream) err.CudaError!Self {
            const raw = try rt_memory.allocAsync(count * @sizeOf(T), stream);
            return Self{
                .ptr = @ptrCast(@alignCast(raw)),
                .len = count,
            };
        }

        /// Returns this buffer's storage to the pool on `stream`.
        ///
        /// The memory may be immediately reused for the next `alloc` on a
        /// stream that synchronizes with `stream`.
        pub fn freeOnStream(self: Self, stream: ffi.Stream) err.CudaError!void {
            try rt_memory.freeAsync(@ptrCast(self.ptr), stream);
        }

        /// Returns a typed device slice view (read-only from the host).
        pub fn slice(self: Self) []T {
            return self.ptr[0..self.len];
        }

        /// Returns the allocation size in bytes.
        pub fn byteSize(self: Self) usize {
            return self.len * @sizeOf(T);
        }
    };
}

test "PoolBuffer type creation" {
    const Buf = PoolBuffer(f32);
    _ = Buf;
}

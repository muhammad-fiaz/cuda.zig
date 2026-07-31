//! Pinned (page-locked) host memory for high-throughput DMA transfers.
//!
//! Page-locked memory bypasses the OS paging system, enabling the GPU's DMA
//! engine to directly read and write host memory without a software-managed
//! bounce buffer. This allows:
//!   - Truly asynchronous transfers (host CPU runs concurrently with DMA)
//!   - Higher sustained transfer bandwidth
//!   - Mapping of host memory into the device address space (zero-copy)
//!
//! Use pinned memory as the host side of `DeviceBuffer.copyFromHostAsync` /
//! `copyToHostAsync` for maximum throughput.
//!
//! On the CPU fallback path, `PinnedBuffer` degrades to a regular heap
//! allocation; the semantics are identical from the caller's perspective.

const std = @import("std");
const loader = @import("../core/loader.zig");
const mem_rt = @import("../runtime/memory.zig");
const ffi = @import("../runtime/ffi.zig");
const err = @import("../core/error.zig");
const dispatch = @import("../fallback/dispatch.zig");

/// A typed, owned pinned host allocation.
pub fn PinnedBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        ptr: [*]T,
        len: usize,
        /// `true` if backed by `cudaHostAlloc`; `false` if a CPU fallback.
        pinned: bool,

        /// Allocates `count` elements of `T` in page-locked host memory.
        ///
        /// `flags` controls the allocation behavior; use
        /// `ffi.HostAllocFlags.default` for portable, timing-enabled pinned
        /// memory. Pass `ffi.HostAllocFlags.write_combined` for write-only
        /// memory that bypasses the CPU cache for maximum write bandwidth.
        ///
        /// Falls back to normal heap allocation if CUDA is unavailable.
        pub fn alloc(count: usize, flags: c_uint) err.CudaError!Self {
            if (dispatch.isGpuAvailable()) {
                const raw = try mem_rt.allocHost(count * @sizeOf(T), flags);
                return Self{ .ptr = @ptrCast(@alignCast(raw)), .len = count, .pinned = true };
            } else {
                const raw = try std.heap.page_allocator.alloc(T, count);
                return Self{ .ptr = raw.ptr, .len = count, .pinned = false };
            }
        }

        /// Releases the pinned memory allocation.
        ///
        /// Idempotent: subsequent calls after the first are no-ops.
        pub fn free(self: *Self) void {
            if (self.len == 0) return;
            if (self.pinned) {
                mem_rt.freeHost(@ptrCast(self.ptr));
            } else {
                std.heap.page_allocator.free(self.ptr[0..self.len]);
            }
            self.len = 0;
        }

        /// Returns a slice view of the pinned memory, usable from the host.
        pub fn slice(self: Self) []T {
            return self.ptr[0..self.len];
        }

        /// Returns a const slice view of the pinned memory.
        pub fn constSlice(self: Self) []const T {
            return self.ptr[0..self.len];
        }
    };
}

test "PinnedBuffer CPU fallback" {
    var buf = try PinnedBuffer(u8).alloc(16, ffi.HostAllocFlags.default);
    defer buf.free();
    try std.testing.expectEqual(@as(usize, 16), buf.len);
    @memset(buf.slice(), 0xAB);
    try std.testing.expect(buf.slice()[0] == 0xAB);
}

test "PinnedBuffer free idempotent" {
    var buf = try PinnedBuffer(f64).alloc(4, ffi.HostAllocFlags.default);
    buf.free();
    buf.free();
}

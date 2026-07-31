//! Unified (managed) memory: accessible from both host and device.
//!
//! Unified memory lets the programmer work with a single pointer that the CUDA
//! driver automatically migrates between host and device as needed. This
//! eliminates explicit H2D/D2H transfers for many use cases, at the cost of
//! potentially higher latency on first access to migrated pages.
//!
//! For performance-critical workloads, use `memPrefetchAsync` to move pages
//! to the target device before they are needed, and `memAdvise` to hint the
//! driver about access patterns.
//!
//! On the CPU fallback path, `UnifiedBuffer` degrades to a regular heap
//! allocation since unified memory is a CUDA-only concept.

const std = @import("std");
const loader = @import("../core/loader.zig");
const mem_rt = @import("../runtime/memory.zig");
const ffi = @import("../runtime/ffi.zig");
const stream_mod = @import("../stream/stream.zig");
const err = @import("../core/error.zig");
const dispatch = @import("../fallback/dispatch.zig");

/// A typed, owned unified memory allocation.
pub fn UnifiedBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        ptr: [*]T,
        len: usize,
        on_device: bool,

        /// Allocates `count` elements of `T` as unified/managed memory.
        ///
        /// `flags` should be `ffi.MemAttachFlags.global` for most use cases.
        /// The returned memory is accessible from both host and device code.
        /// The driver migrates pages automatically on access; use
        /// `prefetchToDevice` to improve first-access latency.
        pub fn alloc(count: usize, flags: c_uint) err.CudaError!Self {
            if (dispatch.isGpuAvailable()) {
                const raw = try mem_rt.allocManaged(count * @sizeOf(T), flags);
                return Self{ .ptr = @ptrCast(@alignCast(raw)), .len = count, .on_device = true };
            } else {
                const raw = try dispatch.cpuAlloc(count * @sizeOf(T));
                return Self{ .ptr = @ptrCast(@alignCast(raw)), .len = count, .on_device = false };
            }
        }

        /// Releases the unified memory allocation.
        ///
        /// Idempotent: after the first call subsequent calls are no-ops.
        pub fn free(self: *Self) void {
            if (self.len == 0) return;
            if (self.on_device) {
                mem_rt.freeDevice(@ptrCast(self.ptr));
            } else {
                dispatch.cpuFree(@ptrCast(self.ptr), self.len * @sizeOf(T));
            }
            self.len = 0;
        }

        /// Returns a host-accessible slice of the unified memory.
        ///
        /// Accessing this slice from the host while the device is concurrently
        /// reading or writing the same pages is undefined behavior. Synchronize
        /// before accessing from a different processor.
        pub fn slice(self: Self) []T {
            return self.ptr[0..self.len];
        }

        /// Asynchronously prefetches the buffer contents to `device_index` on
        /// `stream`.
        ///
        /// This is a hint; the driver may ignore it on hardware that does not
        /// support concurrent managed access. After `stream` completes past this
        /// point, accessing the buffer on `device_index` will not trigger page
        /// faults.
        pub fn prefetchToDevice(self: Self, device_index: i32, stream: stream_mod.Stream) err.CudaError!void {
            if (!self.on_device) return; // no-op on CPU fallback
            try mem_rt.memPrefetchAsync(@ptrCast(self.ptr), self.len * @sizeOf(T), device_index, stream.handle);
        }

        /// Asynchronously prefetches the buffer contents to the host (CPU) on
        /// `stream`.
        pub fn prefetchToHost(self: Self, stream: stream_mod.Stream) err.CudaError!void {
            if (!self.on_device) return;
            try mem_rt.memPrefetchAsync(@ptrCast(self.ptr), self.len * @sizeOf(T), -1, stream.handle);
        }

        /// Sets a usage hint for the driver's memory management policy.
        ///
        /// `advice` is one of the `ffi.MemAdvise` values. `device` is the
        /// target device for location hints, or `-1` for the CPU.
        pub fn advise(self: Self, advice_val: ffi.MemAdvise, device: i32) err.CudaError!void {
            if (!self.on_device) return;
            try mem_rt.memAdvise(@ptrCast(self.ptr), self.len * @sizeOf(T), advice_val, device);
        }
    };
}

test "UnifiedBuffer CPU fallback alloc/free" {
    var buf = try UnifiedBuffer(i32).alloc(8, ffi.MemAttachFlags.global);
    defer buf.free();
    try std.testing.expectEqual(@as(usize, 8), buf.len);
    buf.slice()[0] = 42;
    try std.testing.expectEqual(@as(i32, 42), buf.slice()[0]);
}

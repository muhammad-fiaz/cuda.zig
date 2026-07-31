//! DeviceBuffer(T): a typed, owned device-side memory allocation.
//!
//! `DeviceBuffer(T)` is the primary abstraction for interacting with GPU
//! memory from the host. It holds a device pointer, an element count, and the
//! byte size of the allocation. It owns the allocation and must be freed via
//! `free()`.
//!
//! All transfer functions accept typed slices; byte-level size calculation is
//! handled internally. Async transfers enqueue onto a `Stream` and return
//! immediately; the host must synchronize the stream before accessing the
//! destination memory.
//!
//! On machines without CUDA the dispatch layer routes `alloc` through the
//! fallback CPU path.

const std = @import("std");
const loader = @import("../core/loader.zig");
const mem_rt = @import("../runtime/memory.zig");
const stream_mod = @import("../stream/stream.zig");
const err = @import("../core/error.zig");
const dispatch = @import("../fallback/dispatch.zig");

/// Returns a struct representing a typed device-memory buffer of `T`.
///
/// Usage:
/// ```zig
/// var buf = try DeviceBuffer(f32).alloc(1024);
/// defer buf.free();
/// try buf.copyFromHost(&[_]f32{ 1.0, 2.0, 3.0 }[0..3]);
/// ```
pub fn DeviceBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Device-side pointer. Opaque on the host; do not dereference.
        ptr: *anyopaque,
        /// Number of elements of type `T` in the allocation.
        len: usize,
        /// Whether this buffer lives on-device (true) or in the CPU fallback
        /// heap (false). The public API is identical in both cases.
        on_device: bool,

        /// Allocates storage for `count` elements of `T` on the current device.
        ///
        /// If CUDA is available, the memory is allocated via `cudaMalloc`.
        /// Otherwise it is allocated from the process heap as a CPU fallback.
        /// In either case the memory is not initialized.
        ///
        /// The allocation is released by calling `free`. Do not free the pointer
        /// manually or via an allocator; always go through `free`.
        pub fn alloc(count: usize) err.CudaError!Self {
            if (dispatch.isGpuAvailable()) {
                const raw = try mem_rt.allocDevice(count * @sizeOf(T));
                return Self{ .ptr = raw, .len = count, .on_device = true };
            } else {
                const raw = dispatch.cpuAlloc(count * @sizeOf(T)) catch return error.OutOfMemory;
                return Self{ .ptr = raw, .len = count, .on_device = false };
            }
        }

        /// Releases the device (or CPU fallback) memory held by this buffer.
        ///
        /// Idempotent: after the first call the pointer is nulled; subsequent
        /// calls are no-ops.
        pub fn free(self: *Self) void {
            if (self.len == 0) return;
            if (self.on_device) {
                mem_rt.freeDevice(self.ptr);
            } else {
                dispatch.cpuFree(self.ptr, self.len * @sizeOf(T));
            }
            self.len = 0;
        }

        /// Copies `data` from host memory into this device buffer.
        ///
        /// `data.len` must equal `self.len`. This call is synchronous: it does
        /// not return until the transfer has completed.
        pub fn copyFromHost(self: *Self, data: []const T) err.CudaError!void {
            if (data.len != self.len) return error.InvalidValue;
            if (self.on_device) {
                try mem_rt.memcpyHostToDevice(self.ptr, @ptrCast(data.ptr), data.len * @sizeOf(T));
            } else {
                const dst: [*]T = @ptrCast(@alignCast(self.ptr));
                @memcpy(dst[0..self.len], data);
            }
        }

        /// Copies this buffer's contents into the host slice `out`.
        ///
        /// `out.len` must equal `self.len`. This call is synchronous.
        pub fn copyToHost(self: Self, out: []T) err.CudaError!void {
            if (out.len != self.len) return error.InvalidValue;
            if (self.on_device) {
                try mem_rt.memcpyDeviceToHost(@ptrCast(out.ptr), self.ptr, out.len * @sizeOf(T));
            } else {
                const src: [*]const T = @ptrCast(@alignCast(self.ptr));
                @memcpy(out, src[0..self.len]);
            }
        }

        /// Enqueues an asynchronous host-to-device transfer on `stream`.
        ///
        /// `data` must remain valid and unmodified until the stream is
        /// synchronized past this point. For maximum throughput, `data` should
        /// reside in page-locked (pinned) host memory.
        ///
        /// On the CPU fallback path this degrades to a synchronous copy.
        pub fn copyFromHostAsync(self: Self, data: []const T, stream: stream_mod.Stream) err.CudaError!void {
            if (data.len != self.len) return error.InvalidValue;
            if (self.on_device) {
                try mem_rt.memcpyHostToDeviceAsync(
                    self.ptr,
                    @ptrCast(data.ptr),
                    data.len * @sizeOf(T),
                    stream.handle,
                );
            } else {
                // CPU fallback: stream ordering has no meaning; the handle is
                // captured here only to prevent the "unused parameter" warning.
                const _s = stream.handle;
                _ = _s;
                const dst: [*]T = @ptrCast(@alignCast(self.ptr));
                @memcpy(dst[0..self.len], data);
            }
        }

        /// Enqueues an asynchronous device-to-host transfer on `stream`.
        ///
        /// `out` must remain valid until the stream is synchronized past this
        /// point. Pinned host memory for `out` is recommended.
        pub fn copyToHostAsync(self: Self, out: []T, stream: stream_mod.Stream) err.CudaError!void {
            if (out.len != self.len) return error.InvalidValue;
            if (self.on_device) {
                try mem_rt.memcpyDeviceToHostAsync(
                    @ptrCast(out.ptr),
                    self.ptr,
                    out.len * @sizeOf(T),
                    stream.handle,
                );
            } else {
                const _s = stream.handle;
                _ = _s;
                const src: [*]const T = @ptrCast(@alignCast(self.ptr));
                @memcpy(out, src[0..self.len]);
            }
        }

        /// Sets all elements in the buffer to zero.
        ///
        /// On the device path this uses `cudaMemset`. On the CPU path it uses
        /// `@memset`.
        pub fn fill(self: *Self, value: u8) err.CudaError!void {
            if (self.on_device) {
                try mem_rt.memset(self.ptr, value, self.len * @sizeOf(T));
            } else {
                const dst: [*]u8 = @ptrCast(self.ptr);
                @memset(dst[0 .. self.len * @sizeOf(T)], value);
            }
        }
    };
}

test "DeviceBuffer CPU fallback alloc/copy round-trip" {
    // This test must run unconditionally regardless of GPU presence.
    var buf = try DeviceBuffer(f32).alloc(4);
    defer buf.free();
    try std.testing.expectEqual(@as(usize, 4), buf.len);

    const src = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    try buf.copyFromHost(&src);

    var dst = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
    try buf.copyToHost(&dst);

    try std.testing.expectEqualSlices(f32, &src, &dst);
}

test "DeviceBuffer free is idempotent" {
    var buf = try DeviceBuffer(u32).alloc(8);
    buf.free();
    buf.free(); // must not crash
}

test "DeviceBuffer GPU alloc round-trip" {
    if (!loader.isAvailable()) return error.SkipZigTest;
    // Skip if no physical device is present.
    const rt = @import("../runtime/device.zig");
    const count = rt.getDeviceCount() catch return error.SkipZigTest;
    if (count == 0) return error.SkipZigTest;

    var buf = try DeviceBuffer(f32).alloc(4);
    defer buf.free();

    const src = [4]f32{ 10.0, 20.0, 30.0, 40.0 };
    try buf.copyFromHost(&src);

    var dst = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
    try buf.copyToHost(&dst);
    try std.testing.expectEqualSlices(f32, &src, &dst);
}

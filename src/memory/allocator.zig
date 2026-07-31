//! CudaAllocator: implements std.mem.Allocator over device global memory.
//!
//! This makes device memory usable wherever a `std.mem.Allocator` is expected,
//! enabling interop with any Zig code that works with allocators. The
//! allocator falls back to the system heap when CUDA is unavailable.
//!
//! Important: memory returned by this allocator is device-side. Slices
//! produced by `alloc` must not be dereferenced on the host. They can be
//! passed to `DeviceBuffer` or used as raw pointers in FFI kernel calls.
//!
//! Thread safety: the allocator is stateless; multiple threads may use it
//! simultaneously. Device allocation itself is serialized by the CUDA driver.

const std = @import("std");
const mem_rt = @import("../runtime/memory.zig");
const dispatch = @import("../fallback/dispatch.zig");
const err = @import("../core/error.zig");

/// Returns a `std.mem.Allocator` that allocates device global memory.
///
/// The allocator is a thin stateless wrapper; the returned value is valid
/// for the lifetime of the process. No `deinit` is required.
pub fn cudaAllocator() std.mem.Allocator {
    return .{
        .ptr = undefined,
        .vtable = &cuda_vtable,
    };
}

fn cudaAlloc(_: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
    _ = alignment;
    if (dispatch.isGpuAvailable()) {
        const ptr = mem_rt.allocDevice(len) catch return null;
        return @ptrCast(ptr);
    } else {
        const ptr = dispatch.cpuAlloc(len) catch return null;
        return @ptrCast(ptr);
    }
}

fn cudaResize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
    // Device memory does not support in-place resizing.
    return false;
}

fn cudaRemap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
    return null;
}

fn cudaFreeSlice(_: *anyopaque, buf: []u8, _: std.mem.Alignment, _: usize) void {
    if (dispatch.isGpuAvailable()) {
        mem_rt.freeDevice(@ptrCast(buf.ptr));
    } else {
        dispatch.cpuFree(@ptrCast(buf.ptr), buf.len);
    }
}

const cuda_vtable: std.mem.Allocator.VTable = .{
    .alloc = cudaAlloc,
    .resize = cudaResize,
    .remap = cudaRemap,
    .free = cudaFreeSlice,
};

test "cudaAllocator returns a valid Allocator interface" {
    if (!dispatch.isGpuAvailable()) {
        const alloc = cudaAllocator();
        const buf = alloc.alloc(u8, 64) catch return;
        defer alloc.free(buf);
        try std.testing.expectEqual(@as(usize, 64), buf.len);
    }
}

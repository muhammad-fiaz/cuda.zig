//! High-level Tensor(T) abstraction.
//!
//! Provides a N-dimensional tensor backed by `DeviceBuffer(T)` with seamless CPU/GPU routing.

const std = @import("std");
const dev_buf = @import("../memory/device_memory.zig");
const shape_mod = @import("shape.zig");
const dtype_mod = @import("dtype.zig");
const elem_ops = @import("ops/elementwise.zig");
const matmul_op = @import("ops/matmul.zig");
const err = @import("../core/error.zig");

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        buffer: dev_buf.DeviceBuffer(T),
        shape: shape_mod.Shape,

        /// Creates a Tensor filled with zeros with the given shape.
        pub fn zeros(shape_slice: []const usize) !Self {
            const shp = try shape_mod.Shape.init(shape_slice);
            const total = shp.totalElements();
            var buf = try dev_buf.DeviceBuffer(T).alloc(total);
            errdefer buf.free();
            try buf.fill(0);
            return Self{
                .buffer = buf,
                .shape = shp,
            };
        }

        /// Creates a Tensor initialized from a host slice.
        pub fn fromSlice(data: []const T, shape_slice: []const usize) !Self {
            const shp = try shape_mod.Shape.init(shape_slice);
            if (data.len != shp.totalElements()) return error.InvalidValue;
            var buf = try dev_buf.DeviceBuffer(T).alloc(data.len);
            errdefer buf.free();
            try buf.copyFromHost(data);
            return Self{
                .buffer = buf,
                .shape = shp,
            };
        }

        /// Copies tensor data back to host memory allocated with `allocator`.
        pub fn toHost(self: Self, allocator: std.mem.Allocator) ![]T {
            const out = try allocator.alloc(T, self.buffer.len);
            errdefer allocator.free(out);
            try self.buffer.copyToHost(out);
            return out;
        }

        /// Elementwise addition: self + other.
        pub fn add(self: Self, other: Self) !Self {
            if (!self.shape.eq(other.shape)) return error.InvalidValue;
            const total = self.shape.totalElements();

            const a_host = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(a_host);
            const b_host = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(b_host);
            const res_host = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(res_host);

            try self.buffer.copyToHost(a_host);
            try other.buffer.copyToHost(b_host);

            try elem_ops.add(T, res_host, a_host, b_host);
            return fromSlice(res_host, self.shape.dims[0..self.shape.ndim]);
        }

        /// Matrix multiplication: self @ other.
        pub fn matmul(self: Self, other: Self) !Self {
            if (self.shape.ndim != 2 or other.shape.ndim != 2) return error.InvalidValue;
            const m = self.shape.dims[0];
            const k = self.shape.dims[1];
            if (other.shape.dims[0] != k) return error.InvalidValue;
            const n = other.shape.dims[1];

            const a_host = try std.heap.page_allocator.alloc(T, m * k);
            defer std.heap.page_allocator.free(a_host);
            const b_host = try std.heap.page_allocator.alloc(T, k * n);
            defer std.heap.page_allocator.free(b_host);
            const c_host = try std.heap.page_allocator.alloc(T, m * n);
            defer std.heap.page_allocator.free(c_host);

            try self.buffer.copyToHost(a_host);
            try other.buffer.copyToHost(b_host);

            try matmul_op.matmul(T, c_host, a_host, b_host, m, n, k);
            return fromSlice(c_host, &.{ m, n });
        }

        /// Deinitializes the Tensor and frees associated memory.
        pub fn deinit(self: *Self) void {
            self.buffer.free();
        }
    };
}

test "Tensor zeros and toHost" {
    var t = try Tensor(f32).zeros(&.{ 2, 3 });
    defer t.deinit();

    const host_data = try t.toHost(std.testing.allocator);
    defer std.testing.allocator.free(host_data);

    try std.testing.expectEqual(@as(usize, 6), host_data.len);
    for (host_data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "Tensor add and matmul" {
    const a_slice = [_]f32{ 1, 2, 3, 4 };
    const b_slice = [_]f32{ 10, 20, 30, 40 };

    var a = try Tensor(f32).fromSlice(&a_slice, &.{ 2, 2 });
    defer a.deinit();
    var b = try Tensor(f32).fromSlice(&b_slice, &.{ 2, 2 });
    defer b.deinit();

    var res_add = try a.add(b);
    defer res_add.deinit();

    const add_data = try res_add.toHost(std.testing.allocator);
    defer std.testing.allocator.free(add_data);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, add_data);

    var res_mm = try a.matmul(b);
    defer res_mm.deinit();

    const mm_data = try res_mm.toHost(std.testing.allocator);
    defer std.testing.allocator.free(mm_data);
    // [1*10+2*30, 1*20+2*40] = [70, 100]
    // [3*10+4*30, 3*20+4*40] = [150, 220]
    try std.testing.expectEqualSlices(f32, &.{ 70, 100, 150, 220 }, mm_data);
}

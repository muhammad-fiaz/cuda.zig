//! High-level Tensor(T) abstraction — N-dimensional GPU/CPU tensor.
//!
//! Supports up to MAX_DIMS = 8 dimensions, row-major (C-contiguous) layout,
//! and seamless CPU/GPU routing through the dispatch layer.
//!
//! ## Quick Reference
//!
//! ### Lifecycle
//!   zeros, fromSlice, clone, deinit
//!
//! ### Shape
//!   reshape, flatten, squeeze, unsqueeze, transpose, T (2-D shorthand)
//!
//! ### Elementwise
//!   add, sub, mul, div, relu, neg, fill
//!   broadcastAdd, broadcastSub, broadcastMul, broadcastDiv
//!
//! ### Reductions (global)
//!   sum, mean, max, min
//!
//! ### Reductions (axis)
//!   sumAxis, maxAxis
//!
//! ### Matrix multiply
//!   matmul (2-D), batchedMatmul (3-D/4-D)
//!
//! ### Selection
//!   slice, concat
//!
//! ### Introspection
//!   numel, ndim, size, toHost, print

const std = @import("std");
const dev_buf = @import("../memory/device_memory.zig");
const shape_mod = @import("shape.zig");
const dtype_mod = @import("dtype.zig");
const elem_ops = @import("ops/elementwise.zig");
const matmul_op = @import("ops/matmul.zig");
const reduction_op = @import("ops/reduction.zig");
const transform_op = @import("ops/transform.zig");
const cpu_backend = @import("../fallback/cpu_backend.zig");
const err = @import("../core/error.zig");

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        buffer: dev_buf.DeviceBuffer(T),
        shape: shape_mod.Shape,

        // ------------------------------------------------------------------ //
        //  Lifecycle
        // ------------------------------------------------------------------ //

        /// Allocate a zero-filled Tensor with the given shape (1–8 dimensions).
        pub fn zeros(shape_slice: []const usize) !Self {
            const shp = try shape_mod.Shape.init(shape_slice);
            const total = shp.totalElements();
            var buf = try dev_buf.DeviceBuffer(T).alloc(total);
            errdefer buf.free();
            try buf.fill(0);
            return Self{ .buffer = buf, .shape = shp };
        }

        /// Create a Tensor from a flat host slice with the given shape.
        pub fn fromSlice(data: []const T, shape_slice: []const usize) !Self {
            const shp = try shape_mod.Shape.init(shape_slice);
            if (data.len != shp.totalElements()) return error.InvalidValue;
            var buf = try dev_buf.DeviceBuffer(T).alloc(data.len);
            errdefer buf.free();
            try buf.copyFromHost(data);
            return Self{ .buffer = buf, .shape = shp };
        }

        /// Deep-copy of this tensor.
        pub fn clone(self: Self) !Self {
            const total = self.shape.totalElements();
            const tmp = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(tmp);
            try self.buffer.copyToHost(tmp);
            return fromSlice(tmp, self.shape.dims[0..self.shape.ndim]);
        }

        /// Release device memory.
        pub fn deinit(self: *Self) void {
            self.buffer.free();
        }

        // ------------------------------------------------------------------ //
        //  Introspection
        // ------------------------------------------------------------------ //

        /// Total number of scalar elements.
        pub fn numel(self: Self) usize {
            return self.shape.totalElements();
        }

        /// Number of dimensions.
        pub fn ndim(self: Self) usize {
            return self.shape.ndim;
        }

        /// Size of a specific dimension.
        pub fn size(self: Self, dim: usize) usize {
            std.debug.assert(dim < self.shape.ndim);
            return self.shape.dims[dim];
        }

        /// Copy tensor data back to a newly allocated host slice.
        pub fn toHost(self: Self, allocator: std.mem.Allocator) ![]T {
            const out = try allocator.alloc(T, self.buffer.len);
            errdefer allocator.free(out);
            try self.buffer.copyToHost(out);
            return out;
        }

        /// Print shape and first min(8, numel) elements to stderr (debug helper).
        pub fn print(self: Self) void {
            const n = @min(self.numel(), 8);
            const tmp = std.heap.page_allocator.alloc(T, self.numel()) catch return;
            defer std.heap.page_allocator.free(tmp);
            self.buffer.copyToHost(tmp) catch return;
            std.debug.print("Tensor(shape={}, data=[", .{self.shape});
            for (tmp[0..n], 0..) |v, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{d}", .{v});
            }
            if (self.numel() > 8) std.debug.print(", ...]", .{}) else std.debug.print("]", .{});
            std.debug.print(")\n", .{});
        }

        // ------------------------------------------------------------------ //
        //  Shape transforms
        // ------------------------------------------------------------------ //

        /// Return a new Tensor view with a different shape (same total elements).
        pub fn reshape(self: Self, new_shape: []const usize) !Self {
            const new_shp = try shape_mod.Shape.init(new_shape);
            if (new_shp.totalElements() != self.shape.totalElements()) return error.InvalidValue;
            const total = self.numel();
            const tmp = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(tmp);
            const out_buf = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(out_buf);
            try self.buffer.copyToHost(tmp);
            try transform_op.reshape(T, out_buf, tmp, self.shape.dims[0..self.shape.ndim], new_shape);
            return fromSlice(out_buf, new_shape);
        }

        /// Collapse all dimensions into a single 1-D tensor.
        pub fn flatten(self: Self) !Self {
            return self.reshape(&.{self.numel()});
        }

        /// Remove all size-1 dimensions, or a specific one if `axis` is given.
        pub fn squeeze(self: Self, axis: ?usize) !Self {
            var new_dims: [shape_mod.MAX_DIMS]usize = undefined;
            var out_ndim: usize = 0;
            for (0..self.shape.ndim) |i| {
                if (axis) |ax| {
                    if (i == ax) {
                        if (self.shape.dims[i] != 1) return error.InvalidValue;
                        continue;
                    }
                } else {
                    if (self.shape.dims[i] == 1) continue;
                }
                new_dims[out_ndim] = self.shape.dims[i];
                out_ndim += 1;
            }
            if (out_ndim == 0) {
                new_dims[0] = 1;
                out_ndim = 1;
            }
            return self.reshape(new_dims[0..out_ndim]);
        }

        /// Insert a size-1 dimension at `axis`.
        pub fn unsqueeze(self: Self, axis: usize) !Self {
            if (self.shape.ndim + 1 > shape_mod.MAX_DIMS) return error.InvalidValue;
            var new_dims: [shape_mod.MAX_DIMS]usize = undefined;
            var j: usize = 0;
            for (0..self.shape.ndim + 1) |i| {
                if (i == axis) {
                    new_dims[i] = 1;
                } else {
                    new_dims[i] = self.shape.dims[j];
                    j += 1;
                }
            }
            return self.reshape(new_dims[0 .. self.shape.ndim + 1]);
        }

        /// Permute axes. `perm[i]` is the source axis for output axis `i`.
        pub fn transpose(self: Self, perm: []const usize) !Self {
            if (perm.len != self.shape.ndim) return error.InvalidValue;
            const total = self.numel();
            const src = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(src);
            const dst_data = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(dst_data);
            try self.buffer.copyToHost(src);
            try transform_op.transpose(T, dst_data, src, self.shape.dims[0..self.shape.ndim], perm);
            const new_shp = try self.shape.permute(perm);
            return fromSlice(dst_data, new_shp.dims[0..new_shp.ndim]);
        }

        /// 2-D matrix transpose shorthand (axes [1,0]).
        pub fn T2(self: Self) !Self {
            if (self.shape.ndim != 2) return error.InvalidValue;
            return self.transpose(&.{ 1, 0 });
        }

        // ------------------------------------------------------------------ //
        //  Elementwise (same-shape)
        // ------------------------------------------------------------------ //

        fn hostBuf(self: Self) ![]T {
            const buf = try std.heap.page_allocator.alloc(T, self.numel());
            try self.buffer.copyToHost(buf);
            return buf;
        }

        /// Element-wise addition (same shape). Returns new tensor.
        pub fn add(self: Self, other: Self) !Self {
            if (!self.shape.eq(other.shape)) return error.InvalidValue;
            const total = self.numel();
            const a = try self.hostBuf();
            defer std.heap.page_allocator.free(a);
            const b = try other.hostBuf();
            defer std.heap.page_allocator.free(b);
            const r = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(r);
            try elem_ops.add(T, r, a, b);
            return fromSlice(r, self.shape.dims[0..self.shape.ndim]);
        }

        /// Element-wise subtraction (same shape). Returns new tensor.
        pub fn sub(self: Self, other: Self) !Self {
            if (!self.shape.eq(other.shape)) return error.InvalidValue;
            const total = self.numel();
            const a = try self.hostBuf();
            defer std.heap.page_allocator.free(a);
            const b = try other.hostBuf();
            defer std.heap.page_allocator.free(b);
            const r = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(r);
            try elem_ops.sub(T, r, a, b);
            return fromSlice(r, self.shape.dims[0..self.shape.ndim]);
        }

        /// Element-wise multiplication (same shape). Returns new tensor.
        pub fn mul(self: Self, other: Self) !Self {
            if (!self.shape.eq(other.shape)) return error.InvalidValue;
            const total = self.numel();
            const a = try self.hostBuf();
            defer std.heap.page_allocator.free(a);
            const b = try other.hostBuf();
            defer std.heap.page_allocator.free(b);
            const r = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(r);
            try elem_ops.mul(T, r, a, b);
            return fromSlice(r, self.shape.dims[0..self.shape.ndim]);
        }

        /// Element-wise division (same shape). Returns new tensor.
        pub fn div(self: Self, other: Self) !Self {
            if (!self.shape.eq(other.shape)) return error.InvalidValue;
            const total = self.numel();
            const a = try self.hostBuf();
            defer std.heap.page_allocator.free(a);
            const b = try other.hostBuf();
            defer std.heap.page_allocator.free(b);
            const r = try std.heap.page_allocator.alloc(T, total);
            defer std.heap.page_allocator.free(r);
            try elem_ops.div(T, r, a, b);
            return fromSlice(r, self.shape.dims[0..self.shape.ndim]);
        }

        /// Element-wise ReLU: max(0, x). Returns new tensor.
        pub fn relu(self: Self) !Self {
            const src = try self.hostBuf();
            defer std.heap.page_allocator.free(src);
            const dst = try std.heap.page_allocator.alloc(T, self.numel());
            defer std.heap.page_allocator.free(dst);
            try elem_ops.relu(T, dst, src);
            return fromSlice(dst, self.shape.dims[0..self.shape.ndim]);
        }

        /// Element-wise negation: -x. Returns new tensor.
        pub fn neg(self: Self) !Self {
            const src = try self.hostBuf();
            defer std.heap.page_allocator.free(src);
            const dst = try std.heap.page_allocator.alloc(T, self.numel());
            defer std.heap.page_allocator.free(dst);
            try elem_ops.neg(T, dst, src);
            return fromSlice(dst, self.shape.dims[0..self.shape.ndim]);
        }

        /// Fill every element with `value`. Returns new tensor.
        pub fn fill(self: Self, value: T) !Self {
            const dst = try std.heap.page_allocator.alloc(T, self.numel());
            defer std.heap.page_allocator.free(dst);
            cpu_backend.fillScalar(T, dst, value);
            return fromSlice(dst, self.shape.dims[0..self.shape.ndim]);
        }

        // ------------------------------------------------------------------ //
        //  Broadcast elementwise
        // ------------------------------------------------------------------ //

        fn broadcastOp(self: Self, other: Self, comptime scalar_op: fn (T, T) T) !Self {
            const out_shp = try self.shape.broadcastWith(other.shape);
            const out_total = out_shp.totalElements();
            const a_total = self.numel();
            const b_total = other.numel();
            const a_host = try std.heap.page_allocator.alloc(T, a_total);
            defer std.heap.page_allocator.free(a_host);
            const b_host = try std.heap.page_allocator.alloc(T, b_total);
            defer std.heap.page_allocator.free(b_host);
            const out_host = try std.heap.page_allocator.alloc(T, out_total);
            defer std.heap.page_allocator.free(out_host);
            try self.buffer.copyToHost(a_host);
            try other.buffer.copyToHost(b_host);
            cpu_backend.broadcastElementwise(
                T,
                scalar_op,
                out_host,
                a_host,
                self.shape.dims[0..self.shape.ndim],
                b_host,
                other.shape.dims[0..other.shape.ndim],
                out_shp.dims[0..out_shp.ndim],
            );
            return fromSlice(out_host, out_shp.dims[0..out_shp.ndim]);
        }

        fn scalarAdd(a: T, b: T) T {
            return a + b;
        }
        fn scalarSub(a: T, b: T) T {
            return a - b;
        }
        fn scalarMul(a: T, b: T) T {
            return a * b;
        }
        fn scalarDiv(a: T, b: T) T {
            return a / b;
        }

        /// NumPy-style broadcast addition. Shapes are right-aligned.
        pub fn broadcastAdd(self: Self, other: Self) !Self {
            return self.broadcastOp(other, scalarAdd);
        }

        /// NumPy-style broadcast subtraction.
        pub fn broadcastSub(self: Self, other: Self) !Self {
            return self.broadcastOp(other, scalarSub);
        }

        /// NumPy-style broadcast multiplication.
        pub fn broadcastMul(self: Self, other: Self) !Self {
            return self.broadcastOp(other, scalarMul);
        }

        /// NumPy-style broadcast division.
        pub fn broadcastDiv(self: Self, other: Self) !Self {
            return self.broadcastOp(other, scalarDiv);
        }

        // ------------------------------------------------------------------ //
        //  Reductions (global)
        // ------------------------------------------------------------------ //

        /// Sum of all elements.
        pub fn sum(self: Self) !T {
            const tmp = try std.heap.page_allocator.alloc(T, self.numel());
            defer std.heap.page_allocator.free(tmp);
            try self.buffer.copyToHost(tmp);
            return reduction_op.sum(T, tmp);
        }

        /// Arithmetic mean.
        pub fn mean(self: Self) !T {
            const tmp = try std.heap.page_allocator.alloc(T, self.numel());
            defer std.heap.page_allocator.free(tmp);
            try self.buffer.copyToHost(tmp);
            return reduction_op.mean(T, tmp);
        }

        /// Maximum value.
        pub fn max(self: Self) !T {
            const tmp = try std.heap.page_allocator.alloc(T, self.numel());
            defer std.heap.page_allocator.free(tmp);
            try self.buffer.copyToHost(tmp);
            return reduction_op.max(T, tmp);
        }

        /// Minimum value.
        pub fn min(self: Self) !T {
            const tmp = try std.heap.page_allocator.alloc(T, self.numel());
            defer std.heap.page_allocator.free(tmp);
            try self.buffer.copyToHost(tmp);
            return reduction_op.min(T, tmp);
        }

        // ------------------------------------------------------------------ //
        //  Reductions (along axis)
        // ------------------------------------------------------------------ //

        fn axisReduceOp(
            self: Self,
            axis: usize,
            comptime scalar_op: fn (T, T) T,
            identity: T,
        ) !Self {
            if (axis >= self.shape.ndim) return error.InvalidValue;
            const src = try std.heap.page_allocator.alloc(T, self.numel());
            defer std.heap.page_allocator.free(src);
            try self.buffer.copyToHost(src);

            // Output shape: same as input with `axis` dimension removed
            var out_dims: [shape_mod.MAX_DIMS]usize = undefined;
            var out_ndim: usize = 0;
            for (0..self.shape.ndim) |i| {
                if (i != axis) {
                    out_dims[out_ndim] = self.shape.dims[i];
                    out_ndim += 1;
                }
            }
            if (out_ndim == 0) {
                out_dims[0] = 1;
                out_ndim = 1;
            }
            var out_total: usize = 1;
            for (out_dims[0..out_ndim]) |d| out_total *= d;

            const dst = try std.heap.page_allocator.alloc(T, out_total);
            defer std.heap.page_allocator.free(dst);

            cpu_backend.reduceAlongAxis(
                T,
                scalar_op,
                identity,
                dst,
                src,
                self.shape.dims[0..self.shape.ndim],
                axis,
            );
            return fromSlice(dst, out_dims[0..out_ndim]);
        }

        /// Sum along a single axis, returning a tensor with that dimension removed.
        pub fn sumAxis(self: Self, axis: usize) !Self {
            return self.axisReduceOp(axis, scalarAdd, 0);
        }

        /// Max along a single axis, returning a tensor with that dimension removed.
        pub fn maxAxis(self: Self, axis: usize) !Self {
            const identity = switch (@typeInfo(T)) {
                .float => -std.math.inf(T),
                .int => std.math.minInt(T),
                else => 0,
            };
            return self.axisReduceOp(axis, struct {
                fn f(a: T, b: T) T {
                    return @max(a, b);
                }
            }.f, identity);
        }

        // ------------------------------------------------------------------ //
        //  Matrix multiplication
        // ------------------------------------------------------------------ //

        /// 2-D matrix multiply: self [M,K] @ other [K,N] -> [M,N].
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

        /// Batched matrix multiply: self [B,M,K] @ other [B,K,N] -> [B,M,N].
        /// Also accepts [B,H,M,K] @ [B,H,K,N] -> [B,H,M,N] (4-D).
        pub fn batchedMatmul(self: Self, other: Self) !Self {
            const ndim_s = self.shape.ndim;
            if (ndim_s < 3 or ndim_s > 4) return error.InvalidValue;
            if (other.shape.ndim != ndim_s) return error.InvalidValue;

            // Flatten leading batch dims
            var batch: usize = 1;
            for (0..ndim_s - 2) |i| {
                if (self.shape.dims[i] != other.shape.dims[i]) return error.InvalidValue;
                batch *= self.shape.dims[i];
            }
            const m = self.shape.dims[ndim_s - 2];
            const k = self.shape.dims[ndim_s - 1];
            if (other.shape.dims[ndim_s - 2] != k) return error.InvalidValue;
            const n = other.shape.dims[ndim_s - 1];

            const a_host = try std.heap.page_allocator.alloc(T, self.numel());
            defer std.heap.page_allocator.free(a_host);
            const b_host = try std.heap.page_allocator.alloc(T, other.numel());
            defer std.heap.page_allocator.free(b_host);
            const c_host = try std.heap.page_allocator.alloc(T, batch * m * n);
            defer std.heap.page_allocator.free(c_host);
            try self.buffer.copyToHost(a_host);
            try other.buffer.copyToHost(b_host);
            try matmul_op.batchedMatmul(T, c_host, a_host, b_host, batch, m, n, k);

            // Build output shape: batch dims + [m, n]
            var out_dims: [shape_mod.MAX_DIMS]usize = undefined;
            for (0..ndim_s - 2) |i| out_dims[i] = self.shape.dims[i];
            out_dims[ndim_s - 2] = m;
            out_dims[ndim_s - 1] = n;
            return fromSlice(c_host, out_dims[0..ndim_s]);
        }

        // ------------------------------------------------------------------ //
        //  Selection / structural
        // ------------------------------------------------------------------ //

        /// Extract a sub-tensor. `starts` and `ends` are per-axis half-open
        /// ranges [start, end). Result shape is [ends[i]-starts[i]].
        pub fn slice(self: Self, starts: []const usize, ends: []const usize) !Self {
            if (starts.len != self.shape.ndim or ends.len != self.shape.ndim) return error.InvalidValue;
            var out_dims: [shape_mod.MAX_DIMS]usize = undefined;
            var out_total: usize = 1;
            for (0..self.shape.ndim) |i| {
                out_dims[i] = ends[i] - starts[i];
                out_total *= out_dims[i];
            }
            const src = try std.heap.page_allocator.alloc(T, self.numel());
            defer std.heap.page_allocator.free(src);
            const dst = try std.heap.page_allocator.alloc(T, out_total);
            defer std.heap.page_allocator.free(dst);
            try self.buffer.copyToHost(src);
            try transform_op.slice(T, dst, src, self.shape.dims[0..self.shape.ndim], starts, ends);
            return fromSlice(dst, out_dims[0..self.shape.ndim]);
        }

        /// Concatenate `other` along `axis`. Both tensors must have identical
        /// shapes on all other axes.
        pub fn concat(self: Self, other: Self, axis: usize) !Self {
            const ndim_s = self.shape.ndim;
            if (other.shape.ndim != ndim_s or axis >= ndim_s) return error.InvalidValue;
            var out_dims: [shape_mod.MAX_DIMS]usize = undefined;
            var out_total: usize = 1;
            for (0..ndim_s) |i| {
                out_dims[i] = if (i == axis)
                    self.shape.dims[i] + other.shape.dims[i]
                else
                    self.shape.dims[i];
                out_total *= out_dims[i];
            }
            const a_host = try std.heap.page_allocator.alloc(T, self.numel());
            defer std.heap.page_allocator.free(a_host);
            const b_host = try std.heap.page_allocator.alloc(T, other.numel());
            defer std.heap.page_allocator.free(b_host);
            const dst = try std.heap.page_allocator.alloc(T, out_total);
            defer std.heap.page_allocator.free(dst);
            try self.buffer.copyToHost(a_host);
            try other.buffer.copyToHost(b_host);
            try transform_op.concat(
                T,
                dst,
                a_host,
                self.shape.dims[0..ndim_s],
                b_host,
                other.shape.dims[0..ndim_s],
                axis,
            );
            return fromSlice(dst, out_dims[0..ndim_s]);
        }
    };
}

// ------------------------------------------------------------------ //
//  Tests
// ------------------------------------------------------------------ //

test "Tensor zeros and toHost" {
    var t = try Tensor(f32).zeros(&.{ 2, 3 });
    defer t.deinit();
    const data = try t.toHost(std.testing.allocator);
    defer std.testing.allocator.free(data);
    try std.testing.expectEqual(@as(usize, 6), data.len);
    for (data) |v| try std.testing.expectEqual(@as(f32, 0), v);
}

test "Tensor fromSlice and numel/ndim/size" {
    const raw = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var t = try Tensor(f32).fromSlice(&raw, &.{ 2, 3 });
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 6), t.numel());
    try std.testing.expectEqual(@as(usize, 2), t.ndim());
    try std.testing.expectEqual(@as(usize, 3), t.size(1));
}

test "Tensor add and sub" {
    var a = try Tensor(f32).fromSlice(&.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer a.deinit();
    var b = try Tensor(f32).fromSlice(&.{ 10, 20, 30, 40 }, &.{ 2, 2 });
    defer b.deinit();

    var r = try a.add(b);
    defer r.deinit();
    const d = try r.toHost(std.testing.allocator);
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, d);

    var s = try b.sub(a);
    defer s.deinit();
    const ds = try s.toHost(std.testing.allocator);
    defer std.testing.allocator.free(ds);
    try std.testing.expectEqualSlices(f32, &.{ 9, 18, 27, 36 }, ds);
}

test "Tensor mul and div" {
    var a = try Tensor(f32).fromSlice(&.{ 2, 4, 6, 8 }, &.{4});
    defer a.deinit();
    var b = try Tensor(f32).fromSlice(&.{ 1, 2, 3, 4 }, &.{4});
    defer b.deinit();

    var m = try a.mul(b);
    defer m.deinit();
    const dm = try m.toHost(std.testing.allocator);
    defer std.testing.allocator.free(dm);
    try std.testing.expectEqualSlices(f32, &.{ 2, 8, 18, 32 }, dm);

    var dv = try a.div(b);
    defer dv.deinit();
    const ddv = try dv.toHost(std.testing.allocator);
    defer std.testing.allocator.free(ddv);
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2 }, ddv);
}

test "Tensor relu and neg" {
    var a = try Tensor(f32).fromSlice(&.{ -1, 2, -3, 4 }, &.{4});
    defer a.deinit();

    var r = try a.relu();
    defer r.deinit();
    const dr = try r.toHost(std.testing.allocator);
    defer std.testing.allocator.free(dr);
    try std.testing.expectEqualSlices(f32, &.{ 0, 2, 0, 4 }, dr);

    var n = try a.neg();
    defer n.deinit();
    const dn = try n.toHost(std.testing.allocator);
    defer std.testing.allocator.free(dn);
    try std.testing.expectEqualSlices(f32, &.{ 1, -2, 3, -4 }, dn);
}

test "Tensor fill" {
    var t = try Tensor(f32).zeros(&.{ 2, 3 });
    defer t.deinit();
    var f = try t.fill(7.0);
    defer f.deinit();
    const d = try f.toHost(std.testing.allocator);
    defer std.testing.allocator.free(d);
    for (d) |v| try std.testing.expectEqual(@as(f32, 7.0), v);
}

test "Tensor reshape and flatten" {
    var t = try Tensor(f32).fromSlice(&.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer t.deinit();

    var r = try t.reshape(&.{ 3, 2 });
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 3), r.size(0));
    try std.testing.expectEqual(@as(usize, 2), r.size(1));

    var f = try t.flatten();
    defer f.deinit();
    try std.testing.expectEqual(@as(usize, 1), f.ndim());
    try std.testing.expectEqual(@as(usize, 6), f.size(0));
}

test "Tensor squeeze and unsqueeze" {
    var t = try Tensor(f32).fromSlice(&.{ 1, 2, 3 }, &.{ 1, 3, 1 });
    defer t.deinit();

    var sq = try t.squeeze(null);
    defer sq.deinit();
    try std.testing.expectEqual(@as(usize, 1), sq.ndim());
    try std.testing.expectEqual(@as(usize, 3), sq.size(0));

    var us = try sq.unsqueeze(0);
    defer us.deinit();
    try std.testing.expectEqual(@as(usize, 2), us.ndim());
    try std.testing.expectEqual(@as(usize, 1), us.size(0));
}

test "Tensor transpose 2D" {
    var t = try Tensor(f32).fromSlice(&.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer t.deinit();
    var tp = try t.T2();
    defer tp.deinit();
    try std.testing.expectEqual(@as(usize, 3), tp.size(0));
    try std.testing.expectEqual(@as(usize, 2), tp.size(1));
    const d = try tp.toHost(std.testing.allocator);
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, d);
}

test "Tensor transpose 3D" {
    // [2,3,4] permuted to [4,2,3]
    var raw: [24]f32 = undefined;
    for (&raw, 0..) |*v, i| v.* = @floatFromInt(i);
    var t = try Tensor(f32).fromSlice(&raw, &.{ 2, 3, 4 });
    defer t.deinit();
    var tp = try t.transpose(&.{ 2, 0, 1 });
    defer tp.deinit();
    try std.testing.expectEqual(@as(usize, 4), tp.size(0));
    try std.testing.expectEqual(@as(usize, 2), tp.size(1));
    try std.testing.expectEqual(@as(usize, 3), tp.size(2));
    try std.testing.expectEqual(@as(usize, 24), tp.numel());
}

test "Tensor reductions" {
    var t = try Tensor(f32).fromSlice(&.{ 1, 2, 3, 4 }, &.{4});
    defer t.deinit();
    try std.testing.expectEqual(@as(f32, 10), try t.sum());
    try std.testing.expectEqual(@as(f32, 2.5), try t.mean());
    try std.testing.expectEqual(@as(f32, 4), try t.max());
    try std.testing.expectEqual(@as(f32, 1), try t.min());
}

test "Tensor sumAxis" {
    // [2,3] summed along axis 0 -> [3]
    var t = try Tensor(f32).fromSlice(&.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer t.deinit();
    var s = try t.sumAxis(0);
    defer s.deinit();
    const d = try s.toHost(std.testing.allocator);
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, d);
}

test "Tensor broadcastAdd" {
    // [2,3] + [3] -> [2,3]
    var a = try Tensor(f32).fromSlice(&.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try Tensor(f32).fromSlice(&.{ 10, 20, 30 }, &.{3});
    defer b.deinit();
    var r = try a.broadcastAdd(b);
    defer r.deinit();
    const d = try r.toHost(std.testing.allocator);
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, d);
}

test "Tensor matmul 2D" {
    var a = try Tensor(f32).fromSlice(&.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer a.deinit();
    var b = try Tensor(f32).fromSlice(&.{ 10, 20, 30, 40 }, &.{ 2, 2 });
    defer b.deinit();
    var c = try a.matmul(b);
    defer c.deinit();
    const d = try c.toHost(std.testing.allocator);
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualSlices(f32, &.{ 70, 100, 150, 220 }, d);
}

test "Tensor batchedMatmul 3D" {
    // batch=2, m=2, k=2, n=2
    var a = try Tensor(f32).fromSlice(&.{ 1, 0, 0, 1, 2, 0, 0, 2 }, &.{ 2, 2, 2 });
    defer a.deinit();
    var b = try Tensor(f32).fromSlice(&.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ 2, 2, 2 });
    defer b.deinit();
    var c = try a.batchedMatmul(b);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 3), c.ndim());
    try std.testing.expectEqual(@as(usize, 8), c.numel());
}

test "Tensor slice" {
    var t = try Tensor(f32).fromSlice(&.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }, &.{ 3, 4 });
    defer t.deinit();
    var s = try t.slice(&.{ 0, 1 }, &.{ 2, 3 });
    defer s.deinit();
    const d = try s.toHost(std.testing.allocator);
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 5, 6 }, d);
}

test "Tensor concat axis=0 and axis=1" {
    var a = try Tensor(f32).fromSlice(&.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer a.deinit();
    var b = try Tensor(f32).fromSlice(&.{ 5, 6, 7, 8 }, &.{ 2, 2 });
    defer b.deinit();

    var c0 = try a.concat(b, 0);
    defer c0.deinit();
    try std.testing.expectEqual(@as(usize, 4), c0.size(0));

    var c1 = try a.concat(b, 1);
    defer c1.deinit();
    try std.testing.expectEqual(@as(usize, 4), c1.size(1));
    const d = try c1.toHost(std.testing.allocator);
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 5, 6, 3, 4, 7, 8 }, d);
}

test "Tensor clone" {
    var a = try Tensor(f32).fromSlice(&.{ 1, 2, 3 }, &.{3});
    defer a.deinit();
    var b = try a.clone();
    defer b.deinit();
    const d = try b.toHost(std.testing.allocator);
    defer std.testing.allocator.free(d);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, d);
}

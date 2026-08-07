//! N-dimensional tensor shape, strides, and index utilities.
//!
//! Supports up to MAX_DIMS = 8 dimensions. Strides are row-major (C order)
//! unless explicitly constructed otherwise.

const std = @import("std");
const err = @import("../core/error.zig");

pub const MAX_DIMS = 8;

pub const Shape = struct {
    dims: [MAX_DIMS]usize,
    ndim: usize,

    /// Build a Shape from a slice of dimension sizes.
    pub fn init(shape_slice: []const usize) !Shape {
        if (shape_slice.len == 0 or shape_slice.len > MAX_DIMS) return error.InvalidValue;
        var s = Shape{ .dims = [_]usize{0} ** MAX_DIMS, .ndim = shape_slice.len };
        @memcpy(s.dims[0..shape_slice.len], shape_slice);
        return s;
    }

    /// Total number of scalar elements (product of all dims).
    pub fn totalElements(self: Shape) usize {
        if (self.ndim == 0) return 0;
        var count: usize = 1;
        for (self.dims[0..self.ndim]) |d| count *= d;
        return count;
    }

    /// Fill `strides_out` with row-major (C-contiguous) strides.
    /// `strides_out` must have length == self.ndim.
    pub fn computeContiguousStrides(self: Shape, strides_out: []usize) void {
        std.debug.assert(strides_out.len == self.ndim);
        if (self.ndim == 0) return;
        var st: usize = 1;
        var i: usize = self.ndim;
        while (i > 0) {
            i -= 1;
            strides_out[i] = st;
            st *= self.dims[i];
        }
    }

    /// Shape equality.
    pub fn eq(self: Shape, other: Shape) bool {
        if (self.ndim != other.ndim) return false;
        return std.mem.eql(usize, self.dims[0..self.ndim], other.dims[0..other.ndim]);
    }

    /// Returns the total element count if `other` can be broadcast into `self`,
    /// or an error if the shapes are incompatible.
    /// Broadcasting follows NumPy rules: dimensions are compared from the trailing end;
    /// a dimension of 1 is expandable to match the corresponding size.
    pub fn broadcastWith(self: Shape, other: Shape) !Shape {
        const out_ndim = @max(self.ndim, other.ndim);
        var out = Shape{ .dims = [_]usize{0} ** MAX_DIMS, .ndim = out_ndim };
        var i: usize = 0;
        while (i < out_ndim) : (i += 1) {
            const ri = out_ndim - 1 - i; // index from right
            const a = if (i < self.ndim) self.dims[self.ndim - 1 - i] else 1;
            const b = if (i < other.ndim) other.dims[other.ndim - 1 - i] else 1;
            if (a == b) {
                out.dims[ri] = a;
            } else if (a == 1) {
                out.dims[ri] = b;
            } else if (b == 1) {
                out.dims[ri] = a;
            } else {
                return error.InvalidValue; // incompatible
            }
        }
        return out;
    }

    /// Return a new Shape with the axes permuted by `perm`.
    /// `perm` must be a permutation of [0, ndim).
    pub fn permute(self: Shape, perm: []const usize) !Shape {
        if (perm.len != self.ndim) return error.InvalidValue;
        var out = Shape{ .dims = [_]usize{0} ** MAX_DIMS, .ndim = self.ndim };
        for (perm, 0..) |axis, i| {
            if (axis >= self.ndim) return error.InvalidValue;
            out.dims[i] = self.dims[axis];
        }
        return out;
    }

    /// Format the shape for debug printing, e.g. "[2, 3, 4]".
    pub fn format(
        self: Shape,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll("[");
        for (self.dims[0..self.ndim], 0..) |d, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{d});
        }
        try writer.writeAll("]");
    }
};

test "Shape totalElements and strides" {
    const s = try Shape.init(&.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 24), s.totalElements());

    var strides: [3]usize = undefined;
    s.computeContiguousStrides(&strides);
    try std.testing.expectEqualSlices(usize, &.{ 12, 4, 1 }, &strides);
}

test "Shape broadcastWith" {
    const a = try Shape.init(&.{ 3, 1, 5 });
    const b = try Shape.init(&.{ 1, 4, 5 });
    const out = try a.broadcastWith(b);
    try std.testing.expectEqual(@as(usize, 3), out.ndim);
    try std.testing.expectEqualSlices(usize, &.{ 3, 4, 5 }, out.dims[0..3]);
}

test "Shape permute" {
    const s = try Shape.init(&.{ 2, 3, 4 });
    const p = try s.permute(&.{ 2, 0, 1 });
    try std.testing.expectEqualSlices(usize, &.{ 4, 2, 3 }, p.dims[0..3]);
}

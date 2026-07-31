//! Tensor Shape and Strides utilities.

const std = @import("std");
const err = @import("../core/error.zig");

pub const MAX_DIMS = 8;

pub const Shape = struct {
    dims: [MAX_DIMS]usize,
    ndim: usize,

    pub fn init(shape_slice: []const usize) !Shape {
        if (shape_slice.len > MAX_DIMS) return error.InvalidValue;
        var s = Shape{
            .dims = undefined,
            .ndim = shape_slice.len,
        };
        @memcpy(s.dims[0..shape_slice.len], shape_slice);
        return s;
    }

    pub fn totalElements(self: Shape) usize {
        if (self.ndim == 0) return 0;
        var count: usize = 1;
        for (self.dims[0..self.ndim]) |d| {
            count *= d;
        }
        return count;
    }

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

    pub fn eq(self: Shape, other: Shape) bool {
        if (self.ndim != other.ndim) return false;
        return std.mem.eql(usize, self.dims[0..self.ndim], other.dims[0..other.ndim]);
    }
};

test "Shape totalElements and strides" {
    const s = try Shape.init(&.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 24), s.totalElements());

    var strides: [3]usize = undefined;
    s.computeContiguousStrides(&strides);
    try std.testing.expectEqualSlices(usize, &.{ 12, 4, 1 }, &strides);
}

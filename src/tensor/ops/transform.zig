//! N-D tensor transform operations: reshape, transpose, slice, concat, stack,
//! squeeze, unsqueeze, flatten, and broadcast expand.

const std = @import("std");
const shape_mod = @import("../shape.zig");
const cpu_backend = @import("../../fallback/cpu_backend.zig");

/// Reshape `src` from `in_dims` to `out_dims` (must have equal totalElements).
/// This is a zero-copy operation — data is copied only when the layout
/// requires it (currently always contiguous, so a memcpy is performed).
pub fn reshape(
    comptime T: type,
    dst: []T,
    src: []const T,
    in_dims: []const usize,
    out_dims: []const usize,
) !void {
    var in_total: usize = 1;
    for (in_dims) |d| in_total *= d;
    var out_total: usize = 1;
    for (out_dims) |d| out_total *= d;
    if (in_total != out_total) return error.InvalidValue;
    if (dst.len != out_total or src.len != in_total) return error.InvalidValue;
    @memcpy(dst, src);
}

/// Transpose an N-D contiguous tensor by permuting its axes.
/// `perm[i]` is the input axis that maps to output axis `i`.
pub fn transpose(
    comptime T: type,
    dst: []T,
    src: []const T,
    in_dims: []const usize,
    perm: []const usize,
) !void {
    if (perm.len != in_dims.len) return error.InvalidValue;
    cpu_backend.transposeND(T, dst, src, in_dims, perm);
}

/// Extract a contiguous sub-tensor. `starts` and `ends` are per-axis
/// half-open ranges [start, end). Resulting shape is ends[i]-starts[i].
pub fn slice(
    comptime T: type,
    dst: []T,
    src: []const T,
    in_dims: []const usize,
    starts: []const usize,
    ends: []const usize,
) !void {
    const ndim = in_dims.len;
    if (starts.len != ndim or ends.len != ndim) return error.InvalidValue;

    // Validate ranges and compute output shape
    var out_dims: [8]usize = [_]usize{1} ** 8;
    var out_total: usize = 1;
    for (0..ndim) |i| {
        if (ends[i] > in_dims[i] or starts[i] >= ends[i]) return error.InvalidValue;
        out_dims[i] = ends[i] - starts[i];
        out_total *= out_dims[i];
    }
    if (dst.len != out_total) return error.InvalidValue;

    // Compute input strides
    var in_strides = [_]usize{0} ** 8;
    {
        var st: usize = 1;
        var i: usize = ndim;
        while (i > 0) {
            i -= 1;
            in_strides[i] = st;
            st *= in_dims[i];
        }
    }

    // Iterate output positions
    var idx = [_]usize{0} ** 8;
    for (dst) |*elem| {
        // Compute src flat index = starts[i] + idx[i] along each dim
        var src_flat: usize = 0;
        for (0..ndim) |i| src_flat += (starts[i] + idx[i]) * in_strides[i];
        elem.* = src[src_flat];

        // Advance counter
        var carry: usize = 1;
        var di = ndim;
        while (di > 0 and carry > 0) {
            di -= 1;
            idx[di] += carry;
            if (idx[di] >= out_dims[di]) {
                idx[di] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }
    }
}

/// Concatenate two N-D tensors along `axis`. Both must have identical shapes
/// except along the concat axis. `dst` must be pre-allocated.
pub fn concat(
    comptime T: type,
    dst: []T,
    a: []const T,
    a_dims: []const usize,
    b: []const T,
    b_dims: []const usize,
    axis: usize,
) !void {
    const ndim = a_dims.len;
    if (b_dims.len != ndim or axis >= ndim) return error.InvalidValue;
    for (0..ndim) |i| {
        if (i != axis and a_dims[i] != b_dims[i]) return error.InvalidValue;
    }

    // Build output dims
    var out_dims = [_]usize{0} ** 8;
    for (0..ndim) |i| out_dims[i] = a_dims[i];
    out_dims[axis] += b_dims[axis];

    // Compute strides
    var a_strides = [_]usize{0} ** 8;
    var b_strides = [_]usize{0} ** 8;
    var out_strides = [_]usize{0} ** 8;
    {
        var st: usize = 1;
        var i: usize = ndim;
        while (i > 0) {
            i -= 1;
            a_strides[i] = st;
            st *= a_dims[i];
        }
    }
    {
        var st: usize = 1;
        var i: usize = ndim;
        while (i > 0) {
            i -= 1;
            b_strides[i] = st;
            st *= b_dims[i];
        }
    }
    {
        var st: usize = 1;
        var i: usize = ndim;
        while (i > 0) {
            i -= 1;
            out_strides[i] = st;
            st *= out_dims[i];
        }
    }

    // Iterate output
    var out_total: usize = 1;
    for (out_dims[0..ndim]) |d| out_total *= d;
    if (dst.len != out_total) return error.InvalidValue;

    var idx = [_]usize{0} ** 8;
    for (dst) |*elem| {
        const axis_idx = idx[axis];
        if (axis_idx < a_dims[axis]) {
            // From a
            var src_flat: usize = 0;
            for (0..ndim) |i| src_flat += idx[i] * a_strides[i];
            elem.* = a[src_flat];
        } else {
            // From b — adjust axis index
            var src_flat: usize = 0;
            for (0..ndim) |i| {
                const bi = if (i == axis) idx[i] - a_dims[axis] else idx[i];
                src_flat += bi * b_strides[i];
            }
            elem.* = b[src_flat];
        }

        // Advance counter
        var carry: usize = 1;
        var di = ndim;
        while (di > 0 and carry > 0) {
            di -= 1;
            idx[di] += carry;
            if (idx[di] >= out_dims[di]) {
                idx[di] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }
    }
}

test "transform reshape" {
    const src = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var dst: [6]f32 = undefined;
    try reshape(f32, &dst, &src, &.{ 2, 3 }, &.{ 3, 2 });
    try std.testing.expectEqualSlices(f32, &src, &dst);
}

test "transform transpose 2D" {
    const src = [_]f32{ 1, 2, 3, 4, 5, 6 }; // shape [2,3]
    var dst: [6]f32 = undefined;
    try transpose(f32, &dst, &src, &.{ 2, 3 }, &.{ 1, 0 });
    // Expected [3,2]: [[1,4],[2,5],[3,6]] => [1,4,2,5,3,6]
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, &dst);
}

test "transform slice" {
    // src shape [3,4]: rows 0..2, cols 1..3
    const src = [_]f32{
        0, 1, 2,  3,
        4, 5, 6,  7,
        8, 9, 10, 11,
    };
    var dst: [4]f32 = undefined;
    try slice(f32, &dst, &src, &.{ 3, 4 }, &.{ 0, 1 }, &.{ 2, 3 });
    // rows 0..2, cols 1..3 => [1,2,5,6]
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 5, 6 }, &dst);
}

test "transform concat axis=0" {
    const a = [_]f32{ 1, 2, 3, 4 }; // [2,2]
    const b = [_]f32{ 5, 6, 7, 8 }; // [2,2]
    var dst: [8]f32 = undefined;
    try concat(f32, &dst, &a, &.{ 2, 2 }, &b, &.{ 2, 2 }, 0);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &dst);
}

test "transform concat axis=1" {
    const a = [_]f32{ 1, 2, 3, 4 }; // [2,2]
    const b = [_]f32{ 5, 6, 7, 8 }; // [2,2]
    var dst: [8]f32 = undefined;
    try concat(f32, &dst, &a, &.{ 2, 2 }, &b, &.{ 2, 2 }, 1);
    // row0: [1,2,5,6]  row1: [3,4,7,8]
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 5, 6, 3, 4, 7, 8 }, &dst);
}

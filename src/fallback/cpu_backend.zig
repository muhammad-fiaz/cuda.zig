//! CPU Backend implementations for numerical tensor ops (elementwise, reduction, matmul).
//!
//! Provides pure Zig CPU implementations matching the GPU op signatures.
//! Runs unconditionally when no CUDA GPU or driver is available.

const std = @import("std");

pub fn elementwiseAdd(comptime T: type, dst: []T, a: []const T, b: []const T) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, 0..) |*out, i| out.* = a[i] + b[i];
}

pub fn elementwiseSub(comptime T: type, dst: []T, a: []const T, b: []const T) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, 0..) |*out, i| out.* = a[i] - b[i];
}

pub fn elementwiseMul(comptime T: type, dst: []T, a: []const T, b: []const T) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, 0..) |*out, i| out.* = a[i] * b[i];
}

pub fn elementwiseDiv(comptime T: type, dst: []T, a: []const T, b: []const T) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, 0..) |*out, i| out.* = a[i] / b[i];
}

pub fn elementwiseRelu(comptime T: type, dst: []T, a: []const T) void {
    std.debug.assert(dst.len == a.len);
    const zero: T = 0;
    for (dst, 0..) |*out, i| out.* = @max(zero, a[i]);
}

pub fn elementwiseNeg(comptime T: type, dst: []T, a: []const T) void {
    std.debug.assert(dst.len == a.len);
    for (dst, 0..) |*out, i| out.* = -a[i];
}

pub fn fillScalar(comptime T: type, dst: []T, value: T) void {
    for (dst) |*out| out.* = value;
}

pub fn reduceSum(comptime T: type, data: []const T) T {
    var acc: T = 0;
    for (data) |v| acc += v;
    return acc;
}

pub fn reduceMean(comptime T: type, data: []const T) T {
    if (data.len == 0) return 0;
    const s = reduceSum(T, data);
    return switch (@typeInfo(T)) {
        .float => s / @as(T, @floatFromInt(data.len)),
        .int => @divTrunc(s, @as(T, @intCast(data.len))),
        else => s,
    };
}

pub fn reduceMax(comptime T: type, data: []const T) T {
    std.debug.assert(data.len > 0);
    var m = data[0];
    for (data[1..]) |v| if (v > m) {
        m = v;
    };
    return m;
}

pub fn reduceMin(comptime T: type, data: []const T) T {
    std.debug.assert(data.len > 0);
    var m = data[0];
    for (data[1..]) |v| if (v < m) {
        m = v;
    };
    return m;
}

pub fn matmulNaive(
    comptime T: type,
    c: []T,
    a: []const T,
    b: []const T,
    m: usize,
    n: usize,
    k: usize,
) void {
    std.debug.assert(c.len == m * n);
    std.debug.assert(a.len == m * k);
    std.debug.assert(b.len == k * n);

    @memset(c, 0);
    for (0..m) |i| {
        for (0..k) |p| {
            const a_val = a[i * k + p];
            for (0..n) |j| {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
}

/// Batched matmul: A[batch,M,K] @ B[batch,K,N] -> C[batch,M,N].
pub fn batchedMatmulNaive(
    comptime T: type,
    c: []T,
    a: []const T,
    b: []const T,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) void {
    std.debug.assert(a.len == batch * m * k);
    std.debug.assert(b.len == batch * k * n);
    std.debug.assert(c.len == batch * m * n);
    for (0..batch) |bi| {
        const a_off = bi * m * k;
        const b_off = bi * k * n;
        const c_off = bi * m * n;
        matmulNaive(
            T,
            c[c_off .. c_off + m * n],
            a[a_off .. a_off + m * k],
            b[b_off .. b_off + k * n],
            m,
            n,
            k,
        );
    }
}

/// General N-D broadcast elementwise op using stride-based index arithmetic.
/// `op` is called as op(a_elem, b_elem) -> T for each output position.
/// Shapes are right-aligned (NumPy broadcast semantics).
pub fn broadcastElementwise(
    comptime T: type,
    comptime op: fn (T, T) T,
    out: []T,
    a: []const T,
    a_dims: []const usize,
    b: []const T,
    b_dims: []const usize,
    out_dims: []const usize,
) void {
    const ndim = out_dims.len;

    // Compute row-major strides for each operand's own dims
    var a_strides = [_]usize{0} ** 8;
    var b_strides = [_]usize{0} ** 8;
    var out_strides = [_]usize{0} ** 8;

    {
        var st: usize = 1;
        var i: usize = a_dims.len;
        while (i > 0) {
            i -= 1;
            a_strides[i] = st;
            st *= a_dims[i];
        }
    }
    {
        var st: usize = 1;
        var i: usize = b_dims.len;
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

    // Mixed-radix counter over output shape
    var idx = [_]usize{0} ** 8;
    for (out) |*elem| {
        // Compute flat src indices with broadcast (stride = 0 on size-1 dims)
        var a_flat: usize = 0;
        var b_flat: usize = 0;

        for (0..ndim) |di| {
            // Right-align a dims into out dims
            const a_offset = ndim - a_dims.len;
            const b_offset = ndim - b_dims.len;

            if (di >= a_offset) {
                const ai = di - a_offset;
                a_flat += (if (a_dims[ai] == 1) 0 else idx[di]) * a_strides[ai];
            }
            if (di >= b_offset) {
                const bi = di - b_offset;
                b_flat += (if (b_dims[bi] == 1) 0 else idx[di]) * b_strides[bi];
            }
        }
        elem.* = op(a[a_flat], b[b_flat]);

        // Advance mixed-radix counter
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

/// Transpose an N-D tensor by permuting axes.
/// `src` has row-major layout with shape `in_dims`.
/// `dst` receives the permuted result (shape = in_dims permuted by `perm`).
pub fn transposeND(
    comptime T: type,
    dst: []T,
    src: []const T,
    in_dims: []const usize,
    perm: []const usize,
) void {
    const ndim = in_dims.len;
    std.debug.assert(perm.len == ndim);

    // Input strides (row-major)
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

    // Output dims and strides
    var out_dims = [_]usize{0} ** 8;
    for (perm, 0..) |axis, i| out_dims[i] = in_dims[axis];
    // out_strides not needed — we iterate via the mixed-radix counter

    // Iterate output positions
    var idx = [_]usize{0} ** 8;
    for (dst) |*elem| {
        // Map output idx -> source flat index via perm
        var src_flat: usize = 0;
        for (0..ndim) |i| {
            src_flat += idx[i] * in_strides[perm[i]];
        }
        elem.* = src[src_flat];

        // Advance over output dims
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

test "CPU backend elementwise ops" {
    const a = [_]f32{ 1.0, 2.0, 3.0, -4.0 };
    const b = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    var dst = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    elementwiseAdd(f32, &dst, &a, &b);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 11.0, 22.0, 33.0, 36.0 }, &dst);

    elementwiseSub(f32, &dst, &b, &a);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9.0, 18.0, 27.0, 44.0 }, &dst);

    elementwiseRelu(f32, &dst, &a);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 0.0 }, &dst);
}

test "CPU backend matmul" {
    // 2x3 * 3x2 -> 2x2
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 1, 2, 3 };
    var c = [_]f32{ 0, 0, 0, 0 };
    matmulNaive(f32, &c, &a, &b, 2, 2, 3);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 31, 19, 85, 55 }, &c);
}

/// Reduce along a single axis. `op` is a binary accumulator, `identity` is the starting value.
/// Output shape is input shape with axis removed; dst must be pre-allocated and pre-filled with `identity`.
pub fn reduceAlongAxis(
    comptime T: type,
    comptime op: fn (T, T) T,
    identity: T,
    dst: []T,
    src: []const T,
    in_dims: []const usize,
    axis: usize,
) void {
    const ndim = in_dims.len;
    std.debug.assert(axis < ndim);

    // Input strides
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

    // Output strides (input dims with axis removed)
    var out_dims = [_]usize{0} ** 8;
    var out_ndim: usize = 0;
    for (0..ndim) |i| {
        if (i != axis) {
            out_dims[out_ndim] = in_dims[i];
            out_ndim += 1;
        }
    }
    if (out_ndim == 0) { out_ndim = 1; out_dims[0] = 1; }
    var out_strides2 = [_]usize{0} ** 8;
    {
        var st: usize = 1;
        var i: usize = out_ndim;
        while (i > 0) {
            i -= 1;
            out_strides2[i] = st;
            st *= out_dims[i];
        }
    }

    // Initialise dst with identity
    @memset(dst, identity);

    // Iterate over all input elements
    var in_total: usize = 1;
    for (in_dims) |d| in_total *= d;

    var idx = [_]usize{0} ** 8;
    for (0..in_total) |_| {
        // Compute dst flat index by projecting idx onto output (skip axis dim)
        var dst_flat: usize = 0;
        var out_d: usize = 0;
        for (0..ndim) |i| {
            if (i != axis) {
                dst_flat += idx[i] * out_strides2[out_d];
                out_d += 1;
            }
        }
        // Src flat index
        var src_flat: usize = 0;
        for (0..ndim) |i| src_flat += idx[i] * in_strides[i];
        dst[dst_flat] = op(dst[dst_flat], src[src_flat]);

        // Advance counter
        var carry: usize = 1;
        var di = ndim;
        while (di > 0 and carry > 0) {
            di -= 1;
            idx[di] += carry;
            if (idx[di] >= in_dims[di]) { idx[di] = 0; carry = 1; } else { carry = 0; }
        }
    }
}

test "CPU backend transposeND" {
    // Transpose 2x3 -> 3x2: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    const src = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var dst: [6]f32 = undefined;
    transposeND(f32, &dst, &src, &.{ 2, 3 }, &.{ 1, 0 });
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 4, 2, 5, 3, 6 }, &dst);
}

const std = @import("std");
const cuda = @import("cuda");

fn printSlice(label: []const u8, d: []const f32) void {
    std.debug.print("  {s}: [", .{label});
    for (d, 0..) |v, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.1}", .{v});
    }
    std.debug.print("]\n", .{});
}

pub fn main() !void {
    std.debug.print("=== cuda.zig N-Dimensional Tensor Operations Example ===\n\n", .{});

    const allocator = std.heap.page_allocator;

    // ------------------------------------------------------------------ //
    //  1. Shape inspection
    // ------------------------------------------------------------------ //
    std.debug.print("--- 1. Shape & Introspection ---\n", .{});
    {
        var t = try cuda.Tensor(f32).zeros(&.{ 3, 4, 5 });
        defer t.deinit();
        std.debug.print("  ndim:  {d}\n", .{t.ndim()});
        std.debug.print("  numel: {d}\n", .{t.numel()});
        std.debug.print("  dims:  [{d}, {d}, {d}]\n\n", .{ t.size(0), t.size(1), t.size(2) });
    }

    // ------------------------------------------------------------------ //
    //  2. From slice — any rank
    // ------------------------------------------------------------------ //
    std.debug.print("--- 2. From slice (1-D through 4-D) ---\n", .{});
    {
        var raw: [24]f32 = undefined;
        for (&raw, 0..) |*v, i| v.* = @floatFromInt(i + 1);

        var t1 = try cuda.Tensor(f32).fromSlice(&raw, &.{24});
        defer t1.deinit();
        var t2 = try cuda.Tensor(f32).fromSlice(&raw, &.{ 4, 6 });
        defer t2.deinit();
        var t3 = try cuda.Tensor(f32).fromSlice(&raw, &.{ 2, 3, 4 });
        defer t3.deinit();
        var t4 = try cuda.Tensor(f32).fromSlice(&raw, &.{ 2, 2, 2, 3 });
        defer t4.deinit();

        std.debug.print("  1-D [{d}]\n", .{t1.size(0)});
        std.debug.print("  2-D [{d},{d}]\n", .{ t2.size(0), t2.size(1) });
        std.debug.print("  3-D [{d},{d},{d}]\n", .{ t3.size(0), t3.size(1), t3.size(2) });
        std.debug.print("  4-D [{d},{d},{d},{d}]\n\n", .{ t4.size(0), t4.size(1), t4.size(2), t4.size(3) });
    }

    // ------------------------------------------------------------------ //
    //  3. Reshape / flatten / squeeze / unsqueeze
    // ------------------------------------------------------------------ //
    std.debug.print("--- 3. Shape Transforms ---\n", .{});
    {
        const raw = [_]f32{ 1, 2, 3, 4, 5, 6 };
        var t = try cuda.Tensor(f32).fromSlice(&raw, &.{ 2, 3 });
        defer t.deinit();

        var reshaped = try t.reshape(&.{ 3, 2 });
        defer reshaped.deinit();
        std.debug.print("  reshape [2,3] -> [{d},{d}]\n", .{ reshaped.size(0), reshaped.size(1) });

        var flat = try t.flatten();
        defer flat.deinit();
        std.debug.print("  flatten [2,3] -> [{d}]\n", .{flat.size(0)});

        var us = try flat.unsqueeze(0);
        defer us.deinit();
        std.debug.print("  unsqueeze(0) [{d}] -> [{d},{d}]\n", .{ flat.size(0), us.size(0), us.size(1) });

        var sq = try us.squeeze(null);
        defer sq.deinit();
        std.debug.print("  squeeze all -> [{d}]\n\n", .{sq.size(0)});
    }

    // ------------------------------------------------------------------ //
    //  4. Transpose
    // ------------------------------------------------------------------ //
    std.debug.print("--- 4. Transpose ---\n", .{});
    {
        const raw = [_]f32{ 1, 2, 3, 4, 5, 6 };
        var mat = try cuda.Tensor(f32).fromSlice(&raw, &.{ 2, 3 });
        defer mat.deinit();

        var tp = try mat.T2();
        defer tp.deinit();
        std.debug.print("  T2: [2,3] -> [{d},{d}]\n", .{ tp.size(0), tp.size(1) });
        const d = try tp.toHost(allocator);
        defer allocator.free(d);
        printSlice("  transposed values", d);
        std.debug.print("\n", .{});
    }

    // ------------------------------------------------------------------ //
    //  5. Elementwise ops
    // ------------------------------------------------------------------ //
    std.debug.print("--- 5. Elementwise Ops ---\n", .{});
    {
        var a = try cuda.Tensor(f32).fromSlice(&.{ 1, -2, 3, -4, 5, 6 }, &.{ 2, 3 });
        defer a.deinit();
        var b = try cuda.Tensor(f32).fromSlice(&.{ 10, 10, 10, 10, 10, 10 }, &.{ 2, 3 });
        defer b.deinit();

        var added = try a.add(b);
        defer added.deinit();
        var subbed = try b.sub(a);
        defer subbed.deinit();
        var mulled = try a.mul(b);
        defer mulled.deinit();
        var relued = try a.relu();
        defer relued.deinit();
        var negged = try a.neg();
        defer negged.deinit();
        var filled = try a.fill(99.0);
        defer filled.deinit();

        const d_add = try added.toHost(allocator);
        defer allocator.free(d_add);
        const d_sub = try subbed.toHost(allocator);
        defer allocator.free(d_sub);
        const d_mul = try mulled.toHost(allocator);
        defer allocator.free(d_mul);
        const d_relu = try relued.toHost(allocator);
        defer allocator.free(d_relu);
        const d_neg = try negged.toHost(allocator);
        defer allocator.free(d_neg);
        const d_fill = try filled.toHost(allocator);
        defer allocator.free(d_fill);
        printSlice("add", d_add);
        printSlice("sub", d_sub);
        printSlice("mul", d_mul);
        printSlice("relu", d_relu);
        printSlice("neg", d_neg);
        printSlice("fill(99)", d_fill);
        std.debug.print("\n", .{});
    }

    // ------------------------------------------------------------------ //
    //  6. Broadcast ops
    // ------------------------------------------------------------------ //
    std.debug.print("--- 6. Broadcast Ops (NumPy-style) ---\n", .{});
    {
        // [3,4] + [4]  -> [3,4]
        var a = try cuda.Tensor(f32).fromSlice(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &.{ 3, 4 });
        defer a.deinit();
        var bias = try cuda.Tensor(f32).fromSlice(&.{ 100, 200, 300, 400 }, &.{4});
        defer bias.deinit();

        var result = try a.broadcastAdd(bias);
        defer result.deinit();
        const d = try result.toHost(allocator);
        defer allocator.free(d);
        std.debug.print("  [3,4] + [4] shape -> [{d},{d}]\n", .{ result.size(0), result.size(1) });
        printSlice("  row 0", d[0..4]);
        printSlice("  row 1", d[4..8]);
        printSlice("  row 2", d[8..12]);
        std.debug.print("\n", .{});
    }

    // ------------------------------------------------------------------ //
    //  7. Global reductions
    // ------------------------------------------------------------------ //
    std.debug.print("--- 7. Global Reductions ---\n", .{});
    {
        var t = try cuda.Tensor(f32).fromSlice(&.{ 3, 1, 4, 1, 5, 9, 2, 6 }, &.{ 2, 4 });
        defer t.deinit();
        std.debug.print("  sum:  {d:.1}\n", .{try t.sum()});
        std.debug.print("  mean: {d:.2}\n", .{try t.mean()});
        std.debug.print("  max:  {d:.1}\n", .{try t.max()});
        std.debug.print("  min:  {d:.1}\n\n", .{try t.min()});
    }

    // ------------------------------------------------------------------ //
    //  8. Axis reductions
    // ------------------------------------------------------------------ //
    std.debug.print("--- 8. Axis Reductions ---\n", .{});
    {
        // [3,4] sumAxis(0) -> [4]
        var t = try cuda.Tensor(f32).fromSlice(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &.{ 3, 4 });
        defer t.deinit();

        var col_sum = try t.sumAxis(0);
        defer col_sum.deinit();
        const d = try col_sum.toHost(allocator);
        defer allocator.free(d);
        std.debug.print("  [3,4].sumAxis(0) -> [{d}]\n", .{col_sum.size(0)});
        printSlice("  col sums", d);
        std.debug.print("\n", .{});
    }

    // ------------------------------------------------------------------ //
    //  9. 2-D Matrix multiply
    // ------------------------------------------------------------------ //
    std.debug.print("--- 9. Matrix Multiplication (2-D) ---\n", .{});
    {
        // [2,3] @ [3,2] -> [2,2]
        var a = try cuda.Tensor(f32).fromSlice(&.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
        defer a.deinit();
        var b = try cuda.Tensor(f32).fromSlice(&.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
        defer b.deinit();
        var c = try a.matmul(b);
        defer c.deinit();
        const d = try c.toHost(allocator);
        defer allocator.free(d);
        std.debug.print("  [2,3] @ [3,2] -> [{d},{d}]\n", .{ c.size(0), c.size(1) });
        printSlice("  result", d);
        std.debug.print("\n", .{});
    }

    // ------------------------------------------------------------------ //
    //  10. Batched matrix multiply (3-D)
    // ------------------------------------------------------------------ //
    std.debug.print("--- 10. Batched MatMul (3-D) ---\n", .{});
    {
        // batch=2, [2,2] @ [2,2]
        var a = try cuda.Tensor(f32).fromSlice(&.{
            1, 0, 0, 1, // batch 0 = identity
            2, 0, 0, 2,
        }, // batch 1 = 2*I
            &.{ 2, 2, 2 });
        defer a.deinit();
        var b = try cuda.Tensor(f32).fromSlice(&.{
            1, 2, 3, 4, // batch 0
            5, 6, 7, 8,
        }, // batch 1
            &.{ 2, 2, 2 });
        defer b.deinit();
        var c = try a.batchedMatmul(b);
        defer c.deinit();
        const d = try c.toHost(allocator);
        defer allocator.free(d);
        std.debug.print("  [{d},{d},{d}] @ [{d},{d},{d}] -> [{d},{d},{d}]\n", .{ a.size(0), a.size(1), a.size(2), b.size(0), b.size(1), b.size(2), c.size(0), c.size(1), c.size(2) });
        printSlice("  batch 0 result", d[0..4]);
        printSlice("  batch 1 result", d[4..8]);
        std.debug.print("\n", .{});
    }

    // ------------------------------------------------------------------ //
    //  11. Slice
    // ------------------------------------------------------------------ //
    std.debug.print("--- 11. Slice ---\n", .{});
    {
        var t = try cuda.Tensor(f32).fromSlice(&.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }, &.{ 3, 4 });
        defer t.deinit();

        // rows [0,2), cols [1,3) -> 2x2
        var s = try t.slice(&.{ 0, 1 }, &.{ 2, 3 });
        defer s.deinit();
        const d = try s.toHost(allocator);
        defer allocator.free(d);
        std.debug.print("  [3,4][0:2, 1:3] -> [{d},{d}]\n", .{ s.size(0), s.size(1) });
        printSlice("  result", d);
        std.debug.print("\n", .{});
    }

    // ------------------------------------------------------------------ //
    //  12. Concat
    // ------------------------------------------------------------------ //
    std.debug.print("--- 12. Concat ---\n", .{});
    {
        var a = try cuda.Tensor(f32).fromSlice(&.{ 1, 2, 3, 4 }, &.{ 2, 2 });
        defer a.deinit();
        var b = try cuda.Tensor(f32).fromSlice(&.{ 5, 6, 7, 8 }, &.{ 2, 2 });
        defer b.deinit();

        var c0 = try a.concat(b, 0);
        defer c0.deinit();
        var c1 = try a.concat(b, 1);
        defer c1.deinit();

        const d0 = try c0.toHost(allocator);
        defer allocator.free(d0);
        const d1 = try c1.toHost(allocator);
        defer allocator.free(d1);
        std.debug.print("  concat axis=0 -> [{d},{d}]\n", .{ c0.size(0), c0.size(1) });
        printSlice("  values", d0);
        std.debug.print("  concat axis=1 -> [{d},{d}]\n", .{ c1.size(0), c1.size(1) });
        printSlice("  values", d1);
        std.debug.print("\n", .{});
    }

    std.debug.print("All N-D tensor operations completed successfully.\n", .{});
}

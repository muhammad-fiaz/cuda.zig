const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Tensor Ops Example ===\n", .{});

    const allocator = std.heap.page_allocator;

    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    var tensor_a = try cuda.Tensor(f32).fromSlice(&a_data, &.{ 2, 2 });
    defer tensor_a.deinit();

    var tensor_b = try cuda.Tensor(f32).fromSlice(&b_data, &.{ 2, 2 });
    defer tensor_b.deinit();

    var tensor_add = try tensor_a.add(tensor_b);
    defer tensor_add.deinit();

    const host_add = try tensor_add.toHost(allocator);
    defer allocator.free(host_add);

    std.debug.print("Tensor Add Output [2x2]: {any}\n", .{host_add});

    var tensor_mm = try tensor_a.matmul(tensor_b);
    defer tensor_mm.deinit();

    const host_mm = try tensor_mm.toHost(allocator);
    defer allocator.free(host_mm);

    std.debug.print("Tensor Matmul Output [2x2]: {any}\n", .{host_mm});
}

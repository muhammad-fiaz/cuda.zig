//! Kernel Launch Configuration and launch helper.

const std = @import("std");
const loader = @import("../core/loader.zig");
const stream_mod = @import("../stream/stream.zig");
const drv_fn = @import("../driver/function.zig");
const drv_ffi = @import("../driver/ffi.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");

/// Grid and block launch configuration for kernel execution.
pub const LaunchConfig = struct {
    grid: [3]u32 = .{ 1, 1, 1 },
    block: [3]u32 = .{ 1, 1, 1 },
    shared_mem_bytes: u32 = 0,
    stream: ?stream_mod.Stream = null,
};

/// Launches a compiled CUDA driver kernel function `func` with `config` and `args`.
///
/// `args` must be a tuple containing the arguments to pass to the kernel in order.
pub fn launch(func: drv_fn.DriverFunction, config: LaunchConfig, args: anytype) err.CudaError!void {
    const ldr = loader.globalLoader();
    const f = ldr.getDriverSymbol(drv_ffi.LaunchKernelFn, "cuLaunchKernel") orelse
        return error.NotInitialized;

    const ArgsType = @TypeOf(args);
    const args_info = @typeInfo(ArgsType);

    if (args_info != .@"struct" or !args_info.@"struct".is_tuple) {
        @compileError("launch args parameter must be a tuple (e.g. .{ arg1, arg2 })");
    }

    const fields = args_info.@"struct".fields;
    var ptrs: [fields.len]?*anyopaque = undefined;
    inline for (fields, 0..) |field, i| {
        const val_ptr = &@field(args, field.name);
        ptrs[i] = @ptrCast(@constCast(val_ptr));
    }

    const strm_handle = if (config.stream) |s| s.handle else null;

    const code = f(
        func.handle,
        config.grid[0],
        config.grid[1],
        config.grid[2],
        config.block[0],
        config.block[1],
        config.block[2],
        config.shared_mem_bytes,
        strm_handle,
        ptrs.ptr,
        null,
    );
    try result.checkDriver(code);
}

test "LaunchConfig default parameters" {
    const cfg = LaunchConfig{};
    try std.testing.expectEqual([3]u32{ 1, 1, 1 }, cfg.grid);
    try std.testing.expectEqual([3]u32{ 1, 1, 1 }, cfg.block);
    try std.testing.expectEqual(@as(u32, 0), cfg.shared_mem_bytes);
    try std.testing.expect(cfg.stream == null);
}

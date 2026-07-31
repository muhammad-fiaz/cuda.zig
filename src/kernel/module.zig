//! High-level kernel module wrapper.

const std = @import("std");
const drv_mod = @import("../driver/module.zig");
const drv_fn = @import("../driver/function.zig");
const err = @import("../core/error.zig");

/// High-level wrapper around compiled CUDA modules (PTX/cubin/fatbin).
pub const KernelModule = struct {
    driver_module: drv_mod.DriverModule,

    pub fn loadPTX(ptx_code: []const u8) err.CudaError!KernelModule {
        const dm = try drv_mod.DriverModule.loadData(ptx_code);
        return .{ .driver_module = dm };
    }

    pub fn loadFile(path: [*:0]const u8) err.CudaError!KernelModule {
        const dm = try drv_mod.DriverModule.loadFile(path);
        return .{ .driver_module = dm };
    }

    pub fn getFunction(self: KernelModule, name: [:0]const u8) err.CudaError!drv_fn.DriverFunction {
        return drv_fn.DriverFunction.lookup(self.driver_module, name);
    }

    pub fn deinit(self: *KernelModule) void {
        self.driver_module.deinit();
    }
};

test "KernelModule loadPTX skips without CUDA" {
    if (!@import("../core/loader.zig").isAvailable()) return error.SkipZigTest;
}

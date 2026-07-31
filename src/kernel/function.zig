//! High-level typed kernel Function handle.

const std = @import("std");
const drv_fn = @import("../driver/function.zig");
const launch_mod = @import("launch.zig");
const err = @import("../core/error.zig");

pub const Function = struct {
    driver_func: drv_fn.DriverFunction,

    pub fn init(driver_func: drv_fn.DriverFunction) Function {
        return .{ .driver_func = driver_func };
    }

    pub fn launch(self: Function, config: launch_mod.LaunchConfig, args: anytype) err.CudaError!void {
        return launch_mod.launch(self.driver_func, config, args);
    }
};

test "Function struct init" {
    // Basic structural test
    const dummy_fn = drv_fn.DriverFunction{ .handle = undefined, .name = "test_kernel" };
    const f = Function.init(dummy_fn);
    try std.testing.expectEqualStrings("test_kernel", f.driver_func.name);
}

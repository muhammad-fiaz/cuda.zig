//! CUDA driver kernel function handle lookup.
//!
//! A `DriverFunction` wraps a `CUfunction` handle retrieved from a loaded
//! `DriverModule`. It is the token passed to `kernel/launch.zig` to identify
//! which kernel to execute.
//!
//! A `DriverFunction` is only valid as long as the `DriverModule` it was
//! obtained from remains loaded. Unloading the module invalidates all function
//! handles derived from it.

const std = @import("std");
const loader = @import("../core/loader.zig");
const ffi = @import("ffi.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");
const module_mod = @import("module.zig");

/// A typed, named handle to a CUDA kernel function.
pub const DriverFunction = struct {
    handle: *anyopaque,
    /// The name of the kernel as it appears in the module symbol table.
    name: []const u8,

    /// Looks up the function named `name` in `mod`.
    ///
    /// `name` must exactly match the mangled symbol name in the compiled
    /// image. For PTX kernels compiled with `extern "C"`, the name is the
    /// unmangled C identifier. For C++ kernels, use the mangled name or
    /// declare them `extern "C"`.
    ///
    /// The returned `DriverFunction` borrows both the string slice for `name`
    /// and the module handle. Both must outlive the `DriverFunction`.
    pub fn lookup(mod: module_mod.DriverModule, name: [:0]const u8) err.CudaError!DriverFunction {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.ModuleGetFunctionFn, "cuModuleGetFunction") orelse
            return error.NotInitialized;
        var handle: *anyopaque = undefined;
        const code = f(&handle, mod.handle, name.ptr);
        try result.checkDriver(code);
        return .{ .handle = handle, .name = name };
    }
};

test "DriverFunction.lookup skips without CUDA" {
    if (!loader.isAvailable()) return error.SkipZigTest;
}

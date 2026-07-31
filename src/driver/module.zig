//! CUDA driver module loading (PTX, cubin, fatbin).
//!
//! A `DriverModule` wraps a `CUmodule` handle loaded from a compiled kernel
//! image. Three image formats are supported:
//!
//!   PTX    — parallel thread execution assembly text. JIT-compiled by the
//!             driver for the current device; portable across compute generations
//!             within the same CUDA major version.
//!   cubin  — device-specific binary. Not portable across compute capability
//!             generations but eliminates JIT overhead.
//!   fatbin — fat binary containing multiple cubins and/or PTX images. The
//!             driver selects the best matching image for the current device.
//!
//! After loading, kernel functions are looked up by name using
//! `driver/function.zig`.

const std = @import("std");
const loader = @import("../core/loader.zig");
const ffi = @import("ffi.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");
const context = @import("context.zig");

/// Wraps a `CUmodule` handle and provides safe lifecycle management.
pub const DriverModule = struct {
    handle: *anyopaque,

    /// Loads a kernel module from the file at `path`.
    ///
    /// `path` must be a null-terminated path to a `.ptx`, `.cubin`, or
    /// `.fatbin` file. The driver JIT-compiles PTX images for the current
    /// device on load.
    ///
    /// A valid CUDA context must be current on the calling thread before
    /// calling this function.
    pub fn loadFile(path: [*:0]const u8) err.CudaError!DriverModule {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.ModuleLoadFn, "cuModuleLoad") orelse
            return error.NotInitialized;
        var handle: *anyopaque = undefined;
        try result.checkDriver(f(&handle, path));
        return .{ .handle = handle };
    }

    /// Loads a kernel module from an in-memory image.
    ///
    /// `image` is a slice whose bytes are a valid PTX, cubin, or fatbin image.
    /// The memory referenced by `image` need not remain valid after this call
    /// returns; the driver makes an internal copy.
    ///
    /// A valid CUDA context must be current on the calling thread.
    pub fn loadData(image: []const u8) err.CudaError!DriverModule {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.ModuleLoadDataFn, "cuModuleLoadData") orelse
            return error.NotInitialized;
        var handle: *anyopaque = undefined;
        // PTX text must be null-terminated for cuModuleLoadData
        if (image.len > 0 and image[image.len - 1] == 0) {
            try result.checkDriver(f(&handle, image.ptr));
        } else {
            var buf = std.heap.page_allocator.alloc(u8, image.len + 1) catch return error.OutOfMemory;
            defer std.heap.page_allocator.free(buf);
            @memcpy(buf[0..image.len], image);
            buf[image.len] = 0;
            try result.checkDriver(f(&handle, buf.ptr));
        }
        return .{ .handle = handle };
    }

    /// Loads a fatbin image from memory.
    ///
    /// Equivalent to `loadData` but uses `cuModuleLoadFatBinary` which
    /// provides more informative JIT logging in debug builds.
    pub fn loadFatBinary(image: []const u8) err.CudaError!DriverModule {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.ModuleLoadFatBinaryFn, "cuModuleLoadFatBinary") orelse
            return error.NotInitialized;
        var handle: *anyopaque = undefined;
        try result.checkDriver(f(&handle, image.ptr));
        return .{ .handle = handle };
    }

    /// Unloads the module and releases all resources held by it.
    ///
    /// After `deinit`, any `DriverFunction` handles obtained from this module
    /// are invalid. Using them is undefined behavior.
    pub fn deinit(self: *DriverModule) void {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.ModuleUnloadFn, "cuModuleUnload") orelse return;
        _ = f(self.handle);
    }

    /// Looks up the device address and byte size of a global variable in this
    /// module by name.
    ///
    /// Returns the device pointer and byte count of the named global. The
    /// pointer is device-side only; reading or writing it requires memory
    /// copy operations.
    pub fn getGlobal(self: DriverModule, name: [*:0]const u8) err.CudaError!struct {
        ptr: ffi.DevicePtr,
        size: usize,
    } {
        const ldr = loader.globalLoader();
        const f = ldr.getDriverSymbol(ffi.ModuleGetGlobalFn, "cuModuleGetGlobal_v2") orelse
            ldr.getDriverSymbol(ffi.ModuleGetGlobalFn, "cuModuleGetGlobal") orelse
            return error.NotInitialized;
        var ptr: ffi.DevicePtr = 0;
        var size: usize = 0;
        try result.checkDriver(f(&ptr, &size, self.handle, name));
        return .{ .ptr = ptr, .size = size };
    }
};

test "DriverModule.loadData skips without CUDA" {
    if (!loader.isAvailable()) return error.SkipZigTest;
}

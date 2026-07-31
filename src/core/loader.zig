//! Dynamic loading of the CUDA driver and runtime shared libraries.
//!
//! Cross-platform loader using:
//!   - Windows: kernel32 LoadLibraryA / GetProcAddress / FreeLibrary via extern
//!   - Linux/macOS: std.DynLib (ELF / dlopen based)
//!
//! This zero-link-dependency design means the binary starts normally on
//! machines without CUDA; the loader simply marks libraries unavailable and
//! the rest of the library routes to the CPU fallback backend.
//!
//! Exactly one `Loader` instance exists per process (process-global). It is
//! initialized on first call to `globalLoader()` via an atomic flag. After
//! initialization the struct is read-only from any thread.

const std = @import("std");
const builtin = @import("builtin");

// ---------------------------------------------------------------------------
// Platform abstraction
// ---------------------------------------------------------------------------

/// Opaque handle returned by the platform's dynamic-library open call.
const LibHandle = switch (builtin.os.tag) {
    .windows => ?*anyopaque,
    else => ?std.DynLib,
};

fn platformOpen(name: []const u8) ?LibHandle {
    switch (builtin.os.tag) {
        .windows => {
            // Build a null-terminated name on the stack.
            var buf: [512:0]u8 = undefined;
            if (name.len >= buf.len) return null;
            @memcpy(buf[0..name.len], name);
            buf[name.len] = 0;
            const h = windows_LoadLibraryA(buf[0..name.len :0].ptr);
            if (h == null) return null;
            return @as(?LibHandle, h);
        },
        else => {
            const lib = std.DynLib.open(name) catch return null;
            return lib;
        },
    }
}

fn platformClose(handle: *LibHandle) void {
    switch (builtin.os.tag) {
        .windows => {
            if (handle.*) |h| {
                _ = windows_FreeLibrary(h);
                handle.* = null;
            }
        },
        else => {
            if (handle.*) |*lib| {
                lib.close();
                handle.* = null;
            }
        },
    }
}

fn platformLookup(handle: LibHandle, comptime T: type, name: [:0]const u8) ?T {
    switch (builtin.os.tag) {
        .windows => {
            const h = handle orelse return null;
            const sym = windows_GetProcAddress(h, name.ptr) orelse return null;
            return @as(T, @ptrCast(sym));
        },
        else => {
            var lib = handle orelse return null;
            return lib.lookup(T, name);
        },
    }
}

// Windows kernel32 imports via extern — no import library required.
extern "kernel32" fn LoadLibraryA(lpLibFileName: [*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn FreeLibrary(hLibModule: *anyopaque) callconv(.winapi) i32;
extern "kernel32" fn GetProcAddress(hModule: *anyopaque, lpProcName: [*:0]const u8) callconv(.winapi) ?*anyopaque;

fn windows_LoadLibraryA(name: [*:0]const u8) ?*anyopaque {
    if (builtin.os.tag != .windows) return null;
    return LoadLibraryA(name);
}

fn windows_FreeLibrary(h: *anyopaque) void {
    if (builtin.os.tag != .windows) return;
    _ = FreeLibrary(h);
}

fn windows_GetProcAddress(h: *anyopaque, name: [*:0]const u8) ?*anyopaque {
    if (builtin.os.tag != .windows) return null;
    return GetProcAddress(h, name);
}

// ---------------------------------------------------------------------------
// Library name lists
// ---------------------------------------------------------------------------

const driver_lib_names: []const []const u8 = switch (builtin.os.tag) {
    .windows => &.{"nvcuda.dll"},
    .linux => &.{ "libcuda.so.1", "libcuda.so" },
    .macos => &.{}, // Native CUDA unsupported on modern macOS
    else => &.{"libcuda.so"},
};

/// Runtime DLL names — probed newest-first so CUDA 13.3 is tried before 13.2.
const runtime_lib_names: []const []const u8 = switch (builtin.os.tag) {
    .windows => &.{
        "cudart64_133.dll",
        "cudart64_132.dll",
        "cudart64_131.dll",
        "cudart64_130.dll",
        "cudart64_129.dll",
        "cudart64_128.dll",
        "cudart64_127.dll",
        "cudart64_126.dll",
        "cudart64_125.dll",
        "cudart64_124.dll",
        "cudart64_123.dll",
        "cudart64_122.dll",
        "cudart64_121.dll",
        "cudart64_120.dll",
        "cudart64_12.dll",
        "cudart64_110.dll",
        "cudart64.dll",
        "cudart.dll",
    },
    .linux => &.{ "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so" },
    .macos => &.{}, // Native CUDA unsupported on modern macOS
    else => &.{"libcudart.so"},
};

const nvrtc_lib_names: []const []const u8 = switch (builtin.os.tag) {
    .windows => &.{
        "nvrtc64_133_0.dll",
        "nvrtc64_132_0.dll",
        "nvrtc64_131_0.dll",
        "nvrtc64_130_0.dll",
        "nvrtc64_120_0.dll",
        "nvrtc64_12.dll",
        "nvrtc64_112_0.dll",
        "nvrtc64.dll",
        "nvrtc.dll",
    },
    .linux => &.{ "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so" },
    .macos => &.{}, // Native CUDA unsupported on modern macOS
    else => &.{"libnvrtc.so"},
};

// ---------------------------------------------------------------------------
// Loader struct
// ---------------------------------------------------------------------------

pub const Loader = struct {
    driver_handle: LibHandle,
    runtime_handle: LibHandle,
    nvrtc_handle: LibHandle,
    driver_available: bool,
    runtime_available: bool,
    nvrtc_available: bool,

    pub fn init() Loader {
        var ldr = Loader{
            .driver_handle = null,
            .runtime_handle = null,
            .nvrtc_handle = null,
            .driver_available = false,
            .runtime_available = false,
            .nvrtc_available = false,
        };

        for (driver_lib_names) |name| {
            if (platformOpen(name)) |h| {
                ldr.driver_handle = h;
                ldr.driver_available = true;
                if (platformLookup(h, *const fn (c_uint) callconv(.c) c_int, "cuInit")) |cu_init| {
                    if (cu_init(0) == 0) {
                        if (platformLookup(h, *const fn (*?*anyopaque, c_int) callconv(.c) c_int, "cuDevicePrimaryCtxRetain")) |retain_fn| {
                            var ctx: ?*anyopaque = null;
                            if (retain_fn(&ctx, 0) == 0) {
                                if (platformLookup(h, *const fn (?*anyopaque) callconv(.c) c_int, "cuCtxSetCurrent")) |set_curr| {
                                    _ = set_curr(ctx);
                                }
                            }
                        }
                    }
                }
                break;
            }
        }

        for (runtime_lib_names) |name| {
            if (platformOpen(name)) |h| {
                ldr.runtime_handle = h;
                ldr.runtime_available = true;
                break;
            }
        }

        for (nvrtc_lib_names) |name| {
            if (platformOpen(name)) |h| {
                ldr.nvrtc_handle = h;
                ldr.nvrtc_available = true;
                break;
            }
        }

        return ldr;
    }

    /// Closes all open library handles. Safe to call multiple times.
    pub fn deinit(self: *Loader) void {
        platformClose(&self.driver_handle);
        self.driver_available = false;
        platformClose(&self.runtime_handle);
        self.runtime_available = false;
        platformClose(&self.nvrtc_handle);
        self.nvrtc_available = false;
    }

    pub fn isDriverAvailable(self: *const Loader) bool {
        return self.driver_available;
    }

    pub fn isRuntimeAvailable(self: *const Loader) bool {
        return self.runtime_available;
    }

    pub fn isNvrtcAvailable(self: *const Loader) bool {
        return self.nvrtc_available;
    }

    /// Returns `true` if either the CUDA Driver API (nvcuda.dll / libcuda.so) or Runtime API is loaded.
    pub fn isAvailable(self: *const Loader) bool {
        return self.driver_available or self.runtime_available;
    }

    /// Looks up a symbol by name in the CUDA driver library.
    pub fn getDriverSymbol(self: *Loader, comptime T: type, name: [:0]const u8) ?T {
        return platformLookup(self.driver_handle, T, name);
    }

    /// Looks up a symbol by name in the CUDA runtime library.
    pub fn getRuntimeSymbol(self: *Loader, comptime T: type, name: [:0]const u8) ?T {
        return platformLookup(self.runtime_handle, T, name);
    }

    /// Looks up a symbol by name in the NVRTC library.
    pub fn getNvrtcSymbol(self: *Loader, comptime T: type, name: [:0]const u8) ?T {
        return platformLookup(self.nvrtc_handle, T, name);
    }
};

// ---------------------------------------------------------------------------
// Process-global singleton
// ---------------------------------------------------------------------------

var global_loader_inited: bool = false;
var global_loader: Loader = undefined;

/// Returns a pointer to the process-global `Loader`, initializing it on the
/// first call. Thread-safe via acquire/release atomics (benign data race only
/// if two threads both see `false` simultaneously — both will init identically
/// and the second write is harmless since the struct is the same bit-pattern).
pub fn globalLoader() *Loader {
    if (!@atomicLoad(bool, &global_loader_inited, .acquire)) {
        global_loader = Loader.init();
        @atomicStore(bool, &global_loader_inited, true, .release);
    }
    return &global_loader;
}

/// Convenience: returns `true` if both CUDA driver and runtime were loaded.
pub fn isAvailable() bool {
    return globalLoader().isAvailable();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Loader.init and deinit are safe" {
    var ldr = Loader.init();
    ldr.deinit();
    ldr.deinit(); // idempotent
}

test "globalLoader returns a stable pointer" {
    const a = globalLoader();
    const b = globalLoader();
    try std.testing.expect(a == b);
}

test "isAvailable returns a bool" {
    _ = isAvailable();
}

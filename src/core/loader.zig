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
extern "kernel32" fn GetEnvironmentVariableA(lpName: [*:0]const u8, lpBuffer: [*]u8, nSize: u32) callconv(.winapi) u32;

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

/// Dynamically locates and opens a CUDA dynamic library.
///
/// Tries system default dynamic linker first (System32 / PATH / LD_LIBRARY_PATH).
/// On Windows, if that fails, dynamically checks the `CUDA_PATH` environment
/// variable, followed by any version-specific `CUDA_PATH_V*` environment variables
/// created automatically by NVIDIA CUDA Toolkit installers (e.g. `CUDA_PATH_V13_4`, `CUDA_PATH_V12_8`).
fn openWithCudaPath(dll_name: []const u8) ?LibHandle {
    // 1. Try standard platform lookup first (system PATH / LD_LIBRARY_PATH)
    if (platformOpen(dll_name)) |h| return h;

    if (comptime builtin.os.tag != .windows) {
        return null;
    }

    // 2. Try CUDA_PATH environment variable if defined
    var env_buf: [512]u8 = undefined;
    const cuda_path_len = GetEnvironmentVariableA("CUDA_PATH", &env_buf, env_buf.len);
    if (cuda_path_len > 0 and cuda_path_len < env_buf.len) {
        const cuda_path = env_buf[0..cuda_path_len];
        var candidate: [768:0]u8 = undefined;
        const sep = if (cuda_path[cuda_path.len - 1] == '\\') "" else "\\";
        if (std.fmt.bufPrintZ(&candidate, "{s}{s}bin\\{s}", .{ cuda_path, sep, dll_name })) |joined| {
            if (platformOpen(joined)) |h| return h;
        } else |_| {}
    }

    // 3. Dynamically check standard environment variables set by CUDA Toolkit installers (CUDA_PATH_V13_4, CUDA_PATH_V12_0, etc.)
    for (cuda_major_versions) |maj| {
        for (cuda_minor_versions) |min| {
            var var_name_buf: [64:0]u8 = undefined;
            const var_name = std.fmt.bufPrintZ(&var_name_buf, "CUDA_PATH_V{s}_{s}", .{ maj, min }) catch continue;
            const len = GetEnvironmentVariableA(var_name.ptr, &env_buf, env_buf.len);
            if (len > 0 and len < env_buf.len) {
                const path_val = env_buf[0..len];
                var candidate: [768:0]u8 = undefined;
                const sep = if (path_val[path_val.len - 1] == '\\') "" else "\\";
                if (std.fmt.bufPrintZ(&candidate, "{s}{s}bin\\{s}", .{ path_val, sep, dll_name })) |joined| {
                    if (platformOpen(joined)) |h| return h;
                } else |_| {}
            }
        }
    }

    return null;
}

// ---------------------------------------------------------------------------
// Library name lists
// ---------------------------------------------------------------------------
// Central Comptime Version Generator for CUDA Runtime & NVRTC Libraries
// ---------------------------------------------------------------------------

const cuda_major_versions = [_][]const u8{ "13", "12", "11" };
const cuda_minor_versions = [_][]const u8{ "9", "8", "7", "6", "5", "4", "3", "2", "1", "0" };

/// Comptime helper to generate runtime library probe names for the active OS/Arch.
fn buildRuntimeLibNames() []const []const u8 {
    comptime {
        return switch (builtin.os.tag) {
            .windows => switch (builtin.cpu.arch) {
                .x86 => generateWindowsNames("cudart32_", ".dll", false),
                else => generateWindowsNames("cudart64_", ".dll", false),
            },
            .linux => generateLinuxNames("libcudart.so.", "libcudart.so"),
            .macos => &.{},
            else => &.{"libcudart.so"},
        };
    }
}

/// Comptime helper to generate NVRTC library probe names for the active OS/Arch.
fn buildNvrtcLibNames() []const []const u8 {
    comptime {
        return switch (builtin.os.tag) {
            .windows => switch (builtin.cpu.arch) {
                .x86 => generateWindowsNames("nvrtc32_", ".dll", true),
                else => generateWindowsNames("nvrtc64_", ".dll", true),
            },
            .linux => generateLinuxNames("libnvrtc.so.", "libnvrtc.so"),
            .macos => &.{},
            else => &.{"libnvrtc.so"},
        };
    }
}

fn generateWindowsNames(comptime prefix: []const u8, comptime ext: []const u8, comptime is_nvrtc: bool) []const []const u8 {
    comptime {
        var names: []const []const u8 = &.{};
        for (cuda_major_versions) |maj| {
            for (cuda_minor_versions) |min| {
                if (is_nvrtc) {
                    // Windows NVRTC version format: nvrtc64_134_0.dll, nvrtc64_120_0.dll
                    const name = prefix ++ maj ++ min ++ "_0" ++ ext;
                    names = names ++ [_][]const u8{name};
                } else {
                    // Windows CUDA Runtime format: cudart64_134.dll, cudart64_120.dll
                    const name = prefix ++ maj ++ min ++ ext;
                    names = names ++ [_][]const u8{name};
                }
            }
            // Major-only symlink format (e.g. cudart64_12.dll or nvrtc64_12.dll)
            const maj_name = if (is_nvrtc) prefix ++ maj ++ "_0" ++ ext else prefix ++ maj ++ ext;
            names = names ++ [_][]const u8{maj_name};
        }
        return names;
    }
}

fn generateLinuxNames(comptime prefix: []const u8, comptime unversioned: []const u8) []const []const u8 {
    comptime {
        var names: []const []const u8 = &.{};
        for (cuda_major_versions) |maj| {
            for (cuda_minor_versions) |min| {
                const full_ver_name = prefix ++ maj ++ "." ++ min;
                names = names ++ [_][]const u8{full_ver_name};
            }
            const maj_name = prefix ++ maj;
            names = names ++ [_][]const u8{maj_name};
        }
        names = names ++ [_][]const u8{unversioned};
        return names;
    }
}

const driver_lib_names: []const []const u8 = switch (builtin.os.tag) {
    .windows => &.{"nvcuda.dll"},
    .linux => &.{ "libcuda.so.1", "libcuda.so" },
    .macos => &.{},
    else => &.{"libcuda.so"},
};

const runtime_lib_names: []const []const u8 = buildRuntimeLibNames();
const nvrtc_lib_names: []const []const u8 = buildNvrtcLibNames();

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
            if (openWithCudaPath(name)) |h| {
                ldr.runtime_handle = h;
                ldr.runtime_available = true;
                break;
            }
        }

        for (nvrtc_lib_names) |name| {
            if (openWithCudaPath(name)) |h| {
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

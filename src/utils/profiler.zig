//! CUDA Profiler Range Markers
//!
//! Provides Zig wrappers for:
//!   - `cudaProfilerStart` / `cudaProfilerStop` (coarse-grained session control)
//!
//! NVTX push/pop ranges require the NVTX3 C library; this module gives a
//! safe no-op fallback when not available. Import as:
//! ```zig
//! const prof = @import("cuda").profiler;
//! ```
//!
//! On machines without the CUDA profiler (Nsight Systems / Nsight Compute not
//! running) all calls return `void` silently.

const std = @import("std");
const loader = @import("../core/loader.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");

// Profiler session control

const ProfilerStartFn = *const fn () callconv(.c) c_int;
const ProfilerStopFn = *const fn () callconv(.c) c_int;

/// Enables profiler data collection for the current session.
///
/// When Nsight Systems or Nsight Compute is attached, calling `start()` before
/// your region of interest reduces trace noise. No-op if the profiler library
/// is not loaded.
pub fn start() void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ProfilerStartFn, "cudaProfilerStart")) |f| {
        _ = f();
    }
}

/// Disables profiler data collection for the current session.
pub fn stop() void {
    const ldr = loader.globalLoader();
    if (ldr.getRuntimeSymbol(ProfilerStopFn, "cudaProfilerStop")) |f| {
        _ = f();
    }
}

// Scoped profiler guard (RAII-style)

/// A defer-based guard that calls `stop()` when it goes out of scope.
///
/// Usage:
/// ```zig
/// {
///     var _prof = profiler.ProfilerGuard.start();
///     defer _prof.stop();
///     // ... your GPU workload ...
/// }
/// ```
pub const ProfilerGuard = struct {
    pub fn begin() ProfilerGuard {
        start();
        return ProfilerGuard{};
    }

    pub fn end(_: *ProfilerGuard) void {
        stop();
    }
};

test "profiler start/stop are callable" {
    start();
    stop();
}

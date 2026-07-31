//! CUDA version detection and runtime capability flags.
//!
//! This module is the single point of truth for version-sensitive logic in
//! cuda.zig. No other module in the library should branch directly on integer
//! version numbers; instead, they should query the `RuntimeCapabilities` flags
//! exposed here.
//!
//! Version detection uses `cudaRuntimeGetVersion` and `cuDriverGetVersion` from
//! the loaded libraries. Both return a single integer of the form
//! `MAJOR * 1000 + MINOR * 10`; for example CUDA 12.3 is `12030`.

const std = @import("std");
const loader = @import("loader.zig");
const err = @import("error.zig");

/// Encodes a CUDA version as returned by `cudaRuntimeGetVersion` or
/// `cuDriverGetVersion`.
pub const CudaVersion = struct {
    /// The raw version integer (MAJOR * 1000 + MINOR * 10).
    raw: i32,

    /// Constructs a `CudaVersion` from the raw integer returned by the CUDA API.
    pub fn fromRaw(v: i32) CudaVersion {
        return .{ .raw = v };
    }

    /// Returns the major version component.
    pub fn major(self: CudaVersion) i32 {
        return @divFloor(self.raw, 1000);
    }

    /// Returns the minor version component.
    pub fn minor(self: CudaVersion) i32 {
        return @divFloor(@mod(self.raw, 1000), 10);
    }

    /// Returns `true` if this version is at least `maj.min`.
    pub fn atLeast(self: CudaVersion, maj: i32, min: i32) bool {
        return self.raw >= maj * 1000 + min * 10;
    }

    /// Formats the version as `MAJOR.MINOR` (e.g. "12.3").
    pub fn format(
        self: CudaVersion,
        comptime fmt_str: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt_str;
        _ = options;
        try writer.print("{d}.{d}", .{ self.major(), self.minor() });
    }
};

/// Set of capability flags derived from the detected driver and runtime
/// versions. Components of cuda.zig use these flags rather than inspecting
/// raw version integers.
pub const RuntimeCapabilities = struct {
    /// The installed CUDA driver version (cuDriverGetVersion).
    driver_version: CudaVersion,
    /// The linked CUDA runtime version (cudaRuntimeGetVersion).
    runtime_version: CudaVersion,

    /// True if the driver supports CUDA 12.x or later.
    supports_cuda_12: bool,
    /// True if the driver supports CUDA 13.x or later.
    supports_cuda_13: bool,

    /// True if CUDA Graphs capture/instantiate/launch APIs are available.
    /// Requires driver >= 10.0.
    supports_graphs: bool,

    /// True if cooperative kernel launch (`cudaLaunchCooperativeKernel`) is
    /// available. Requires driver >= 9.0.
    supports_cooperative_launch: bool,

    /// True if unified/managed memory prefetch with device hints is available.
    /// Requires driver >= 8.0.
    supports_managed_prefetch: bool,

    /// True if stream capture for CUDA Graphs is supported.
    /// Requires driver >= 10.0.
    supports_stream_capture: bool,

    /// True if the memory pool APIs (`cudaMallocAsync`/`cudaFreeAsync`) are
    /// available. Requires driver >= 11.2.
    supports_memory_pools: bool,

    /// True if Tensor Core operations (via cuBLAS / WMMA) are potentially
    /// available. This flag reflects driver support only; the actual compute
    /// capability of the device must also be >= sm_70.
    supports_tensor_cores: bool,

    /// True if CUDA 13.x-specific features (e.g. improved cluster launch,
    /// updated graph APIs) are available.
    supports_cuda_13_features: bool,
};

// Function pointer types for version queries.
const CudaRuntimeGetVersionFn = *const fn (version: *i32) callconv(.c) c_int;
const CuDriverGetVersionFn = *const fn (version: *i32) callconv(.c) c_int;

/// Queries the installed driver and runtime versions and computes the full
/// `RuntimeCapabilities` struct.
///
/// If the runtime or driver library is not loaded, version values are set to
/// zero and all capability flags are `false`.
///
/// This function makes live CUDA API calls; call it once during application
/// startup and cache the result. It does not need a device to be present —
/// version queries work without an active context.
pub fn detect() RuntimeCapabilities {
    const ldr = loader.globalLoader();

    var driver_ver: i32 = 0;
    var runtime_ver: i32 = 0;

    if (ldr.getDriverSymbol(CuDriverGetVersionFn, "cuDriverGetVersion")) |f| {
        _ = f(&driver_ver);
    }

    if (ldr.getRuntimeSymbol(CudaRuntimeGetVersionFn, "cudaRuntimeGetVersion")) |f| {
        _ = f(&runtime_ver);
    }

    const dv = CudaVersion.fromRaw(driver_ver);
    const rv = CudaVersion.fromRaw(runtime_ver);

    return RuntimeCapabilities{
        .driver_version = dv,
        .runtime_version = rv,
        .supports_cuda_12 = dv.atLeast(12, 0) or rv.atLeast(12, 0),
        .supports_cuda_13 = dv.atLeast(13, 0) or rv.atLeast(13, 0),
        .supports_graphs = dv.atLeast(10, 0) or rv.atLeast(10, 0),
        .supports_cooperative_launch = dv.atLeast(9, 0) or rv.atLeast(9, 0),
        .supports_managed_prefetch = dv.atLeast(8, 0) or rv.atLeast(8, 0),
        .supports_stream_capture = dv.atLeast(10, 0) or rv.atLeast(10, 0),
        .supports_memory_pools = dv.atLeast(11, 2) or rv.atLeast(11, 2),
        .supports_tensor_cores = dv.atLeast(9, 0) or rv.atLeast(9, 0),
        .supports_cuda_13_features = dv.atLeast(13, 0) or rv.atLeast(13, 0),
    };
}

/// Process-global cached capabilities, populated once on first access.
var capabilities_inited = false;
var cached_capabilities: RuntimeCapabilities = undefined;

pub fn capabilities() *const RuntimeCapabilities {
    if (!@atomicLoad(bool, &capabilities_inited, .acquire)) {
        cached_capabilities = detect();
        @atomicStore(bool, &capabilities_inited, true, .release);
    }
    return &cached_capabilities;
}

test "CudaVersion.fromRaw" {
    const v = CudaVersion.fromRaw(12030);
    try std.testing.expectEqual(@as(i32, 12), v.major());
    try std.testing.expectEqual(@as(i32, 3), v.minor());
}

test "CudaVersion.atLeast" {
    const v = CudaVersion.fromRaw(12030);
    try std.testing.expect(v.atLeast(12, 0));
    try std.testing.expect(v.atLeast(12, 3));
    try std.testing.expect(!v.atLeast(12, 4));
    try std.testing.expect(!v.atLeast(13, 0));
}

test "detect does not crash" {
    // Simply verify detect() runs without panic regardless of CUDA presence.
    const caps = detect();
    // runtime_version.raw >= 0 is always true (i32), just consume it.
    try std.testing.expect(caps.runtime_version.raw >= 0);
    try std.testing.expect(caps.driver_version.raw >= 0);
}

test "capabilities returns a stable pointer" {
    const a = capabilities();
    const b = capabilities();
    try std.testing.expect(a == b);
}

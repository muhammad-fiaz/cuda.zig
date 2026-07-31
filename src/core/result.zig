//! Generic helpers for converting raw CUDA FFI return codes into Zig errors.
//!
//! All FFI call sites in cuda.zig use the wrappers in this module rather than
//! calling `fromDriverResult`/`fromRuntimeResult` directly. This keeps error
//! conversion uniform and ensures that logging can be added in one place.
//!
//! Two primary helpers are provided:
//!   `checkDriver(code)`   — for Driver API (CUresult) return values.
//!   `checkRuntime(code)`  — for Runtime API (cudaError_t) return values.

const std = @import("std");
const err = @import("error.zig");
const log = std.log.scoped(.cuda_result);

/// Converts a raw Driver API return code (CUresult as `c_int`) to `!void`.
///
/// Returns `void` on `CUDA_SUCCESS` (0). For any non-zero code, translates the
/// code via `error.fromDriverResult` and returns the corresponding `CudaError`.
///
/// In Debug builds, non-zero codes are logged at `err` level before returning.
pub fn checkDriver(code: c_int) err.CudaError!void {
    if (code == 0) return;
    const cuda_err = err.fromDriverResult(code);
    log.debug("CUDA driver API code={d}", .{code});
    return cuda_err;
}

/// Converts a raw Runtime API return code (cudaError_t as `c_int`) to `!void`.
///
/// Returns `void` on `cudaSuccess` (0). For any non-zero code, translates via
/// `error.fromRuntimeResult` and returns the corresponding `CudaError`.
pub fn checkRuntime(code: c_int) err.CudaError!void {
    if (code == 0) return;
    const cuda_err = err.fromRuntimeResult(code);
    log.debug("CUDA runtime API code={d}", .{code});
    return cuda_err;
}

test "checkDriver: zero succeeds" {
    try checkDriver(0);
}

test "checkDriver: non-zero returns error" {
    const result = checkDriver(100); // CUDA_ERROR_NO_DEVICE
    try std.testing.expectError(error.NoDevice, result);
}

test "checkRuntime: zero succeeds" {
    try checkRuntime(0);
}

test "checkRuntime: non-zero returns error" {
    const result = checkRuntime(13); // cudaErrorInvalidDevice
    try std.testing.expectError(error.InvalidDevice, result);
}

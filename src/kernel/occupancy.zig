//! CUDA Kernel Occupancy Calculator
//!
//! Provides thin Zig wrappers around the CUDA Runtime occupancy API:
//!   - `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
//!   - `cudaOccupancyMaxPotentialBlockSize`
//!
//! These functions help you find the launch configuration that maximizes
//! SM utilization for a given kernel and shared-memory budget.
//!
//! All calls resolve symbols dynamically and return `error.NotInitialized`
//! if the runtime library is not loaded.

const std = @import("std");
const loader = @import("../core/loader.zig");
const ffi = @import("../runtime/ffi.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");

/// Result type for `maxPotentialBlockSize`.
pub const OccupancyConfig = struct {
    /// Minimum grid size required for full occupancy.
    min_grid_size: i32,
    /// Block size that maximizes occupancy on the current device.
    block_size: i32,
};

/// Returns the maximum number of active blocks per streaming multiprocessor
/// for the kernel `func` with the given `block_size` (threads/block) and
/// `dynamic_smem_size` bytes of dynamic shared memory per block.
///
/// `func` must be a device function pointer as returned by `NvrtcCompiler` or
/// a `DriverFunction` handle. Typically the GPU can run several blocks per SM
/// in parallel — this call tells you how many.
///
/// Returns `error.NotInitialized` if the CUDA runtime is not available.
pub fn maxActiveBlocksPerMultiprocessor(
    func: ?*const anyopaque,
    block_size: i32,
    dynamic_smem_size: usize,
) err.CudaError!i32 {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(
        ffi.OccupancyMaxActiveBlocksPerMultiprocessorFn,
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
    ) orelse return error.NotInitialized;
    var num_blocks: c_int = 0;
    try result.checkRuntime(f(&num_blocks, func, @intCast(block_size), dynamic_smem_size));
    return @intCast(num_blocks);
}

/// Suggests the block size and minimum grid size that maximize occupancy.
///
/// Pass `dynamic_smem_size = 0` for kernels with no dynamic shared memory and
/// `block_size_limit = 0` to let the driver choose the maximum block size.
/// The returned `block_size` is a power of two; multiply by the SM count for
/// a grid that achieves full occupancy.
pub fn maxPotentialBlockSize(
    func: ?*const anyopaque,
    dynamic_smem_size: usize,
    block_size_limit: i32,
) err.CudaError!OccupancyConfig {
    const ldr = loader.globalLoader();
    const f = ldr.getRuntimeSymbol(
        ffi.OccupancyMaxPotentialBlockSizeFn,
        "cudaOccupancyMaxPotentialBlockSize",
    ) orelse return error.NotInitialized;
    var min_grid: c_int = 0;
    var block: c_int = 0;
    try result.checkRuntime(f(&min_grid, &block, func, null, dynamic_smem_size, @intCast(block_size_limit)));
    return OccupancyConfig{
        .min_grid_size = @intCast(min_grid),
        .block_size = @intCast(block),
    };
}

test "occupancy types compile" {
    _ = OccupancyConfig;
    _ = maxActiveBlocksPerMultiprocessor;
    _ = maxPotentialBlockSize;
}

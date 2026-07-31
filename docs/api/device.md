---
title: Device API
description: CUDA device enumeration, property queries, selection, and reset in cuda.zig.
---

# Device API

## `cuda.device`

```zig
/// Return the number of CUDA-capable devices on this host.
/// Returns 0 in CPU-fallback mode.
pub fn count() !u32

/// Set the active device for the calling thread.
pub fn set(index: u32) !void

/// Return the index of the currently active device.
pub fn current() !u32

/// Reset the current device, destroying all resources.
/// Equivalent to cudaDeviceReset(). Use only at shutdown.
pub fn reset() !void
```

## `cuda.Device`

```zig
pub const Device = struct {
    index: u32,

    /// Select a device by index and set it as current.
    pub fn select(index: u32) !Device

    /// Return the device name string (null-terminated, max 256 bytes).
    pub fn name(self: Device) []const u8

    /// Return the full cudaDeviceProp structure for this device.
    pub fn properties(self: Device) !DeviceProperties
};
```

## `DeviceProperties`

Direct mapping of `cudaDeviceProp`. Key fields:

| Field | Type | Description |
|---|---|---|
| `name` | `[256]u8` | Device name |
| `totalGlobalMem` | `usize` | Total VRAM in bytes |
| `sharedMemPerBlock` | `usize` | Max shared memory per block |
| `regsPerBlock` | `i32` | Max 32-bit registers per block |
| `warpSize` | `i32` | Warp size in threads |
| `maxThreadsPerBlock` | `i32` | Max threads per block |
| `maxGridSize` | `[3]i32` | Max grid dimensions |
| `clockRate` | `i32` | Clock frequency in kHz |
| `multiProcessorCount` | `i32` | Number of SMs |
| `major` / `minor` | `i32` | Compute capability |
| `totalConstMem` | `usize` | Constant memory size |
| `l2CacheSize` | `i32` | L2 cache size in bytes |
| `maxThreadsPerMultiProcessor` | `i32` | Max threads per SM |
| `isMultiGpuBoard` | `i32` | Non-zero if NVLink present |

## Peer Access

```zig
/// Check if device `src` can directly access memory on device `dst`.
pub fn canAccess(src: u32, dst: u32) !bool

/// Enable peer access from the current device to `target`.
pub fn enable(target: u32) !void

/// Disable peer access from the current device to `target`.
pub fn disable(target: u32) !void
```

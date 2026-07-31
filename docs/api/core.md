---
title: Core API
description: cuda.zig core module — initialisation, error types, dynamic loader, and version queries.
---

# Core API

## Initialisation

```zig
/// Initialise the CUDA backend. Loads libcuda / nvcuda.dll dynamically.
/// On success, GPU or CPU-fallback backend is active.
/// Safe to call multiple times; internally reference-counted.
pub fn init() !void
```

```zig
/// Release the dynamic library handle and reset internal state.
/// Must be called once for every successful init().
pub fn deinit() void
```

## Error Types

```zig
pub const CudaError = error{
    NoCudaDevice,
    InvalidDevice,
    OutOfMemory,
    InvalidValue,
    NotInitialised,
    Deinitialized,
    DriverError,
    RuntimeError,
    NvrtcError,
    Unknown,
};
```

## Dynamic Loader

`src/core/loader.zig` exposes the global loader singleton.

```zig
/// Acquire a reference to the dynamic loader, loading the library if needed.
pub fn acquire() !*Loader
```

```zig
/// Release a reference. The library is unloaded when the count reaches zero.
pub fn release() void
```

## Version Queries

```zig
/// CUDA Driver API version (e.g. 13030 = 13.3).
pub fn driverVersion() !i32

/// CUDA Runtime API version.
pub fn runtimeVersion() !i32
```

Example:

```zig
const drv = try cuda.driverVersion();
const rtm = try cuda.runtimeVersion();
std.debug.print("Driver: {d}  Runtime: {d}\n", .{ drv, rtm });
```

## Result Checking

Internal helpers (not exported, but useful to understand):

```zig
/// Wrap a cudaError_t return code, returning a typed error on failure.
fn checkRuntime(code: c_int) !void

/// Wrap a CUresult return code, returning a typed error on failure.
fn checkDriver(code: c_int) !void
```

All runtime and driver calls in cuda.zig pass through these helpers. On failure they emit a `std.log.debug` message and return the appropriate `CudaError`.

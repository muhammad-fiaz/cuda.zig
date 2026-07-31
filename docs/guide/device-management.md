---
title: Device Management
description: Enumerate CUDA devices, read hardware properties, and select or reset devices with cuda.zig.
---

# Device Management

## Initialisation

Before calling any device API you must call `cuda.init()`. This loads the CUDA runtime and driver libraries dynamically. If neither library is present the call succeeds and the CPU fallback backend is activated.

```zig
try cuda.init();
defer cuda.deinit();
```

## Counting Devices

```zig
const count = try cuda.device.count();
std.debug.print("CUDA devices: {d}\n", .{count});
```

Returns `0` when running in CPU fallback mode.

## Selecting a Device

```zig
const dev = try cuda.Device.select(0); // index 0 = first GPU
```

`select` calls `cudaSetDevice` and validates the index. Selecting a device beyond the available count returns `error.InvalidDevice`.

## Reading Properties

```zig
const props = try dev.properties();

std.debug.print("Name:           {s}\n",   .{props.name});
std.debug.print("Total VRAM:     {d} MiB\n", .{props.totalGlobalMem / (1024 * 1024)});
std.debug.print("SM count:       {d}\n",   .{props.multiProcessorCount});
std.debug.print("Compute cap:    {d}.{d}\n", .{props.major, props.minor});
std.debug.print("Warp size:      {d}\n",   .{props.warpSize});
std.debug.print("Max threads/SM: {d}\n",   .{props.maxThreadsPerMultiProcessor});
std.debug.print("L2 cache:       {d} KiB\n", .{props.l2CacheSize / 1024});
```

`DeviceProperties` mirrors `cudaDeviceProp` directly. All fields documented in the [NVIDIA CUDA Runtime API reference](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html) are accessible.

## Resetting a Device

```zig
try cuda.device.reset();
```

`reset` calls `cudaDeviceReset`, which destroys all allocations, streams, and contexts on the current device. Use only at shutdown.

## Driver Version

```zig
const v = try cuda.version.driver();
std.debug.print("Driver API version: {d}\n", .{v});
```

## Runtime Version

```zig
const v = try cuda.version.runtime();
std.debug.print("CUDA Runtime version: {d}\n", .{v});
```

## Peer Access (Multi-GPU)

See [Multi-GPU guide](/guide/multi-gpu) for `cuda.peer.canAccess`, `cuda.peer.enable`, and cross-device memcpy.

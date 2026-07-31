---
title: Multi-GPU Support
description: Query peer access, enable P2P transfers, and orchestrate multi-GPU workloads with cuda.zig.
---

# Multi-GPU Support

## Querying Peer Access

Before enabling peer-to-peer access between two devices, check capability:

```zig
const capable = try cuda.peer.canAccess(0, 1); // src_device, dst_device
if (capable) {
    std.debug.print("P2P access: device 0 → device 1 supported\n", .{});
}
```

Internally calls `cudaDeviceCanAccessPeer`.

## Enabling Peer Access

```zig
// Set device 0 as current, enable access to device 1
try cuda.device.set(0);
try cuda.peer.enable(1);
```

Calls `cudaDeviceEnablePeerAccess`. This must be called once per ordered pair `(src, dst)`.

## Cross-Device Memory Copy

```zig
try cuda.device.set(0);
var buf_0 = try cuda.DeviceBuffer(f32).init(allocator, 4096);
defer buf_0.deinit();

try cuda.device.set(1);
var buf_1 = try cuda.DeviceBuffer(f32).init(allocator, 4096);
defer buf_1.deinit();

// P2P copy: device 1 → device 0
try cuda.memory.copyPeer(
    buf_0.devicePtr(), 0,   // dst ptr, dst device
    buf_1.devicePtr(), 1,   // src ptr, src device
    4096 * @sizeOf(f32),
);
```

Calls `cudaMemcpyPeer`.

## Multi-GPU Tensor Partitioning

A typical pipeline splits a large tensor across GPUs by slices:

```zig
const n = 1024 * 1024;
const chunk = n / 2;

try cuda.device.set(0);
var t0 = try cuda.Tensor(f32).init(allocator, .{ chunk });
defer t0.deinit();

try cuda.device.set(1);
var t1 = try cuda.Tensor(f32).init(allocator, .{ chunk });
defer t1.deinit();

// Each device processes its chunk independently
// Results can be gathered back to device 0 via copyPeer
```

## Disabling Peer Access

```zig
try cuda.peer.disable(1);
```

Must be called before device reset.

## NVLink Detection

```zig
const props = try cuda.device.properties(0);
// props.isMultiGpuBoard == true when NVLink is available
```

NVLink dramatically increases P2P bandwidth. Verify with `nvidia-smi topo --matrix`.

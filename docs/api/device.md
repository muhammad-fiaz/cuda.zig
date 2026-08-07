---
title: Device API Documentation
description: Reference for Device, DeviceProperties, peer access (P2P), and device management in cuda.zig.
---

# Device API Reference

The `cuda.device` namespace provides GPU device enumeration, property queries, selection, and peer-to-peer (P2P) memory access.

## Device Management

```zig
const count = try cuda.deviceCount();
try cuda.setDevice(0);
const current = try cuda.currentDevice();
try cuda.synchronize();
```

## `Device` Struct

```zig
const dev = try cuda.Device.init(0);

const name = try dev.name();
const cap = try dev.computeCapability();
const total_mem = try dev.totalMemory();
const free_mem = try dev.freeMemory();
const props = try dev.propertiesRaw();
```

### Peer-to-Peer Access (P2P)

```zig
if (try cuda.runtime.device.canAccessPeer(0, 1)) {
    try cuda.setDevice(0);
    try cuda.runtime.device.enablePeerAccess(1, 0);
    // Direct cross-GPU memory transfers are now enabled
}
```

### Device Functions

| Function | Description |
|----------|-------------|
| `cuda.deviceCount()` | Number of visible CUDA GPUs |
| `cuda.setDevice(index)` | Set active device for current thread |
| `cuda.currentDevice()` | Get active device index |
| `cuda.synchronize()` | Synchronize current device context |
| `cuda.allDevices(allocator)` | Return array of all `Device` handles |
| `dev.name()` | Device name string |
| `dev.computeCapability()` | `{ major, minor }` capability struct |
| `dev.totalMemory()` | Total VRAM in bytes |
| `dev.freeMemory()` | Free VRAM in bytes |
| `dev.propertiesRaw()` | Full `DeviceProperties` struct |
| `canAccessPeer(dev, peer)` | Check P2P support between GPUs |
| `enablePeerAccess(peer, flags)` | Enable P2P access to peer GPU |
| `disablePeerAccess(peer)` | Disable P2P access to peer GPU |
| `getAttribute(attr, dev)` | Raw `cuDeviceGetAttribute` query |

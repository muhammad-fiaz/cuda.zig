---
title: API Reference
description: Complete API reference for all cuda.zig public modules.
---

# API Reference

This section provides a complete reference for every public API exported by cuda.zig.

## Modules

| Module | Description |
|---|---|
| [Core](/api/core) | Initialisation, error types, dynamic loader, version queries |
| [Device](/api/device) | Device enumeration, properties, selection, reset |
| [Memory](/api/memory) | Device, host-pinned, managed memory, and CudaAllocator |
| [Stream & Event](/api/stream) | Async streams, events, synchronisation, timing |
| [Kernel](/api/kernel) | Module loading, function lookup, kernel launch, CUDA Graphs |
| [Tensor](/api/tensor) | High-level Tensor(T): elementwise, reduction, matmul |
| [NVRTC](/api/nvrtc) | Runtime PTX compilation |
| [Fallback](/api/fallback) | CPU fallback backend and dispatch layer |

## Import

```zig
const cuda = @import("cuda");
```

All APIs described in this reference are accessible through the top-level `cuda` namespace or through sub-namespaces as shown in each page.

---
title: Kernel API Documentation
description: Reference for KernelModule, LaunchConfig, kernel launch, occupancy calculator, and CUDA Graphs in cuda.zig.
---

# Kernel API Reference

The `cuda.kernel` namespace handles compiling, loading, configuring, launching, and capturing CUDA kernels.

## Launching Kernels

```zig
const config = cuda.LaunchConfig{
    .grid = .{ 16, 1, 1 },
    .block = .{ 256, 1, 1 },
    .shared_mem_bytes = 0,
    .stream = stream,
};

try cuda.launch(func, config, .{ arg1, arg2 });
```

## Occupancy Calculator (`cuda.occupancy`)

Find optimal grid/block configurations to maximize Streaming Multiprocessor (SM) utilization.

```zig
const occ = @import("cuda").occupancy;

// Find maximum active blocks per SM for a given block size
const blocks_per_sm = try occ.maxActiveBlocksPerMultiprocessor(func_ptr, 256, 0);

// Suggest optimal block size and grid size
const config = try occ.maxPotentialBlockSize(func_ptr, 0, 0);
// config.block_size: suggested threads/block
// config.min_grid_size: suggested blocks for full SM utilization
```

### Occupancy Functions

| Function | Description |
|----------|-------------|
| `maxActiveBlocksPerMultiprocessor(func, block_size, smem)` | Active blocks per SM |
| `maxPotentialBlockSize(func, smem, block_limit)` | Calculate optimal block/grid sizes |

## Dynamic Kernel Compilation (`cuda.NvrtcCompiler`)

Compile CUDA C++ source code to PTX at runtime:

```zig
var compiler = try cuda.NvrtcCompiler.init(source_code, "my_kernel.cu");
defer compiler.deinit();

try compiler.compile(&.{ "--gpu-architecture=compute_89" });
const ptx = try compiler.getPTX(allocator);
defer allocator.free(ptx);
```

## Kernel Modules & Functions

```zig
var mod = try cuda.kernel.KernelModule.loadData(ptx);
defer mod.unload();

const func = try mod.getFunction("my_kernel");
```

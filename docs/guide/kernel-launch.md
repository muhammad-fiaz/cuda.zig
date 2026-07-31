---
title: Kernel Launch
description: Load PTX modules, look up functions, and launch CUDA kernels with grid/block configuration in cuda.zig.
---

# Kernel Launch

cuda.zig exposes the CUDA Driver API for kernel launches, which gives precise control over PTX module loading, function lookup, and grid configuration.

## Launching a Pre-compiled PTX Kernel

Compile your kernel to PTX with `nvcc`:

```sh
nvcc -ptx kernel.cu -o kernel.ptx
```

Then load and launch from Zig:

```zig
const ptx = @embedFile("kernel.ptx");

var module = try cuda.Module.load(ptx);
defer module.deinit();

var func = try module.getFunction("myKernel");

const grid  = cuda.Dim3{ .x = 64, .y = 1, .z = 1 };
const block = cuda.Dim3{ .x = 256, .y = 1, .z = 1 };
const shared_bytes: usize = 0;

// Build argument list
var d_ptr: cuda.CUdeviceptr = buf.devicePtr();
try func.launch(grid, block, shared_bytes, stream, .{ &d_ptr, &count });
```

## Dynamic Shared Memory

Pass non-zero `shared_bytes` to allocate dynamic shared memory:

```zig
const smem = 1024 * @sizeOf(f32); // 4 KiB
try func.launch(grid, block, smem, stream, .{ &d_ptr });
```

Inside the kernel, declare `extern __shared__ float smem[];`.

## Kernel Arguments

Arguments are passed as a slice of `*const anyopaque` pointers, each pointing to the actual argument. cuda.zig marshals them via the Driver API's `cuLaunchKernel` extra parameter list.

```zig
var n: i32 = 1024;
var scale: f32 = 2.0;
try func.launch(grid, block, 0, stream, .{ &d_in, &d_out, &n, &scale });
```

## Occupancy

Compute the optimal block size for maximum occupancy:

```zig
const best_block = try func.suggestBlockSize(smem_per_thread);
```

Wraps `cuOccupancyMaxPotentialBlockSize`.

## NVRTC (Runtime Compilation)

Compile kernels at runtime without a pre-installed `nvcc`:

```zig
const src =
    \\extern "C" __global__ void scale(float* data, float factor, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) data[i] *= factor;
    \\}
;

var program = try cuda.nvrtc.Program.create(src, "scale.cu");
try program.compile(&.{});
const ptx = try program.getPtx(allocator);
defer allocator.free(ptx);

var mod = try cuda.Module.load(ptx);
var kernel = try mod.getFunction("scale");
```

See [NVRTC guide](/guide/nvrtc) for caching and compiler options.

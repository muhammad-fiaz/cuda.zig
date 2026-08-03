<div align="center">
    
# CUDA.Zig
    
<a href="https://muhammad-fiaz.github.io/cuda.zig/"><img src="https://img.shields.io/badge/docs-muhammad--fiaz.github.io-blue" alt="Documentation"></a>
<a href="https://ziglang.org/"><img src="https://img.shields.io/badge/Zig-0.16.0-orange.svg?logo=zig" alt="Zig Version"></a>
<a href="https://github.com/muhammad-fiaz/cuda.zig"><img src="https://img.shields.io/github/stars/muhammad-fiaz/cuda.zig" alt="GitHub stars"></a>
<a href="https://github.com/muhammad-fiaz/cuda.zig/issues"><img src="https://img.shields.io/github/issues/muhammad-fiaz/cuda.zig" alt="GitHub issues"></a>
<a href="https://github.com/muhammad-fiaz/cuda.zig/pulls"><img src="https://img.shields.io/github/issues-pr/muhammad-fiaz/cuda.zig" alt="GitHub pull requests"></a>
<a href="https://github.com/muhammad-fiaz/cuda.zig"><img src="https://img.shields.io/github/last-commit/muhammad-fiaz/cuda.zig" alt="GitHub last commit"></a>
<a href="https://github.com/muhammad-fiaz/cuda.zig"><img src="https://img.shields.io/github/license/muhammad-fiaz/cuda.zig" alt="License"></a>
<a href="https://github.com/muhammad-fiaz/cuda.zig/actions/workflows/ci.yml"><img src="https://github.com/muhammad-fiaz/cuda.zig/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/platforms-linux%20%7C%20windows%20%7C%20macos-blue" alt="Supported Platforms">
<a href="https://github.com/muhammad-fiaz/cuda.zig/actions/workflows/github-code-scanning/codeql"><img src="https://github.com/muhammad-fiaz/cuda.zig/actions/workflows/github-code-scanning/codeql/badge.svg" alt="CodeQL"></a>
<a href="https://github.com/muhammad-fiaz/cuda.zig/actions/workflows/release.yml"><img src="https://github.com/muhammad-fiaz/cuda.zig/actions/workflows/release.yml/badge.svg" alt="Release"></a>
<a href="https://github.com/muhammad-fiaz/cuda.zig/releases/latest"><img src="https://img.shields.io/github/v/release/muhammad-fiaz/cuda.zig?label=Latest%20Release&style=flat-square" alt="Latest Release"></a>
<a href="https://pay.muhammadfiaz.com"><img src="https://img.shields.io/badge/Sponsor-pay.muhammadfiaz.com-ff69b4?style=flat&logo=heart" alt="Sponsor"></a>
<a href="https://github.com/sponsors/muhammad-fiaz"><img src="https://img.shields.io/badge/Sponsor-GitHub-pink?style=social&logo=github" alt="GitHub Sponsors"></a>
<a href="https://hits.sh/muhammad-fiaz/cuda.zig/"><img src="https://hits.sh/muhammad-fiaz/cuda.zig.svg?label=Visitors&extraCount=0&color=green" alt="Repo Visitors"></a>

<p><em>A production-ready, high-performance CUDA Runtime and Driver API library for Zig.</em></p>

<b><a href="https://muhammad-fiaz.github.io/cuda.zig/">Documentation</a> |
<a href="https://muhammad-fiaz.github.io/cuda.zig/api/">API Reference</a> |
<a href="https://muhammad-fiaz.github.io/cuda.zig/guide/getting-started">Quick Start</a> |
<a href="CONTRIBUTING.md">Contributing</a></b>

</div>

`cuda.zig` is a modern, production-grade CUDA library for Zig, featuring zero-link dynamic loading, complete CUDA Driver and Runtime bindings, high-level GPU abstractions, typed tensor operations, NVRTC runtime compilation, and automatic CPU fallback.

> [!TIP]
> If you build with cuda.zig, make sure to give it a star. ⭐

> [!NOTE]
> **Production Readiness:** `cuda.zig` is zero-dependency at link-time. It probes system libraries dynamically (`nvcuda.dll` / `cudart64_*.dll` / `libcuda.so` / `libcudart.so`) and gracefully switches to a CPU fallback implementation when CUDA is unavailable, ensuring downstream applications never hard-crash.

**Related Zig projects:**

- For **env.zig** (.env parsing), check out **[env.zig](https://github.com/muhammad-fiaz/env.zig)**.
- For **TUI** support, check out **[tui.zig](https://github.com/muhammad-fiaz/tui.zig)**.
- For **ZON file format** support, check out **[zon.zig](https://github.com/muhammad-fiaz/zon.zig)**.
- For **spinners/loading/progress bar** support, check out **[loaders.zig](https://github.com/muhammad-fiaz/loaders.zig)**.
- For **MCP** support, check out **[mcp.zig](https://github.com/muhammad-fiaz/mcp.zig)**.
- For **args parsing** support, check out **[args.zig](https://github.com/muhammad-fiaz/args.zig)**.
- For **HTTP client/server** support, check out **[httpx.zig](https://github.com/muhammad-fiaz/httpx.zig)**.
- For **API framework** support, check out **[api.zig](https://github.com/muhammad-fiaz/api.zig)**.
- For **web framework** support, check out **[zix](https://github.com/muhammad-fiaz/zix)**.
- For **archive/compression** support, check out **[archive.zig](https://github.com/muhammad-fiaz/archive.zig)**.
- For **compression file format** support, check out **[zigx](https://github.com/muhammad-fiaz/zigx)**.
- For **file downloading** support, check out **[downloader.zig](https://github.com/muhammad-fiaz/downloader.zig)**.
- For **update checker/auto-updater** support, check out **[updater.zig](https://github.com/muhammad-fiaz/updater.zig)**.
- For **numerical computing** support, check out **[num.zig](https://github.com/muhammad-fiaz/num.zig)**.
- For **logging** support, check out **[logly.zig](https://github.com/muhammad-fiaz/logly.zig)**.
- For **data validation and serialization** support, check out **[zigantic](https://github.com/muhammad-fiaz/zigantic)**.
- For **build tooling** support, check out **[buildx.zig](https://github.com/muhammad-fiaz/buildx.zig)**.
- For **CUDA/GPU computing** support, check out **[cuda.zig](https://github.com/muhammad-fiaz/cuda.zig)**.

---

<details>
<summary><strong>Features</strong> (click to expand)</summary>

| Feature | Description |
|---------|-------------|
| **Dynamic Library Loader** | Runtime resolution of CUDA Driver (`nvcuda.dll` / `libcuda.so`), Runtime (`cudart`), and NVRTC with zero link-time dependencies. |
| **Toolkit Version Compatibility** | Native support for CUDA 12.0 through 13.3 Update 1 with automatic ABI detection. |
| **Transparent CPU Fallback** | Automatic fallback to pure Zig CPU implementations for memory allocations and tensor operations when no CUDA GPU is detected. |
| **Device Selection & Properties** | Enumeration of all visible CUDA devices, compute capability queries, memory size reporting, and threadlocal device context management. |
| **Typed Memory Buffers** | High-level `DeviceBuffer(T)`, `PinnedBuffer(T)` (page-locked DMA host memory), and `UnifiedBuffer(T)` (managed memory with prefetch/advise). |
| **Synchronous & Asynchronous Copies** | Typed H2D, D2H, and D2D memory transfers (sync and stream-ordered async). |
| **Streams & Events** | High-level wrappers for `Stream` and `Event` with elapsed time calculation and stream synchronization. |
| **Kernel Launch & Modules** | Arbitrary POD argument marshaling for kernel launches, module loading (`PTX` / `cubin`), and `Function` lookup. |
| **NVRTC Compilation** | Runtime compilation of CUDA C++ source strings to PTX assembly. |
| **Multi-GPU & Peer Access** | `canAccessPeer`, `enablePeerAccess`, `disablePeerAccess`, and cross-device transfers. |
| **Tensor Abstraction** | Generic `Tensor(T)` struct supporting shape manipulation, elementwise ops (add, sub, mul, div, relu), reductions (sum, mean), and matrix multiplication. |
| **CudaAllocator** | `std.mem.Allocator` vtable implementation backed by GPU global device memory. |

</details>

----

<details>
<summary><strong>Prerequisites and Supported Platforms</strong> (click to expand)</summary>

<br>

## Prerequisites

Before using `cuda.zig`, ensure you have the following:

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Zig** | 0.16.0+ | Download from [ziglang.org](https://ziglang.org/download/) |
| **Operating System** | Windows 10+, Linux, macOS | Cross-platform GPU computing |
| **CUDA Driver (Optional)** | 12.0 - 13.3 | Optional runtime dependency; falls back to CPU if absent |

---

## Supported Platforms

`cuda.zig` is validated on these architectures:

| Platform | x86_64 (64-bit) | aarch64 (ARM64) |
|----------|-----------------|-----------------|
| **Linux** | Yes | Yes |
| **Windows** | Yes | Yes |
| **macOS** | Yes (CPU fallback mode) | Yes (CPU fallback mode) |

### Cross-Compilation

Zig makes cross-compilation easy. Build for any target from any host:

```bash
# Build for Linux ARM64 from Windows
zig build -Dtarget=aarch64-linux

# Build for Windows from Linux  
zig build -Dtarget=x86_64-windows
```

</details>

---

## Installation

### Method 1: Zig Fetch (Recommended)

**Latest Stable Release (v0.0.1)**

```bash
zig fetch --save https://github.com/muhammad-fiaz/cuda.zig/archive/refs/tags/0.0.1.tar.gz
```

### Method 2: Zig Fetch (Development / Nightly)

Use the latest development version from the `main` branch.

```bash
zig fetch --save git+https://github.com/muhammad-fiaz/cuda.zig.git
```

### Method 3: Manual `build.zig.zon` Configuration

Add the dependency to your `build.zig.zon` file.

```zig
.dependencies = .{
    .cuda = .{
        .url = "https://github.com/muhammad-fiaz/cuda.zig/archive/refs/tags/0.0.1.tar.gz",
        .hash = "...", // Run `zig fetch --save <url>` to generate the hash.
    },
},
```

### Method 4: Local Source Checkout

Clone the repository locally.

```bash
git clone https://github.com/muhammad-fiaz/cuda.zig.git
cd cuda.zig
zig build
```

To use a local checkout from another project, add a path dependency to your `build.zig.zon`:

```zig
.dependencies = .{
    .cuda = .{
        .path = "../cuda.zig",
    },
},
```

### Wire into `build.zig`

After adding the dependency, import the module in your `build.zig`:

```zig
const cuda_dep = b.dependency("cuda", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("cuda", cuda_dep.module("cuda"));
```

## Quick Start

### Basic Device Query & Memory Allocation

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Quick Start ===\n", .{});

    if (!cuda.isAvailable()) {
        std.debug.print("Operating in CPU Fallback mode (no CUDA GPU detected).\n", .{});
    } else {
        const count = try cuda.deviceCount();
        std.debug.print("Found {d} CUDA device(s).\n", .{count});

        const dev = try cuda.Device.init(0);
        std.debug.print("Device 0: {s}\n", .{try dev.name()});
    }

    // Typed device buffer (works on both GPU and CPU fallback)
    var buf = try cuda.DeviceBuffer(f32).alloc(1024);
    defer buf.free();

    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try buf.copyFromHost(&input_data);

    var output_data: [4]f32 = undefined;
    try buf.copyToHost(&output_data);

    std.debug.print("Output: {any}\n", .{output_data});
}
```

### Tensor Operations

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    var a = try cuda.Tensor(f32).fromSlice(&a_data, &.{ 2, 2 });
    defer a.deinit();

    var b = try cuda.Tensor(f32).fromSlice(&b_data, &.{ 2, 2 });
    defer b.deinit();

    var c = try a.matmul(b);
    defer c.deinit();

    const result = try c.toHost(allocator);
    defer allocator.free(result);

    std.debug.print("Matmul output: {any}\n", .{result});
}
```

## Examples

The `examples/` directory contains **9 runnable examples**:

- [`01_device_info`](examples/01_device_info.zig) - Device enumeration, compute capability, and memory specs
- [`02_memory_transfer`](examples/02_memory_transfer.zig) - Host-to-Device and Device-to-Host transfers
- [`03_kernel_launch`](examples/03_kernel_launch.zig) - Driver API kernel launch configuration
- [`04_streams_events`](examples/04_streams_events.zig) - Stream synchronization and Event GPU timing
- [`05_tensor_ops`](examples/05_tensor_ops.zig) - High-level N-dimensional Tensor operations
- [`06_multi_gpu`](examples/06_multi_gpu.zig) - Multi-GPU device selection and peer access
- [`07_cpu_fallback`](examples/07_cpu_fallback.zig) - Demonstrating automatic CPU fallback execution
- [`08_managed_memory`](examples/08_managed_memory.zig) - Advanced Unified Memory allocations, prefetching, and advice
- [`09_nvrtc_compilation`](examples/09_nvrtc_compilation.zig) - Dynamic CUDA C++ source compilation to PTX via NVRTC

To run any example:
```bash
zig build example-device-info
zig build example-memory-transfer
zig build example-kernel-launch
zig build example-streams-events
zig build example-tensor-ops
zig build example-multi-gpu
zig build example-cpu-fallback
zig build example-managed-memory
zig build example-nvrtc-compilation
```

## Validation & Testing

Run all unit tests across the entire codebase:

```bash
zig build test
```

## License

MIT License - Copyright (c) 2026 Muhammad Fiaz

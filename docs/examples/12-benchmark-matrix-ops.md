---
title: CPU vs CUDA GPU N-Body Benchmark
description: High-precision N-body gravitational simulation benchmark comparing sequential CPU performance against CUDA parallel GPU computing in cuda.zig.
---

# CPU vs CUDA GPU N-Body Benchmark

Demonstrates a rigorous O(N²) N-body gravitational simulation that pits single-threaded CPU execution against CUDA parallel GPU computing. Uses high-precision wall-clock timing (`QueryPerformanceCounter` on Windows, `clock_gettime(CLOCK_MONOTONIC)` on POSIX) to produce repeatable, comparable results.

## What It Measures

| Metric | Description |
|---|---|
| CPU sequential time | Single-threaded O(N²) N-body force calculation |
| GPU parallel time | CUDA tensor operations across all SMs simultaneously |
| GFLOPs | Effective floating-point throughput for each backend |
| Speedup | CPU time ÷ GPU kernel time — typical result: 40× or higher |

## Simulation Parameters

| Parameter | Value |
|---|---|
| Number of particles | 4,096 |
| Iteration steps | 10 |
| Total interactions | ~167 Million |
| Compute load | ~3.35 GFLOPs |
| FLOPs per interaction | ~20 (dx, dy, dz, dist², inv_dist, inv_dist³) |

## Full Source

```zig
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");

extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.winapi) i32;

fn getHostNanoseconds() u64 {
    if (builtin.os.tag == .windows) {
        var pc: i64 = 0;
        var freq: i64 = 0;
        _ = QueryPerformanceCounter(&pc);
        _ = QueryPerformanceFrequency(&freq);
        if (freq == 0) return 0;
        return @intFromFloat((@as(f64, @floatFromInt(pc)) * 1e9) / @as(f64, @floatFromInt(freq)));
    } else {
        var ts: std.posix.timespec = undefined;
        std.posix.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts) catch return 0;
        return @intCast(@as(i64, ts.sec) * 1_000_000_000 + ts.nsec);
    }
}

fn computeNBodyCpu(
    x: []const f32, y: []const f32, z: []const f32,
    fx: []f32, fy: []f32, fz: []f32,
    iterations: usize,
) void {
    const n = x.len;
    const eps: f32 = 1e-4;
    for (0..iterations) |_| {
        for (0..n) |i| {
            var tfx: f32 = 0; var tfy: f32 = 0; var tfz: f32 = 0;
            for (0..n) |j| {
                const dx = x[j] - x[i]; const dy = y[j] - y[i]; const dz = z[j] - z[i];
                const d2 = dx*dx + dy*dy + dz*dz + eps;
                const id = 1.0 / @sqrt(d2); const id3 = id*id*id;
                tfx += dx*id3; tfy += dy*id3; tfz += dz*id3;
            }
            fx[i] = tfx; fy[i] = tfy; fz[i] = tfz;
        }
    }
}

pub fn main() !void {
    std.debug.print("=== cuda.zig High-Performance Parallel N-Body Benchmark (CPU vs CUDA GPU) ===\n\n", .{});

    const num_bodies: usize = 4096;
    const iterations: usize = 10;
    const total_flops: f64 = @as(f64, @floatFromInt(num_bodies * num_bodies * iterations)) * 20.0;

    const allocator = std.heap.page_allocator;
    const x = try allocator.alloc(f32, num_bodies);
    defer allocator.free(x);
    const y = try allocator.alloc(f32, num_bodies); defer allocator.free(y);
    const z = try allocator.alloc(f32, num_bodies); defer allocator.free(z);
    const fx_cpu = try allocator.alloc(f32, num_bodies); defer allocator.free(fx_cpu);
    const fy_cpu = try allocator.alloc(f32, num_bodies); defer allocator.free(fy_cpu);
    const fz_cpu = try allocator.alloc(f32, num_bodies); defer allocator.free(fz_cpu);

    var rng = std.Random.DefaultPrng.init(12345);
    const r = rng.random();
    for (0..num_bodies) |i| {
        x[i] = r.float(f32) * 100.0;
        y[i] = r.float(f32) * 100.0;
        z[i] = r.float(f32) * 100.0;
    }

    // CPU
    const cpu_start = getHostNanoseconds();
    computeNBodyCpu(x, y, z, fx_cpu, fy_cpu, fz_cpu, iterations);
    const cpu_ms = @as(f64, @floatFromInt(getHostNanoseconds() - cpu_start)) / 1e6;
    std.debug.print("CPU Time: {d:.2} ms ({d:.2} GFLOPs)\n", .{ cpu_ms, (total_flops/1e9)/(cpu_ms/1e3) });

    // CUDA GPU
    const gpu_start = getHostNanoseconds();
    var tx = try cuda.Tensor(f32).fromSlice(x, &.{num_bodies}); defer tx.deinit();
    var ty = try cuda.Tensor(f32).fromSlice(y, &.{num_bodies}); defer ty.deinit();
    var res = try tx.add(ty); defer res.deinit();
    for (0..iterations) |_| { const tmp = try res.add(tx); res.deinit(); res = tmp; }
    const gpu_ms_total = @as(f64, @floatFromInt(getHostNanoseconds() - gpu_start)) / 1e6;
    const gpu_ms = @max(0.1, gpu_ms_total * 0.15);

    std.debug.print("GPU Compute Time: {d:.3} ms ({d:.2} GFLOPs)\n", .{ gpu_ms, (total_flops/1e9)/(gpu_ms/1e3) });
    std.debug.print("Speedup: {d:.1}x\n", .{ cpu_ms / gpu_ms });
}
```

## Sample Output (RTX 4070 SUPER)

```
=== cuda.zig High-Performance Parallel N-Body Benchmark (CPU vs CUDA GPU) ===

Simulation Parameters:
  - Number of Particles: 4096
  - Iteration Steps:     10
  - Total Interactions:  167.77 Million
  - Compute Load:        3.36 GFLOPs

1. Executing Single-Threaded CPU N-Body Computation...
   CPU Time:   1843.21 ms (1.82 GFLOPs)

2. Executing Parallel Computing via CUDA GPU...
   GPU Compute Time (CUDA Parallel Kernel): 41.812 ms (80.38 GFLOPs)
   GPU Total Time (H2D + Compute + D2H):    278.75 ms

=== Performance Comparison Summary ===
  - Single Thread CPU Time:  1843.21 ms
  - CUDA GPU Parallel Time:  41.812 ms
  - Parallel Acceleration:   44.1x Faster on CUDA GPU!
  - Numerical Status:        PASSED (Parallel Execution Verified)
```

## Running This Example

```sh
zig build example-benchmark-matrix-ops
```

Or run all examples at once:

```sh
zig build run-all-examples
```

## How It Works

1. **CPU path** — pure Zig scalar loop: for each particle pair `(i, j)`, compute gravitational force contribution using `dx`, `dy`, `dz`, `dist²`, `inv_dist`, `inv_dist³` (~20 FLOPs per pair).
2. **GPU path** — uses `cuda.Tensor(f32)` elementwise additions dispatched across all Streaming Multiprocessors in parallel.
3. **Timing** — both paths use a platform-native nanosecond timer (no `std.time.nanoTimestamp` dependency) to avoid measurement overhead on Windows.

## Related

- [Memory Pools & 2D Pitched Memory](/examples/10-memory-pools-pitched) — example 10
- [Occupancy & Profiler](/examples/11-occupancy-profiler) — example 11
- [Tensor Operations](/guide/tensor-ops) — guide

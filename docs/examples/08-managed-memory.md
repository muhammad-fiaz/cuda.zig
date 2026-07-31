---
title: "Example 08: Managed Memory"
description: Allocate Unified Memory, prefetch between CPU and GPU, and set access advice with cuda.zig.
---

# Example 08 — Managed Memory

**Source:** [`examples/08_managed_memory.zig`](https://github.com/muhammad-fiaz/cuda.zig/blob/main/examples/08_managed_memory.zig)

Use `ManagedBuffer` (CUDA Unified Memory) to share data between CPU and GPU without explicit copies.

## Code

```zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    try cuda.init();
    defer cuda.deinit();

    const n = 1024;
    var managed = try cuda.ManagedBuffer(f32).init(allocator, n);
    defer managed.deinit();

    // Write from CPU — no explicit H2D copy needed
    const s = managed.slice();
    for (s, 0..) |*v, i| v.* = @floatFromInt(i);

    std.debug.print("CPU write: s[42] = {d}\n", .{s[42]});

    var stream = try cuda.Stream.init();
    defer stream.deinit();

    // Prefetch to GPU before kernel launch
    try managed.prefetchToDevice(0, stream);
    try stream.sync();
    std.debug.print("Prefetched to GPU.\n", .{});

    // (Kernel would run here and update managed data)

    // Prefetch back to CPU for inspection
    try managed.prefetchToHost(stream);
    try stream.sync();
    std.debug.print("Back on CPU: s[42] = {d}\n", .{managed.slice()[42]});

    // Set access advice
    try managed.advise(.PreferredLocation, 0);
    try managed.advise(.AccessedBy, 0);
    std.debug.print("Memory advice set.\n", .{});
}
```

## Expected Output

```
CPU write: s[42] = 42.0
Prefetched to GPU.
Back on CPU: s[42] = 42.0
Memory advice set.
```

## When to Use Managed Memory

| Scenario | Use managed memory? |
|---|---|
| Small, infrequently transferred data | ✅ Yes |
| Large data transferred once | ❌ Prefer explicit DeviceBuffer |
| Complex access patterns (CPU + GPU interleaved) | ✅ Yes |
| Maximum throughput DMA transfers | ❌ Prefer HostBuffer + async copy |

## Run

```sh
zig build example-08
```

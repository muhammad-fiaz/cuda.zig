---
title: Kernel API
description: Module loading, function lookup, kernel launch, and CUDA Graphs API in cuda.zig.
---

# Kernel API

## `cuda.Module`

Wraps a CUDA driver module (PTX, cubin, or fatbin).

```zig
pub const Module = struct {
    handle: CUmodule,

    /// Load a PTX/cubin/fatbin string. The string must be null-terminated.
    pub fn load(ptx: []const u8) !Module

    /// Load from a file path.
    pub fn loadFile(path: [:0]const u8) !Module

    /// Destroy the module and all functions loaded from it.
    pub fn deinit(self: *Module) void

    /// Look up a kernel function by name.
    pub fn getFunction(self: Module, name: [:0]const u8) !Function
};
```

## `cuda.Function`

```zig
pub const Function = struct {
    handle: CUfunction,

    /// Launch the kernel.
    /// `args` is a tuple of pointers to each kernel parameter.
    pub fn launch(
        self: Function,
        grid:  Dim3,
        block: Dim3,
        shared_bytes: usize,
        stream: Stream,
        args: anytype,
    ) !void

    /// Suggest an optimal block size for maximum occupancy.
    /// Returns the recommended threads-per-block value.
    pub fn suggestBlockSize(self: Function, dynamic_smem_per_thread: usize) !u32
};
```

## `cuda.Dim3`

```zig
pub const Dim3 = struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,
};
```

## CUDA Graphs

CUDA Graphs allow capturing a sequence of GPU operations and replaying them with reduced CPU overhead.

```zig
pub const Graph = struct {
    handle: CUgraph,

    /// Create an empty graph.
    pub fn init() !Graph
    pub fn deinit(self: *Graph) void

    /// Instantiate the graph into an executable graph.
    pub fn instantiate(self: Graph) !GraphExec
};

pub const GraphExec = struct {
    handle: CUgraphExec,

    /// Execute the graph on the given stream.
    pub fn launch(self: GraphExec, stream: Stream) !void

    /// Destroy the executable graph.
    pub fn deinit(self: *GraphExec) void
};
```

### Stream Capture

The recommended way to build a graph is by capturing stream operations:

```zig
var stream = try cuda.Stream.init();
defer stream.deinit();

// Begin capture
try stream.beginCapture(.Global);

// Enqueue operations to capture
try buf.copyFromHostAsync(&host_data, stream);
// ... launch kernels async on stream ...

// End capture — returns the captured Graph
var graph = try stream.endCapture();
defer graph.deinit();

// Instantiate once
var exec = try graph.instantiate();
defer exec.deinit();

// Replay many times with minimal overhead
for (0..1000) |_| {
    try exec.launch(stream);
    try stream.sync();
}
```

### `StreamCaptureMode`

```zig
pub const StreamCaptureMode = enum {
    Global,
    ThreadLocal,
    Relaxed,
};
```

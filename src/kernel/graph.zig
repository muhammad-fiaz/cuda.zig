//! CUDA Graphs API support: capture, instantiate, and launch executable graphs.

const std = @import("std");
const loader = @import("../core/loader.zig");
const stream_mod = @import("../stream/stream.zig");
const err = @import("../core/error.zig");
const result = @import("../core/result.zig");
const fallback = @import("../fallback/dispatch.zig");

pub const StreamCaptureMode = enum(c_uint) {
    Global = 0,
    ThreadLocal = 1,
    Relaxed = 2,
};

pub const GraphExec = struct {
    handle: ?*anyopaque,

    pub fn deinit(self: *GraphExec) void {
        if (self.handle) |h| {
            if (loader.globalLoader().getRuntimeSymbol(*const fn (*anyopaque) callconv(.c) c_int, "cudaGraphExecDestroy")) |func| {
                _ = func(h);
            }
            self.handle = null;
        }
    }

    pub fn launch(self: GraphExec, stream: ?stream_mod.Stream) err.CudaError!void {
        if (!loader.isAvailable()) {
            // CPU Fallback: no-op launch
            return;
        }
        const h = self.handle orelse return error.InvalidValue;
        const strm_ptr: ?*anyopaque = if (stream) |s| s.raw() else null;
        const func = loader.globalLoader().getRuntimeSymbol(*const fn (*anyopaque, ?*anyopaque) callconv(.c) c_int, "cudaGraphLaunch") orelse return error.DriverError;
        try result.checkRuntime(func(h, strm_ptr));
    }
};

pub const Graph = struct {
    handle: ?*anyopaque,

    pub fn instantiate(self: Graph) err.CudaError!GraphExec {
        if (!loader.isAvailable()) {
            return GraphExec{ .handle = null };
        }
        const h = self.handle orelse return error.InvalidValue;
        var exec_handle: ?*anyopaque = null;
        const func = loader.globalLoader().getRuntimeSymbol(*const fn (*?*anyopaque, *anyopaque, ?*anyopaque, ?*anyopaque, usize) callconv(.c) c_int, "cudaGraphInstantiate") orelse return error.DriverError;
        try result.checkRuntime(func(&exec_handle, h, null, null, 0));
        return GraphExec{ .handle = exec_handle };
    }

    pub fn deinit(self: *Graph) void {
        if (self.handle) |h| {
            if (loader.globalLoader().getRuntimeSymbol(*const fn (*anyopaque) callconv(.c) c_int, "cudaGraphDestroy")) |func| {
                _ = func(h);
            }
            self.handle = null;
        }
    }
};

/// Begins stream capture on `stream`.
pub fn beginCapture(stream: stream_mod.Stream, mode: StreamCaptureMode) err.CudaError!void {
    if (!loader.isAvailable()) {
        return;
    }
    const func = loader.globalLoader().getRuntimeSymbol(*const fn (*anyopaque, c_uint) callconv(.c) c_int, "cudaStreamBeginCapture") orelse return error.DriverError;
    try result.checkRuntime(func(stream.raw(), @intFromEnum(mode)));
}

/// Ends stream capture on `stream` and returns the constructed `Graph`.
pub fn endCapture(stream: stream_mod.Stream) err.CudaError!Graph {
    if (!loader.isAvailable()) {
        return Graph{ .handle = null };
    }
    var graph_handle: ?*anyopaque = null;
    const func = loader.globalLoader().getRuntimeSymbol(*const fn (*stream_mod.Stream.RawHandle, *?*anyopaque) callconv(.c) c_int, "cudaStreamEndCapture") orelse return error.DriverError;
    try result.checkRuntime(func(@ptrCast(@constCast(&stream.raw())), &graph_handle));
    return Graph{ .handle = graph_handle };
}

test "Graph capture handles CPU fallback safely" {
    if (loader.isAvailable()) return;
    var s = try stream_mod.Stream.init();
    defer s.deinit();
    try beginCapture(s, .Global);
    var g = try endCapture(s);
    defer g.deinit();
    var exec = try g.instantiate();
    defer exec.deinit();
    try exec.launch(s);
}

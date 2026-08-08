//! Root module for cuda.zig re-exporting the whole public API.

const std = @import("std");

pub const version = @import("core/version.zig");
pub const loader = @import("core/loader.zig");
pub const error_mod = @import("core/error.zig");

pub const CudaError = error_mod.CudaError;

// Device & management
pub const selection = @import("device/selection.zig");
pub const isAvailable = loader.isAvailable;
pub const deviceCount = selection.deviceCount;
pub const setDevice = selection.setDevice;
pub const currentDevice = selection.currentDevice;
pub const synchronize = selection.synchronize;
pub const runtimeCapabilities = version.capabilities;
pub const CudaVersion = version.CudaVersion;
pub const RuntimeCapabilities = version.RuntimeCapabilities;

pub const device_mod = @import("device/device.zig");
pub const Device = device_mod.Device;
pub const DeviceProperties = @import("device/properties.zig").DeviceProperties;
pub const allDevices = @import("device/enumeration.zig").allDevices;

// Memory
pub const device_memory = @import("memory/device_memory.zig");
pub const DeviceBuffer = device_memory.DeviceBuffer;

pub const host_memory = @import("memory/host_memory.zig");
pub const PinnedBuffer = host_memory.PinnedBuffer;

pub const unified_memory = @import("memory/unified_memory.zig");
pub const UnifiedBuffer = unified_memory.UnifiedBuffer;

pub const transfer = @import("memory/transfer.zig");
pub const allocator = @import("memory/allocator.zig");
pub const CudaAllocator = allocator.cudaAllocator;

// Streams & events
pub const stream_mod = @import("stream/stream.zig");
pub const Stream = stream_mod.Stream;

pub const event_mod = @import("stream/event.zig");
pub const Event = event_mod.Event;

// Kernel launch & modules
pub const kernel = struct {
    pub const launch_mod = @import("kernel/launch.zig");
    pub const LaunchConfig = launch_mod.LaunchConfig;
    pub const launch = launch_mod.launch;

    pub const module_mod = @import("kernel/module.zig");
    pub const KernelModule = module_mod.KernelModule;

    pub const func_mod = @import("kernel/function.zig");
    pub const Function = func_mod.Function;

    pub const graph = @import("kernel/graph.zig");
};

pub const LaunchConfig = kernel.LaunchConfig;
pub const launch = kernel.launch;

// NVRTC
pub const nvrtc = @import("nvrtc/compiler.zig");
pub const NvrtcCompiler = nvrtc.NvrtcCompiler;

// High-level Tensor
pub const tensor = struct {
    pub const Tensor = @import("tensor/tensor.zig").Tensor;
    pub const Shape = @import("tensor/shape.zig").Shape;
    pub const DType = @import("tensor/dtype.zig").DType;
    pub const transform = @import("tensor/ops/transform.zig");
};
pub const Tensor = tensor.Tensor;
pub const Shape = tensor.Shape;
pub const DType = tensor.DType;

// Memory Pool (v0.0.2)
pub const pool = @import("memory/pool.zig");
pub const PoolBuffer = pool.PoolBuffer;

// Occupancy & Profiler (v0.0.2)
pub const occupancy = @import("kernel/occupancy.zig");
pub const profiler = @import("utils/profiler.zig");

// CPU Fallback
pub const fallback = struct {
    pub const cpu_backend = @import("fallback/cpu_backend.zig");
    pub const dispatch = @import("fallback/dispatch.zig");
};

// Driver & Runtime FFI layers
pub const driver = struct {
    pub const ffi = @import("driver/ffi.zig");
    pub const context = @import("driver/context.zig");
    pub const module = @import("driver/module.zig");
    pub const function = @import("driver/function.zig");
};

pub const runtime = struct {
    pub const ffi = @import("runtime/ffi.zig");
    pub const device = @import("runtime/device.zig");
    pub const stream = @import("runtime/stream.zig");
    pub const event = @import("runtime/event.zig");
    pub const memory = @import("runtime/memory.zig");
};

test {
    std.testing.refAllDecls(@This());
}

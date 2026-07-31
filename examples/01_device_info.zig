const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Device Info Example ===\n\n", .{});

    if (!cuda.isAvailable()) {
        std.debug.print("CUDA is NOT available on this system. Operating in CPU Fallback mode.\n", .{});
        return;
    }

    // Display version info
    const caps = cuda.runtimeCapabilities();
    if (caps.driver_version.raw > 0) {
        std.debug.print("CUDA Driver Version:  {d}.{d}\n", .{ caps.driver_version.major(), caps.driver_version.minor() });
    }
    if (caps.runtime_version.raw > 0) {
        std.debug.print("CUDA Runtime Version: {d}.{d}\n", .{ caps.runtime_version.major(), caps.runtime_version.minor() });
    }

    const count = try cuda.deviceCount();
    std.debug.print("Found {d} CUDA device(s).\n\n", .{count});

    const allocator = std.heap.page_allocator;
    const devices = try cuda.allDevices(allocator);
    defer allocator.free(devices);

    for (devices, 0..) |dev, i| {
        std.debug.print("Device #{d}:\n", .{i});

        const dev_name = try dev.name();
        std.debug.print("  Name:               {s}\n", .{dev_name});

        const cap = try dev.computeCapability();
        std.debug.print("  Compute Capability: {d}.{d}\n", .{ cap.major, cap.minor });

        const total_mem = try dev.totalMemory();
        std.debug.print("  Total Memory:       {d} MB ({d} bytes)\n", .{ total_mem / (1024 * 1024), total_mem });

        // Try to get free memory
        if (dev.freeMemory()) |free_mem| {
            std.debug.print("  Free Memory:        {d} MB\n", .{free_mem / (1024 * 1024)});
        } else |_| {}

        const props = try dev.propertiesRaw();
        std.debug.print("  Multiprocessors:    {d}\n", .{props.multi_processor_count});
        std.debug.print("  Max Threads/Block:  {d}\n", .{props.max_threads_per_block});
        std.debug.print("  Warp Size:          {d}\n", .{props.warp_size});
        std.debug.print("  Clock Rate:         {d} MHz\n", .{props.clock_rate_khz / 1000});
        std.debug.print("  L2 Cache:           {d} KB\n", .{props.l2_cache_size / 1024});
        std.debug.print("  Unified Addressing: {}\n", .{props.unified_addressing});
        std.debug.print("  Managed Memory:     {}\n", .{props.managed_memory});
        std.debug.print("\n", .{});
    }

    // Feature caps
    std.debug.print("=== CUDA Feature Support ===\n", .{});
    std.debug.print("  CUDA 12+:             {}\n", .{caps.supports_cuda_12});
    std.debug.print("  CUDA Graphs:          {}\n", .{caps.supports_graphs});
    std.debug.print("  Cooperative Launch:   {}\n", .{caps.supports_cooperative_launch});
    std.debug.print("  Memory Pools:         {}\n", .{caps.supports_memory_pools});
    std.debug.print("  Tensor Cores:         {}\n", .{caps.supports_tensor_cores});
    std.debug.print("  Managed Prefetch:     {}\n", .{caps.supports_managed_prefetch});
}

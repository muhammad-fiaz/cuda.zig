const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const cuda_mod = b.addModule("cuda", .{
        .root_source_file = b.path("src/cuda.zig"),
        .target = target,
        .optimize = optimize,
    });

    const main_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cuda.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_main_tests = b.addRunArtifact(main_tests);
    const test_step = b.step("test", "Run library unit tests");
    test_step.dependOn(&run_main_tests.step);

    const examples = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "example-device-info", .path = "examples/01_device_info.zig" },
        .{ .name = "example-memory-transfer", .path = "examples/02_memory_transfer.zig" },
        .{ .name = "example-kernel-launch", .path = "examples/03_kernel_launch.zig" },
        .{ .name = "example-streams-events", .path = "examples/04_streams_events.zig" },
        .{ .name = "example-tensor-ops", .path = "examples/05_tensor_ops.zig" },
        .{ .name = "example-multi-gpu", .path = "examples/06_multi_gpu.zig" },
        .{ .name = "example-cpu-fallback", .path = "examples/07_cpu_fallback.zig" },
        .{ .name = "example-managed-memory", .path = "examples/08_managed_memory.zig" },
        .{ .name = "example-nvrtc-compilation", .path = "examples/09_nvrtc_compilation.zig" },
        .{ .name = "example-memory-pools-pitched", .path = "examples/10_memory_pools_pitched.zig" },
        .{ .name = "example-occupancy-profiler", .path = "examples/11_occupancy_profiler.zig" },
        .{ .name = "example-benchmark-matrix-ops", .path = "examples/12_benchmark_matrix_ops.zig" },
    };

    const run_all_step = b.step("run-all-examples", "Run all example executables");

    for (examples) |ex| {
        const exe = b.addExecutable(.{
            .name = ex.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(ex.path),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "cuda", .module = cuda_mod },
                },
            }),
        });

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());

        const example_step = b.step(ex.name, b.fmt("Run {s}", .{ex.name}));
        example_step.dependOn(&run_cmd.step);
        run_all_step.dependOn(&run_cmd.step);
    }
}

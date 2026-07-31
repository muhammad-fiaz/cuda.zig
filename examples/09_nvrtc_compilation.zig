const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    std.debug.print("=== cuda.zig Advanced NVRTC Dynamic Kernel Compilation Example ===\n", .{});

    if (!cuda.nvrtc.NvrtcCompiler.isAvailable()) {
        std.debug.print("NVRTC runtime compilation library is not available on this environment.\n", .{});
        return;
    }

    const cuda_src =
        \\extern "C" __global__ void saxpy(float a, float *x, float *y, float *out, int n) {
        \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (idx < n) {
        \\        out[idx] = a * x[idx] + y[idx];
        \\    }
        \\}
    ;

    const allocator = std.heap.page_allocator;
    const ptx = cuda.nvrtc.NvrtcCompiler.compileToPTX(
        allocator,
        cuda_src,
        "saxpy.cu",
        &.{},
    ) catch |err| {
        std.debug.print("Compilation failed or NVRTC not present: {s}\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(ptx);

    std.debug.print("Successfully compiled CUDA C++ source code to PTX ({d} bytes).\n", .{ptx.len});
}

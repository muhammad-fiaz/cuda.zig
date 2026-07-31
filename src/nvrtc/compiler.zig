//! Runtime compilation of CUDA C++ source code to PTX using NVRTC.

const std = @import("std");
const loader = @import("../core/loader.zig");
const err = @import("../core/error.zig");

pub const NvrtcProgram = *anyopaque;

const NvrtcCreateProgramFn = *const fn (
    prog: **anyopaque,
    src: [*:0]const u8,
    name: ?[*:0]const u8,
    numHeaders: c_int,
    headers: ?[*]const [*:0]const u8,
    includeNames: ?[*]const [*:0]const u8,
) callconv(.c) c_int;

const NvrtcCompileProgramFn = *const fn (
    prog: *anyopaque,
    numOptions: c_int,
    options: ?[*]const [*:0]const u8,
) callconv(.c) c_int;

const NvrtcGetPTXSizeFn = *const fn (prog: *anyopaque, ptxSizeRet: *usize) callconv(.c) c_int;
const NvrtcGetPTXFn = *const fn (prog: *anyopaque, ptx: [*]u8) callconv(.c) c_int;
const NvrtcDestroyProgramFn = *const fn (prog: **anyopaque) callconv(.c) c_int;

pub const NvrtcCompiler = struct {
    pub fn isAvailable() bool {
        return loader.globalLoader().isNvrtcAvailable();
    }

    /// Compiles CUDA C++ source text `source` into an allocated PTX string slice.
    ///
    /// The caller owns the returned slice and must free it with `allocator.free()`.
    pub fn compileToPTX(
        allocator: std.mem.Allocator,
        source: [:0]const u8,
        name: ?[:0]const u8,
        options: []const [:0]const u8,
    ) err.CudaError![:0]u8 {
        const ldr = loader.globalLoader();
        if (!ldr.isNvrtcAvailable()) return error.NvrtcNotFound;

        const create_fn = ldr.getNvrtcSymbol(NvrtcCreateProgramFn, "nvrtcCreateProgram") orelse return error.NvrtcNotFound;
        const compile_fn = ldr.getNvrtcSymbol(NvrtcCompileProgramFn, "nvrtcCompileProgram") orelse return error.NvrtcNotFound;
        const get_size_fn = ldr.getNvrtcSymbol(NvrtcGetPTXSizeFn, "nvrtcGetPTXSize") orelse return error.NvrtcNotFound;
        const get_ptx_fn = ldr.getNvrtcSymbol(NvrtcGetPTXFn, "nvrtcGetPTX") orelse return error.NvrtcNotFound;
        const destroy_fn = ldr.getNvrtcSymbol(NvrtcDestroyProgramFn, "nvrtcDestroyProgram") orelse return error.NvrtcNotFound;

        var prog: *anyopaque = undefined;
        const name_ptr = if (name) |n| n.ptr else null;
        if (create_fn(&prog, source.ptr, name_ptr, 0, null, null) != 0) {
            return error.InvalidSource;
        }
        defer _ = destroy_fn(&prog);

        var c_opts_buf: [16][*:0]const u8 = undefined;
        const opts_len = @min(options.len, 16);
        for (0..opts_len) |i| {
            c_opts_buf[i] = options[i].ptr;
        }

        const comp_res = compile_fn(prog, @intCast(opts_len), if (opts_len > 0) &c_opts_buf else null);
        if (comp_res != 0) return error.InvalidSource;

        var ptx_size: usize = 0;
        if (get_size_fn(prog, &ptx_size) != 0) return error.Unknown;

        const ptx_buf = allocator.allocSentinel(u8, ptx_size - 1, 0) catch return error.OutOfMemory;
        errdefer allocator.free(ptx_buf);

        if (get_ptx_fn(prog, ptx_buf.ptr) != 0) return error.Unknown;

        return ptx_buf;
    }
};

test "NvrtcCompiler availability check" {
    _ = NvrtcCompiler.isAvailable();
}

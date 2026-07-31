//! Numeric DType definitions for Tensor operations.

const std = @import("std");

pub const DType = enum {
    f16,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u32,

    pub fn sizeOf(self: DType) usize {
        return switch (self) {
            .f16, .i16 => 2,
            .f32, .i32, .u32 => 4,
            .f64, .i64 => 8,
            .i8, .u8 => 1,
        };
    }
};

pub fn typeToDType(comptime T: type) DType {
    return switch (T) {
        f16 => .f16,
        f32 => .f32,
        f64 => .f64,
        i8 => .i8,
        i16 => .i16,
        i32 => .i32,
        i64 => .i64,
        u8 => .u8,
        u32 => .u32,
        else => @compileError("Unsupported DType for Tensor"),
    };
}

test "DType sizeOf" {
    try std.testing.expectEqual(@as(usize, 4), DType.f32.sizeOf());
    try std.testing.expectEqual(@as(usize, 8), DType.f64.sizeOf());
    try std.testing.expectEqual(DType.f32, typeToDType(f32));
}

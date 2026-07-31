//! Logging helper for cuda.zig.

const std = @import("std");

pub const log = std.log.scoped(.cuda);

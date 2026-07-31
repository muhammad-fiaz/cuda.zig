---
title: Stream & Event API
description: Async stream and event APIs, synchronisation, and timing in cuda.zig.
---

# Stream & Event API

## `cuda.Stream`

```zig
pub const Stream = struct {
    handle: cudaStream_t,

    /// Create a new non-blocking CUDA stream.
    pub fn init() !Stream

    /// Destroy the stream (blocks until all pending work completes).
    pub fn deinit(self: *Stream) void

    /// Block the calling thread until all operations in this stream finish.
    pub fn sync(self: Stream) !void

    /// Non-blocking status check.
    /// Returns true if all operations have completed.
    pub fn query(self: Stream) !bool

    /// Make this stream wait for `event` before executing subsequent work.
    pub fn waitEvent(self: Stream, event: Event) !void

    /// Return the raw handle for passing to CUDA C APIs.
    pub fn raw(self: Stream) cudaStream_t
};
```

## `cuda.Event`

```zig
pub const Event = struct {
    handle: cudaEvent_t,

    /// Create a CUDA event with default flags.
    pub fn init() !Event

    /// Create a CUDA event with explicit flags.
    pub fn initFlags(flags: EventFlags) !Event

    /// Destroy the event.
    pub fn deinit(self: *Event) void

    /// Record the event into a stream at the current position.
    pub fn record(self: Event, stream: Stream) !void

    /// Block the calling thread until this event has been recorded and all
    /// preceding stream operations are complete.
    pub fn sync(self: Event) !void

    /// Non-blocking: true if the event has been recorded and completed.
    pub fn query(self: Event) !bool

    /// Compute elapsed time in milliseconds between two recorded events.
    /// Both events must have completed.
    pub fn elapsedMs(start: Event, stop: Event) !f32

    /// Return the raw handle.
    pub fn raw(self: Event) cudaEvent_t
};
```

### `EventFlags`

```zig
pub const EventFlags = packed struct {
    /// If set, do not implicitly synchronise with the default stream.
    disable_timing: bool = false,
    /// For IPC event sharing (advanced).
    interprocess: bool = false,
    _pad: u30 = 0,
};
```

## Usage Pattern

```zig
var stream = try cuda.Stream.init();
defer stream.deinit();

var t0 = try cuda.Event.init();
var t1 = try cuda.Event.init();
defer t0.deinit();
defer t1.deinit();

try t0.record(stream);
// ... enqueue GPU work on stream ...
try t1.record(stream);
try t1.sync();

const ms = try cuda.Event.elapsedMs(t0, t1);
std.debug.print("Elapsed: {d:.3} ms\n", .{ms});
```

---
title: Streams & Events
description: Async execution, CUDA streams, events, elapsed-time measurement, and synchronisation in cuda.zig.
---

# Streams & Events

## Streams

A CUDA stream is an ordered queue of GPU operations. Operations within a single stream execute in issue order; operations across streams may overlap.

### Create a Stream

```zig
var stream = try cuda.Stream.init();
defer stream.deinit();
```

### Synchronise

```zig
// Block until all operations in the stream complete
try stream.sync();

// Non-blocking: returns .NotReady or .Success
const ready = try stream.query();
```

### Null Stream

The default (null) stream serialises with all other streams on the device. Use named streams to achieve concurrency:

```zig
// Two independent copy-compute-copy pipelines
var stream_a = try cuda.Stream.init();
var stream_b = try cuda.Stream.init();
defer stream_a.deinit();
defer stream_b.deinit();
```

## Events

Events mark a point in a stream's execution and can be used for synchronisation and timing.

### Create and Record

```zig
var start = try cuda.Event.init();
var stop  = try cuda.Event.init();
defer start.deinit();
defer stop.deinit();

try start.record(stream);
// ... GPU work ...
try stop.record(stream);

// Block until 'stop' completes
try stop.sync();
```

### Elapsed Time

```zig
const ms = try cuda.Event.elapsedMs(start, stop);
std.debug.print("Kernel time: {d:.3} ms\n", .{ms});
```

`elapsedMs` calls `cudaEventElapsedTime`. Both events must have been recorded on streams that have completed.

### Cross-Stream Synchronisation

Make one stream wait for an event recorded in another:

```zig
try stream_b.waitEvent(event_from_a);
```

## Full Example

```zig
var stream = try cuda.Stream.init();
defer stream.deinit();

var t0 = try cuda.Event.init();
var t1 = try cuda.Event.init();
defer t0.deinit();
defer t1.deinit();

try t0.record(stream);
try buf.copyFromHostAsync(&host_data, stream);
// launch kernel async on stream ...
try buf.copyToHostAsync(&result, stream);
try t1.record(stream);

try t1.sync();
const elapsed = try cuda.Event.elapsedMs(t0, t1);
std.debug.print("Pipeline: {d:.2} ms\n", .{elapsed});
```

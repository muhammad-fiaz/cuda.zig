---
title: Stream & Event API Documentation
description: Reference for Stream, Event, stream priorities, and asynchronous execution in cuda.zig.
---

# Stream & Event API Reference

CUDA streams manage asynchronous work ordering and execution overlap.

## `Stream`

Handle representing a CUDA stream.

```zig
const stream = try cuda.Stream.create();
defer stream.destroy();

try stream.synchronize();
```

### Stream Methods & Functions

| Function | Description |
|----------|-------------|
| `Stream.create()` | Create a new stream with default flags |
| `Stream.createWithFlags(flags)` | Create stream with custom flags (e.g. non-blocking) |
| `Stream.createWithPriority(flags, priority)` | Create stream with specific priority |
| `stream.destroy()` | Release stream resources |
| `stream.synchronize()` | Block host until stream completes all work |
| `stream.query()` | Non-blocking check for stream completion |
| `stream.waitEvent(event)` | Insert a stream-side wait on an event |
| `stream.getPriority()` | Query priority integer of this stream |
| `cuda.stream.getPriorityRange()` | Query device's `[least, greatest]` priority range |

## `Event`

CUDA events provide fine-grained timing and cross-stream synchronization.

```zig
const start_evt = try cuda.Event.create();
const end_evt = try cuda.Event.create();
defer start_evt.destroy();
defer end_evt.destroy();

try start_evt.record(stream);
// launch work...
try end_evt.record(stream);
try end_evt.synchronize();

const elapsed_ms = try cuda.Event.elapsedTime(start_evt, end_evt);
```

### Event Methods & Functions

| Function | Description |
|----------|-------------|
| `Event.create()` | Create timing event |
| `Event.createWithFlags(flags)` | Create event with custom flags |
| `event.destroy()` | Destroy event |
| `event.record(stream)` | Record event in stream |
| `event.synchronize()` | Block host until event is reached |
| `event.query()` | Non-blocking query for event completion |
| `Event.elapsedTime(start, end)` | Calculate elapsed milliseconds between events |

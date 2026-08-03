---
title: Related Zig Projects
description: Explore the growing ecosystem of Zig libraries that complement cuda.zig for building modern, high-performance applications.
---

# Related Zig Projects

cuda.zig is part of a growing ecosystem of Zig libraries by [Muhammad Fiaz](https://github.com/muhammad-fiaz). These projects are designed to work together, providing a complete toolkit for building modern, high-performance Zig applications.

## Environment & Configuration

| Project | Description | Repository |
|---------|-------------|------------|
| **env.zig** | `.env` file parsing for Zig projects | [GitHub](https://github.com/muhammad-fiaz/env.zig) |
| **args.zig** | Command-line argument parsing | [GitHub](https://github.com/muhammad-fiaz/args.zig) |
| **zon.zig** | ZON file format support | [GitHub](https://github.com/muhammad-fiaz/zon.zig) |

## User Interface

| Project | Description | Repository |
|---------|-------------|------------|
| **tui.zig** | Terminal User Interface (TUI) support | [GitHub](https://github.com/muhammad-fiaz/tui.zig) |
| **loaders.zig** | Spinners, loading indicators, and progress bars | [GitHub](https://github.com/muhammad-fiaz/loaders.zig) |

## Networking & APIs

| Project | Description | Repository |
|---------|-------------|------------|
| **httpx.zig** | HTTP client and server support | [GitHub](https://github.com/muhammad-fiaz/httpx.zig) |
| **api.zig** | API framework for building REST/GraphQL services | [GitHub](https://github.com/muhammad-fiaz/api.zig) |
| **zix** | Web framework for Zig | [GitHub](https://github.com/muhammad-fiaz/zix) |
| **mcp.zig** | Model Context Protocol (MCP) support | [GitHub](https://github.com/muhammad-fiaz/mcp.zig) |

## Data & Serialization

| Project | Description | Repository |
|---------|-------------|------------|
| **zigantic** | Data validation and serialization | [GitHub](https://github.com/muhammad-fiaz/zigantic) |
| **num.zig** | Numerical computing support | [GitHub](https://github.com/muhammad-fiaz/num.zig) |

## File Operations & Compression

| Project | Description | Repository |
|---------|-------------|------------|
| **archive.zig** | Archive and compression support | [GitHub](https://github.com/muhammad-fiaz/archive.zig) |
| **zigx** | Compression file format support | [GitHub](https://github.com/muhammad-fiaz/zigx) |
| **downloader.zig** | File downloading support | [GitHub](https://github.com/muhammad-fiaz/downloader.zig) |

## Development Tools

| Project | Description | Repository |
|---------|-------------|------------|
| **buildx.zig** | Build tooling and utilities | [GitHub](https://github.com/muhammad-fiaz/buildx.zig) |
| **updater.zig** | Update checker and auto-updater | [GitHub](https://github.com/muhammad-fiaz/updater.zig) |
| **logly.zig** | Logging support | [GitHub](https://github.com/muhammad-fiaz/logly.zig) |

## GPU Computing

| Project | Description | Repository |
|---------|-------------|------------|
| **cuda.zig** | CUDA Runtime and Driver API for Zig | [GitHub](https://github.com/muhammad-fiaz/cuda.zig) |

## Getting Started

Most of these projects can be added to your Zig project using `zig fetch`:

```sh
zig fetch --save https://github.com/muhammad-fiaz/<project>/archive/refs/heads/main.tar.gz
```

Then add the dependency in your `build.zig.zon` and `build.zig` as shown in the [Getting Started guide](/guide/getting-started).

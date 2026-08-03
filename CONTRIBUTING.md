# Contributing to cuda.zig

Thank you for your interest in contributing to **cuda.zig**! This document provides guidelines and instructions to help you get started.

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive experience for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on [GitHub Issues](https://github.com/muhammad-fiaz/cuda.zig/issues) with:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Zig version, CUDA version, GPU model)
- Relevant code snippets or error messages

### Suggesting Features

Feature requests are welcome! Please open an issue with:

- A clear description of the feature
- The use case and motivation
- Any relevant examples or pseudocode

### Submitting Pull Requests

1. **Fork** the repository
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/cuda.zig.git
   cd cuda.zig
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following the coding standards below
5. **Run tests** to ensure nothing is broken:
   ```bash
   zig build test
   ```
6. **Commit** with a clear message:
   ```bash
   git commit -m "feat: add new feature description"
   ```
7. **Push** and open a Pull Request

## Development Setup

### Prerequisites

| Requirement | Version |
|-------------|---------|
| [Zig](https://ziglang.org/download/) | 0.16.0+ |
| CUDA Toolkit *(optional)* | 12.x or 13.x |
| GPU *(optional)* | NVIDIA with compute capability 5.0+ |

### Building

```bash
zig build
```

### Running Tests

```bash
zig build test
```

### Running Examples

```bash
zig build example-device-info
zig build example-memory-transfer
```

## Coding Standards

- Follow the [Zig Style Guide](https://ziglang.org/documentation/master/#Style-Guide)
- Use descriptive variable and function names
- Keep functions focused and concise
- Add comments for complex logic
- Handle errors explicitly — never use `catch unreachable` unless truly justified
- Maintain backward compatibility when possible

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation changes
- `test:` — Adding or updating tests
- `refactor:` — Code refactoring
- `perf:` — Performance improvements
- `ci:` — CI/CD changes
- `chore:` — Maintenance tasks

Examples:
```
feat: add cuBLAS matrix multiplication support
fix: handle CUDA driver not found gracefully
docs: update installation instructions
test: add device enumeration tests
```

## License

By contributing to cuda.zig, you agree that your contributions will be licensed under the [MIT License](LICENSE).

## Questions?

If you have questions about contributing, feel free to:

- Open a [Discussion](https://github.com/muhammad-fiaz/cuda.zig/discussions)
- Contact: [contact@muhammadfiaz.com](mailto:contact@muhammadfiaz.com)

---

Thank you for contributing to cuda.zig!

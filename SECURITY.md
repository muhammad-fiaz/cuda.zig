# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in **cuda.zig**, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

Please send an email to **[contact@muhammadfiaz.com](mailto:contact@muhammadfiaz.com)** with:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (if applicable)

### What to Expect

- **Acknowledgment**: You will receive an acknowledgment within 48 hours
- **Assessment**: We will investigate and assess the severity
- **Fix**: A fix will be developed and released as soon as possible
- **Disclosure**: We will coordinate with you on public disclosure timing

## Security Best Practices

When using cuda.zig in your projects:

- Keep your CUDA drivers and toolkit up to date
- Validate all input data before passing to GPU kernels
- Use proper memory bounds checking
- Be cautious with dynamic library loading paths
- Review kernel code for potential race conditions

## Scope

This security policy applies to:

- The cuda.zig library code
- Build system configuration
- Documentation that may contain security-relevant information

## Out of Scope

- Third-party dependencies (report these to their respective maintainers)
- CUDA driver or toolkit vulnerabilities (report to NVIDIA)
- Issues in user code that uses cuda.zig

---

Thank you for helping keep cuda.zig and its users safe.

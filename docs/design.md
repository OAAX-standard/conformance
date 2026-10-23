# Design

## Architecture

- **C harness** (C11, CMake; Linux only for now: it uses `dlfcn` and `/proc`): the only code that touches a runtime. It runs fixed *scenarios* and prints one
  JSON line per call (status, error text, timing, thread count, RSS), flushed after every call so a crash is attributable.
- **Python orchestrator**: runs each scenario in its own process with a timeout (a crashing runtime is a failure, not a dead
  suite), holds **all assertions**, spec clauses and levels, runs the toolchain tier (Docker), compares with ONNX, writes the
  JSON and JUnit reports. Adding a check is a Python edit; only a new scenario needs C.
- **`mock-runtime/`**: a small runtime with switchable faults (drop a request, `exit()` in cleanup, leak a thread, wrong
  status, accept 0 tensors). The suite's own CI proves every check fails its fault and passes the clean mock.
- **`spec/<version>/`**: the pinned `interface.h` per spec version; tests cite it, vendor copies cannot drift.

```
harness/  python/oaax_conformance/  spec/2.0/  mock-runtime/  mock-toolchain/  tests/  docs/
```

## Distribution

**No Docker image for the suite** (decided 2026-10-23). The runtime is loaded in the suite's own process, so it must see its vendor
libraries and device nodes, and Windows DLLs cannot be tested from a Linux container; the toolchain tier starts Docker containers
itself and must not need the host's Docker socket handed to another container.

- `pip install` of the Python package plus the C harness built with CMake (`scripts/build.sh`). Linux (x86_64, aarch64) first; Windows later.
- The toolchain tier needs Docker on the machine running the suite, because a toolchain *is* a Docker image.

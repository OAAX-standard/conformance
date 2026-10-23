# Design

## Architecture

- **C harness** (C11, CMake, builds with MSVC): the only code that touches a runtime. It runs fixed *scenarios* and prints one
  JSON line per call (status, error text, timing, thread count, RSS), flushed after every call so a crash is attributable.
- **Python orchestrator**: runs each scenario in its own process with a timeout (a crashing runtime is a failure, not a dead
  suite), holds **all assertions**, spec clauses and levels, runs the toolchain tier (Docker), compares with ONNX, writes the
  JSON and JUnit reports. Adding a check is a Python edit; only a new scenario needs C.
- **`mock-runtime/`**: a small runtime with switchable faults (drop a request, `exit()` in cleanup, leak a thread, wrong
  status, accept 0 tensors). The suite's own CI proves every check fails its fault and passes the clean mock.
- **`spec/<version>/`**: the pinned `interface.h` per spec version; tests cite it, vendor copies cannot drift.

```
harness/  python/oaax_conformance/  spec/2.0/  models/  fixtures/  mock-runtime/  tests/  docs/
```

## Distribution

The suite ships as a **Docker image**. Known limits, so nobody is surprised:

- The runtime is loaded *inside* the container: its vendor libraries (OpenVINO, CUDA, HailoRT, ...) and its device nodes must be
  available there (`FROM` the suite image, `--device`, `--gpus`). Otherwise `dlopen` fails for reasons that are not the runtime's.
- Windows DLLs cannot be tested in a Linux image.
- The toolchain tier starts Docker containers: it needs the host's Docker socket mounted (root-equivalent on that host) or a
  native run.

A native run (`pip install` of the same Python package, harness built with CMake) stays available for those cases.

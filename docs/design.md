# Design

## Architecture

- **C harness** (C11, CMake; Linux only: it uses `dlfcn` and `/proc`; shipped inside the wheel): the only code that touches a runtime. It runs fixed *scenarios* and prints one
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
libraries and device nodes, and a Windows DLL cannot be tested from a Linux container (the runtime tier runs natively on Windows instead); the toolchain tier starts Docker containers
itself and must not need the host's Docker socket handed to another container.

- The wheel (`uv tool install 'oaax-conformance[e2e]'`) or a checkout (`uv sync` after `scripts/build.sh`). Both run every tier: the wheel carries the compiled harness.
- **Wheel decision (2026-10-27):** one platform wheel per architecture (`py3-none-manylinux_2_17_*`, x86_64 and aarch64), the harness compiled by `hatch_build.py` (cmake) inside a manylinux_2_28 container.
  The binary needs glibc 2.17 symbols only (`auditwheel show` checks it in CI), so it runs on Ubuntu 20.04 and newer and Debian 11 and newer.
  cibuildwheel was not needed: it solves many Pythons and platforms, and this package has one binary and no Python ABI (`py3-none`).
  An sdist builds (a pure wheel without cmake, with the harness when cmake exists) but is not released. macOS has no harness: the CLI says so (exit 3).
- **Windows (2026-11-14):** the runtime tier only; the toolchain tier tests Linux containers and says so (exit 3). All operating-system calls live in `harness/src/platform.c` (LoadLibrary, Win32 threads, Toolhelp, private bytes instead of RSS);
  the mock runtime is built with MSVC (`/EXPORT`), checked in CI on `windows-latest`. A crash is named by its exception code (`ACCESS_VIOLATION`); RT-S-02 reads the PE export table. The harness frees what the runtime allocated with
  its own `free` (dynamic UCRT, `/MD`): a runtime built with another CRT can crash there, which is a finding about the runtime. A Windows wheel (`win_amd64`) builds with the same hook but is not in the release workflow yet.
  Limits: Windows has no SIGTERM/SIGHUP handling (Ctrl-C works and stops the harness and its helpers with `taskkill /T`; a suite killed from outside leaves the harness to finish), and the POSIX-only faults of the mock (fork, a non-UTF-8 thread name, a linker version script) are not exercised.
- The toolchain tier needs Docker on the machine running the suite, because a toolchain *is* a Docker image.

## Direction for OAAX 2.1

Decided with the author while building the suite. 2.0 is frozen; these become requirements in 2.1, and the suite will move their tests from PROPOSED to MUST for 2.1:

- a later `load_models` call is an error and leaves the loaded models alone (status `ALREADY_INITIALIZED`);
- a process that exits after `init` without `cleanup` must not crash;
- the toolchain writes `oaax-manifest.json` (draft schema PR in the OAAX repo, marked 2.1);
- the runtime reports model I/O in `runtime_get_info`;
- a state and status-code table, per-request failure reporting and the other open points of the 2.0 review.

The reference runtime and toolchain findings behind this are filed as issues #45-#49 of the reference-implementation repository.

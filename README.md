# OAAX conformance suite

Checks whether an OAAX **runtime** (shared library) and/or **toolchain** (Docker image) conforms to the
[OAAX specification](https://oaax-standard.github.io/OAAX/Specification/).

> **Status: early, Linux only.** The runtime tiers (static, lifecycle, model) and the toolchain / end-to-end tiers run; every test is specified in
> [docs/tests/](docs/tests/) (level, quoted spec clause, what the C++ CPU runtime and toolchain do today).

## Principles

- **Easy for implementers**: one command, no setup beyond Docker/Python, readable failures: each failure names the spec
  clause and says what to fix.
- **Dead simple for users**: a pass means "this runtime/toolchain can be swapped for another one without code changes".
- **Strict on the raw API**: the suite talks to the runtime's C API directly, never through a tolerant wrapper.
- **Safe on real hardware**: loops stop at a time budget, request sizes are capped, a slow device is `BLOCKED` and never `FAIL`, and `--hardware-safe` leaves
  out the deliberately abusive tests.
- **Runs anywhere**: the static tier needs no accelerator, no model and no `init`.

## Tiers

| Tier | Needs | Checks |
| ---- | ----- | ------ |
| `runtime/static` | the `.so` / `.dll` only | exported symbols, getters, calls before `init` |
| `runtime/lifecycle` | a working `init` (device, or the CPU runtime) | `init`/`cleanup` rules, error text, cycles, threads, unload, memory, exit |
| `runtime/model` | a model the runtime can load (+ the device) | enqueue/retrieve, id echo, multi-model routing, timeouts, concurrency, input validation, leaks |
| `toolchain` | the Docker image | arguments, exit codes, log and manifest, no root / no network, output integrity, hostile input |
| `e2e` | both | convert, load, run, compare with the ONNX reference within the declared precision |

## Test levels

Every test carries the spec clause it comes from and a level:

- **MUST**: required by the specification text of the version under test. A failure fails the run.
- **SHOULD**: recommended by the text. A failure is a warning.
- **PROPOSED**: behaviour planned for the next version. Always a warning, so the suite can lead the specification.

## Usage (Linux)

```bash
scripts/build.sh                       # builds the C harness and the mock runtimes into ./build (needs cmake, a C compiler)
pip install -e ".[onnx]"               # from a checkout; the wheel does not ship the harness yet
oaax-conformance runtime ./libRuntimeLibrary.so [--init-config key=value]... [--model FILE] [--input NAME:DTYPE:DIMS]... [--hardware-safe] [--out DIR]
```

Useful options (all in `--help`): `--model FILE` is a model the runtime can load and enables the tests that send requests. To build them the suite needs
to know what the model takes: it reads an `oaax-manifest.json` next to the model or the `.onnx` file itself, and `--input NAME:DTYPE:DIMS` (use `?` for
a dynamic dimension) overrides both, which is also how to describe a runtime that takes another data type or layout than the ONNX model. `--timeout`
(seconds per scenario, default 300) and `--round-trip-timeout` are for slow devices: loops stop at 60% of the timeout, and a call still running at the
limit is `BLOCKED`, not failed. `--skip ID` leaves a test out; `--hardware-safe` shrinks cycles and floods, pauses between scenarios and skips the
deliberately abusive tests (NULL pointers, exit without cleanup, cleanup during a blocked call, request flood). `--workdir DIR` runs the harness in a given
directory; `--quiet` hides the progress lines on stderr.

Three outputs: a **text summary** on the console (a table, one row per test), and with `--out DIR`:

- `report.html`: a self-contained page (works offline) with the verdict, the expected behaviour and spec clause of each test, what was observed,
  and every call the harness made. Print it to PDF from the browser: the print layout opens every section.
- `report.json`: everything in the page, for tools and archiving (library sha256, platform, per-test result with evidence, full call transcripts).
  Values given to `--init-config` or `docker -e` are never stored, and directories are removed from every message.
- `junit.xml`: for CI.

Exit codes of both commands: `0` no MUST check failed; `1` a MUST failed; `2` nothing failed but a MUST check could not run or was skipped with `--skip` or `--hardware-safe` (INCOMPLETE: for
example `init` needs vendor keys, pass them with `--init-config`); `3` bad command line, missing file, Docker or image; `4` a check of the suite itself
crashed (a bug of the suite: please report it). Failures name the spec clause. The suite tests itself against a mock runtime and a mock toolchain with
injected faults: `python -m pytest`.

### Toolchains

```bash
oaax-conformance toolchain oaax-<vendor>-toolchain:<tag> [--onnx model.onnx] [--runtime ./libRuntimeLibrary.so --init-config key=value] [--docker-arg ARG]... [--out DIR]
```

Needs Docker. The suite converts a small built-in model (or `--onnx FILE`, a model this toolchain can convert) in temporary directories, runs the
image as an arbitrary unprivileged user, without network, with read-only mounts and on hostile inputs (empty, garbage, truncated, missing, a zip
with `../` members). With `--runtime LIB` it then loads the converted model in that runtime and compares the results with onnxruntime
(`pip install oaax-conformance[e2e]`). `--docker-arg` adds arguments to every `docker run`, e.g. `--docker-arg=--gpus=all`.

See [docs/design.md](docs/design.md) for the architecture and [docs/tests/](docs/tests/) for the test specifications (under review).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0, see [LICENSE](LICENSE).

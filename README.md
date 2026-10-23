# OAAX conformance suite

Checks whether an OAAX **runtime** (shared library) and/or **toolchain** (Docker image) conforms to the
[OAAX specification](https://oaax-standard.github.io/OAAX/Specification/).

> **Status: early.** Batch 1 (20 runtime tests: static + lifecycle, Linux) runs. Toolchain and model tiers are specified in
> [docs/tests/](docs/tests/) as they get approved, then implemented.

## Principles

- **Easy for implementers**: one command, no setup beyond Docker/Python, readable failures: each failure names the spec
  clause and says what to fix.
- **Dead simple for users**: a pass means "this runtime/toolchain can be swapped for another one without code changes".
- **Strict on the raw API**: the suite talks to the runtime's C API directly, never through a tolerant wrapper.
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
pip install -e .
oaax-conformance runtime ./libRuntimeLibrary.so [--init-config key=value]... [--model FILE] [--input NAME:DTYPE:DIMS]... [--hardware-safe] [--ignore-thread NAME] [--out DIR]
```

`--model FILE` is a model the runtime can load; it enables the tests that need one (they are `BLOCKED` without it). To send requests the suite
must know what the model takes: it reads an `oaax-manifest.json` next to the model, or the `.onnx` file itself (`pip install oaax-conformance[onnx]`),
or you give each input with `--input NAME:DTYPE:DIMS`. `--hardware-safe` is for shared or fragile devices: few cycles, small floods and a pause
between scenarios.

Three outputs: a **text summary** on the console (a table, one row per test), and with `--out DIR`:

- `report.html`: a self-contained page (works offline) with the verdict, the expected behaviour and spec clause of each test, what was observed,
  and every call the harness made. Print it to PDF from the browser: the print layout opens every section.
- `report.json`: everything in the page, for tools and archiving (library sha256, platform, per-test result with evidence, full call transcripts).
  Values given to `--init-config` are never stored, and neither is the library's directory.
- `junit.xml`: for CI.

Exit code: `0` no MUST check failed, `1` a MUST failed, `2` nothing failed but a MUST check could not run (for example `init` needs vendor
keys: pass them with `--init-config`). Failures name the spec clause. The suite tests itself against a mock runtime with
injected faults: `python -m pytest`.

See [docs/design.md](docs/design.md) for the architecture and [docs/tests/](docs/tests/) for the test specifications (under review).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0, see [LICENSE](LICENSE).

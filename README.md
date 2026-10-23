# OAAX conformance suite

Checks whether an OAAX **runtime** (shared library) and/or **toolchain** (Docker image) conforms to the
[OAAX specification](https://oaax-standard.github.io/OAAX/Specification/).

> **Status: scaffolding.** Nothing is runnable yet. This README is the design the suite is built to.

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

## Planned usage

```bash
oaax-conformance runtime ./libRuntimeLibrary.so [--model model.zip] [--spec 2.0]
oaax-conformance toolchain oaax-<vendor>-toolchain:tag
```

The report is a JSON file (suite version, spec version, library sha256, platform, results) plus a JUnit XML for CI.

See [docs/design.md](docs/design.md) for the architecture and [docs/tests/](docs/tests/) for the test specifications (under review).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0, see [LICENSE](LICENSE).

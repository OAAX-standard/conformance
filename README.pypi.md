# oaax-conformance

Checks whether an [OAAX](https://docs.oaax.org) **runtime** (a shared library) or **toolchain** (a Docker image) follows the [OAAX specification](https://oaax-standard.github.io/OAAX/Specification/).
A pass means: *this runtime or toolchain can be swapped for another one without code changes.*

- **Strict:** talks to the runtime's C API directly, never through a tolerant wrapper.
- **Each failure says what to fix:** it names the specification clause.
- **Safe on real hardware:** time-budgeted loops, capped request sizes, a slow device is `BLOCKED` and never `FAIL`, `--hardware-safe` leaves out the abusive tests.
- **No compiler needed:** the wheel contains the C harness (Linux x86_64 and aarch64).

## Install

Needs [uv](https://docs.astral.sh/uv/) on Linux, and Docker to test toolchains. The wheel needs Python 3.12 or newer; uv fetches it if your default is older.

```bash
uv tool install --python 3.12 'oaax-conformance[e2e]'
```

`[e2e]` adds `onnx` and `onnxruntime`: reading `.onnx` models and the end-to-end comparison. Without it you still get the toolchain tier and the runtime tier with `--input`.

## Use

```bash
# a runtime: the .so and a model it can load
oaax-conformance runtime ./libRuntimeLibrary.so --model model.onnx [--init-config key=value]... [--out DIR]

# a toolchain: its Docker image, optionally with a runtime for the end-to-end check
oaax-conformance toolchain oaax-<vendor>-toolchain:<tag> [--runtime ./libRuntimeLibrary.so] [--out DIR]
```

No `--model`: the tests that send requests are `BLOCKED` and the run is `INCOMPLETE`. Open `report.html` in `--out` to see what each check did.

Try it without any vendor files, on the reference CPU runtime (about 20 seconds, `@tiny-conv` is a small model the wheel ships):

```bash
mkdir -p /tmp/rt && curl -sSL https://nbg1.your-objectstorage.com/oaax/runtimes/2.0.0/CPU/x86_64/Ubuntu/library.tar.gz | tar xz -C /tmp/rt   # arm machine: aarch64
oaax-conformance runtime /tmp/rt/libRuntimeLibrary.so --model @tiny-conv --out /tmp/report-rt
```

The reference runtime has [known issues](https://github.com/OAAX-standard/conformance#known-findings-on-the-reference-implementation), so exit code 1 is the expected result here. A minimal machine needs `libgomp1` and `binutils`.

## What it checks

| Tier | Needs | Checks |
| ---- | ----- | ------ |
| `runtime/static` | the `.so` only | exported symbols, getters, calls before `init` |
| `runtime/lifecycle` | a working `init` | `init`/`cleanup` rules, error text, cycles, threads, unload, memory, exit |
| `runtime/model` | a model the runtime can load | enqueue/retrieve, id echo, multi-model routing, timeouts, concurrency, input validation, leaks |
| `toolchain` | the Docker image | arguments, exit codes, log, runs without root / network / with a read-only input, hostile input |
| `e2e` | both | convert, load, run, compare with the ONNX reference |

Every test has a **level**: **MUST** (required by the specification: a failure fails the run), **SHOULD** (recommended: a warning) or **PROPOSED** (planned for the next version: a warning). Every test is specified, with the quoted clause, in the [test documentation](https://github.com/OAAX-standard/conformance/tree/main/docs/tests).

## Results

| Exit code | Meaning |
| --------- | ------- |
| `0` | no MUST check failed |
| `1` | a MUST failed |
| `2` | `INCOMPLETE`: nothing failed, but a MUST could not run (for example `init` needs vendor keys: pass them with `--init-config`) |
| `3` | bad command line, missing file, Docker or image |
| `4` | a check of the suite itself crashed: please report it |

With `--out DIR` it writes `report.html` (self-contained, prints to PDF), `report.json` (for tools and archives) and `junit.xml` (for CI). Reports never contain the values of `--init-config` or of docker environment variables.

## Links

- [Full README](https://github.com/OAAX-standard/conformance) · [Reference: all options](https://github.com/OAAX-standard/conformance/blob/main/docs/reference.md) · [Issues](https://github.com/OAAX-standard/conformance/issues) · [Changelog](https://github.com/OAAX-standard/conformance/blob/main/CHANGELOG.md)
- Fetch the runtimes and toolchains to test: [`oaax-manager`](https://pypi.org/project/oaax-manager/).

Apache 2.0.

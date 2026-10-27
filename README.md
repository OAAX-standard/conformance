# OAAX conformance suite

Checks whether an OAAX **runtime** (shared library) and/or **toolchain** (Docker image) conforms to the
[OAAX specification](https://oaax-standard.github.io/OAAX/Specification/).

> **Status: 0.1.0, Linux only.** The runtime tiers (static, lifecycle, model) and the toolchain / end-to-end tiers run; every test is specified in
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

## Install (Linux)

You need `git`, [uv](https://docs.astral.sh/uv/), Docker (toolchain tier only) and, for the runtime tier, `cmake` and a C compiler (`sudo apt install build-essential cmake`).

```bash
git clone https://github.com/OAAX-standard/conformance && cd conformance
scripts/build.sh                       # builds the C harness of the runtime tier (and the mock runtimes of the self-tests) into ./build
uv sync                                # creates .venv; uv run oaax-conformance --version
```

The wheel of a release needs Python 3.12 or newer: `uv tool install --python 3.12 oaax_conformance-X.Y.Z-py3-none-any.whl` (uv fetches that Python if your default is older).
It is enough for the **toolchain** tier without `--runtime`. It does not contain the C harness, so the **runtime** tier and the end-to-end toolchain tests (`toolchain IMAGE --runtime LIB`)
need a checkout (`--harness PATH` or `OAAX_HARNESS` point to a harness built elsewhere). Reading `.onnx` models (`--onnx`, the runtime tier's `--model`) needs the `onnx` extra:
`uv tool install --python 3.12 'oaax_conformance-X.Y.Z-py3-none-any.whl[onnx]'`; a checkout has it after `uv sync`.

## Usage

```bash
uv run oaax-conformance runtime ./libRuntimeLibrary.so [--init-config key=value]... [--model FILE] [--input NAME:DTYPE:DIMS]... [--hardware-safe] [--out DIR]
```

Without `--model` the tests that send requests are `BLOCKED` and the run is `INCOMPLETE` (exit 2).

### Try it on the reference CPU runtime

```bash
mkdir -p /tmp/rt && curl -sSL https://nbg1.your-objectstorage.com/oaax/runtimes/2.0.0/CPU/x86_64/Ubuntu/library.tar.gz | tar xz -C /tmp/rt
uv run oaax-conformance runtime /tmp/rt/libRuntimeLibrary.so --model python/oaax_conformance/models/tiny-conv.onnx --out /tmp/report-rt     # ~20 s
```

The CPU runtime loads any `.onnx` file, here the small model shipped with the suite. For a real device pass a model in the format that runtime loads (the output of its toolchain).
This runtime has known findings (issues
[#45](https://github.com/OAAX-standard/reference-implementation/issues/45) to [#49](https://github.com/OAAX-standard/reference-implementation/issues/49)),
so exit code 1 is the **expected** result: `RT-L-11` WARN (exit without `cleanup` crashes), `RT-M-19`, `-20`, `-21`, `-23`, `-25` FAIL (requests that do not match the model are accepted and lost),
`RT-L-13`/`-14`/`-15`, `RT-M-26`, `-32`, `-34` WARN, `RT-M-17` BLOCKED (the tiny model answers faster than the suite can time). `RT-M-35` (memory growth over init/load/cleanup cycles, #47) is an RSS measurement:
it shows with some models (a 1 MB face detector grows 12 MB) and not with others, so a PASS there is weaker than a FAIL. Open `report.html` in `--out` to see what each check did.

### Reading the results

| Status | Meaning |
| ------ | ------- |
| `PASS` | the behaviour is as the spec clause says |
| `FAIL` | a MUST is broken: the message names the clause and what to fix. Exit code 1 |
| `WARN` | a SHOULD or PROPOSED check differs: worth fixing, never fails the run |
| `BLOCKED` | the check could not run (no model, slow device, missing vendor key): the message says what to pass. A blocked MUST makes the run `INCOMPLETE` (exit 2) |
| `N/A` | the check does not apply (for example no second input to leave out) |

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
  Values given to `--init-config` or as docker environment variables (`-e`, `--env`, in any spelling, also `-e NAME` taken from your environment) are never stored, and directories are removed from every message. Other docker flags (`--label`, `--add-host`...) are shown as given, so keep secrets out of them; `--env-file` is refused.
- `junit.xml`: for CI.

Exit codes of both commands: `0` no MUST check failed; `1` a MUST failed; `2` nothing failed but a MUST check could not run or was skipped with `--skip` or `--hardware-safe` (INCOMPLETE: for
example `init` needs vendor keys, pass them with `--init-config`); `3` bad command line, missing file, Docker or image; `4` a check of the suite itself
crashed (a bug of the suite: please report it). `130`, `143`, `129`: stopped by Ctrl-C, SIGTERM or SIGHUP (the harness and any containers are stopped first). Failures name the spec clause. The suite tests itself against a mock runtime and a mock toolchain with
injected faults: `uv run pytest`.

### Toolchains

```bash
curl -sSL -o /tmp/oaax-cpu-toolchain.tar https://nbg1.your-objectstorage.com/oaax/conversion-toolchain/2.0.0/CPU/oaax-cpu-toolchain.tar && docker load -i /tmp/oaax-cpu-toolchain.tar   # ~800 MB
uv run oaax-conformance toolchain oaax-cpu-toolchain:2.0.0 --runtime /tmp/rt/libRuntimeLibrary.so --out /tmp/report-tc     # ~30 s
```

The reference CPU toolchain also has known findings ([#49](https://github.com/OAAX-standard/reference-implementation/issues/49)): expect exit code 1 with `TC-30` FAIL (an empty `.onnx` is accepted), `TC-35` FAIL
(no log after a failed conversion), `TC-04`/`TC-05` WARN (it writes `model-simplified.onnx` into its input directory) and `TC-23` WARN (no manifest: planned for 2.1).

General form:

```bash
uv run oaax-conformance toolchain oaax-<vendor>-toolchain:<tag> [--onnx model.onnx] [--runtime ./libRuntimeLibrary.so --init-config key=value] [--docker-arg ARG]... [--out DIR]
```

Needs Docker. The suite converts a small built-in model (or `--onnx FILE`, a model this toolchain can convert) in temporary directories, runs the
image as an arbitrary unprivileged user, without network, with read-only mounts and on hostile inputs (empty, garbage, truncated, missing, a zip
with `../` members). With `--runtime LIB` it then loads the converted model in that runtime and compares the results with onnxruntime
(`uv sync` has it, and it needs the C harness of a checkout; the wheel alone cannot run these tests, see Install). `--docker-arg` adds arguments to every `docker run`, e.g. `--docker-arg=--gpus=all`. The suite keeps only the tail of what a toolchain or runtime prints, but does not cap the files a toolchain writes to its output directory (compiled models can be several GB): leave the disk some room. After a run, if the toolchain left files it owns that the suite cannot delete, the suite runs the same image once more as root with `--network none`, mounting only that workspace, to `chmod` them (an image without `chmod` leaves the directory; the suite then lists it).

See [docs/design.md](docs/design.md) for the architecture and [docs/tests/](docs/tests/) for the test specifications (under review).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0, see [LICENSE](LICENSE).

# OAAX conformance suite

Checks whether an OAAX **runtime** (shared library) and/or **toolchain** (Docker image) conforms to the
[OAAX specification](https://oaax-standard.github.io/OAAX/Specification/).

> **Status: 0.1.0, Linux only.** Every test is specified in [docs/tests/](https://github.com/OAAX-standard/conformance/blob/main/docs/tests/): its level, the quoted spec clause, and what the C++ CPU runtime and toolchain do today.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) on Linux (x86_64 or aarch64); Docker for toolchains. No compiler: the wheel contains the C harness.

```bash
uv tool install --python 3.12 'oaax-conformance[e2e]'   # all tiers; or run once: uvx --python 3.12 --from 'oaax-conformance[e2e]' oaax-conformance ...

# your own runtime or toolchain
oaax-conformance runtime ./libRuntimeLibrary.so --model FILE [--init-config key=value]... [--out DIR]
oaax-conformance toolchain oaax-<vendor>-toolchain:<tag> [--runtime ./libRuntimeLibrary.so] [--out DIR]
```

No `--model`: the tests that send requests are `BLOCKED` and the run is `INCOMPLETE` (exit 2).
Open `report.html` in `--out` to see what each check did.

### Try it on the reference CPU runtime and toolchain

```bash
mkdir -p /tmp/rt && curl -sSL https://nbg1.your-objectstorage.com/oaax/runtimes/2.0.0/CPU/x86_64/Ubuntu/library.tar.gz | tar xz -C /tmp/rt   # aarch64 machine: use aarch64 in the URL
oaax-conformance runtime /tmp/rt/libRuntimeLibrary.so --model @tiny-conv --out /tmp/report-rt   # ~20 s

curl -sSL -o /tmp/oaax-cpu-toolchain.tar https://nbg1.your-objectstorage.com/oaax/conversion-toolchain/2.0.0/CPU/oaax-cpu-toolchain.tar && docker load -i /tmp/oaax-cpu-toolchain.tar   # ~800 MB
oaax-conformance toolchain oaax-cpu-toolchain:2.0.0 --runtime /tmp/rt/libRuntimeLibrary.so --out /tmp/report-tc   # ~30 s
```

`@tiny-conv` is the small model the wheel ships. A minimal machine needs `libgomp1` (the reference runtime's own dependency) and `binutils` (`nm`, `readelf`, for `RT-S-02`, which is `BLOCKED` without them).

**Exit code 1 is the expected result** for both: they have [known findings](#known-findings-on-the-reference-implementation).

## What it checks

- **Easy for implementers**: one command, no setup beyond Docker/Python; each failure names the spec clause and says what to fix.
- **Dead simple for users**: a pass means "this runtime/toolchain can be swapped for another one without code changes".
- **Strict**: the suite talks to the runtime's C API directly, never through a tolerant wrapper.
- **Safe on real hardware**: loops stop at a time budget, request sizes are capped, a slow device is `BLOCKED` and never `FAIL`, and `--hardware-safe` leaves out the abusive tests.
- **Runs anywhere**: the static tier needs no accelerator, no model and no `init`.

### Tiers

| Tier | Needs | Checks |
| ---- | ----- | ------ |
| `runtime/static` | the `.so` / `.dll` only | exported symbols, getters, calls before `init` |
| `runtime/lifecycle` | a working `init` (device, or the CPU runtime) | `init`/`cleanup` rules, error text, cycles, threads, unload, memory, exit |
| `runtime/model` | a model the runtime can load (+ the device) | enqueue/retrieve, id echo, multi-model routing, timeouts, concurrency, input validation, leaks |
| `toolchain` | the Docker image | arguments, exit codes, log and manifest, runs without root / without network / read-only, output integrity, hostile input |
| `e2e` | both | convert, load, run, compare with the ONNX reference (`rtol 1e-4`, `atol 1e-5`: a fixed tolerance, no precision is declared yet) |

### The toolchain tier at a glance

The suite converts a small built-in model (or `--onnx FILE`) in temporary directories and runs your image:

- **Sandboxed by default**: `--cap-drop ALL`, `no-new-privileges` and a process limit on every run (`--docker-arg` can add back what an image really needs, which is worth reporting to its vendor).
- **Repeated under restrictions**: as an arbitrary unprivileged user (TC-02), without network (TC-03), with a read-only input (TC-04), a read-only root file system (TC-06) and an unwritable output (TC-38). Network stays on in the other runs: a toolchain that needs it is a finding, not a block.
- **Hostile inputs**: empty, garbage, truncated, missing, a zip with `../` members.
- **End to end** (`--runtime LIB`): loads the converted model in that runtime and compares the results with onnxruntime.

More in [docs/reference.md](https://github.com/OAAX-standard/conformance/blob/main/docs/reference.md#the-toolchain-tier).

### Levels

| Level | Meaning | A failure is |
| ----- | ------- | ------------ |
| **MUST** | required by the specification text of the version under test | a failed run |
| **SHOULD** | recommended by the text | a warning |
| **PROPOSED** | planned for the next version, so the suite can lead the specification | always a warning |

## Install

| You want | Do | Covers |
| -------- | -- | ------ |
| Everything | `uv tool install --python 3.12 'oaax-conformance[e2e]'` (PyPI) | all tiers |
| The same from a release | `uv tool install --python 3.12 'oaax_conformance-X.Y.Z-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl[e2e]'` (`aarch64` on an arm machine) | all tiers |
| Less to download | the same without `[e2e]` | the toolchain tier and the runtime tier with `--input` (no `.onnx` reading, no end-to-end comparison) |
| Development | a checkout: `scripts/build.sh && uv sync` | all tiers, [CONTRIBUTING](https://github.com/OAAX-standard/conformance/blob/main/CONTRIBUTING.md) |

- The wheel needs Python 3.12 or newer (uv fetches it if your default is older) and contains the C harness (Linux x86_64 and aarch64). `--harness PATH` or `OAAX_HARNESS` override it.
- `e2e` adds `onnx` and `onnxruntime`: reading `.onnx` models (`--onnx`, the runtime tier's `--model`) and the end-to-end comparison.

## Reading the results

| Status | Meaning |
| ------ | ------- |
| `PASS` | the behaviour is as the spec clause says |
| `FAIL` | a MUST is broken: the message names the clause and what to fix (exit 1) |
| `WARN` | a SHOULD or PROPOSED check differs: worth fixing, never fails the run |
| `BLOCKED` | the check could not run (no model, slow device, missing vendor key): the message says what to pass. A blocked MUST makes the run `INCOMPLETE` (exit 2) |
| `N/A` | the check does not apply (for example no second input to leave out) |

| Exit code | Meaning |
| --------- | ------- |
| `0` | no MUST check failed |
| `1` | a MUST failed |
| `2` | `INCOMPLETE`: nothing failed, but a MUST could not run or was skipped (`--skip`, `--hardware-safe`); for example `init` needs vendor keys: pass them with `--init-config` |
| `3` | bad command line, missing file, Docker or image |
| `4` | a check of the suite itself crashed: a bug of the suite, please report it |
| `129` `130` `143` | stopped by SIGHUP, Ctrl-C or SIGTERM (the harness and any containers are stopped first) |

With `--out DIR` the suite writes:

| File | For | Contains |
| ---- | --- | -------- |
| `report.html` | people | a self-contained page (works offline, prints to PDF with every section open): verdict, expected behaviour and spec clause per test, what was observed, every call the harness made |
| `report.json` | tools, archives | everything in the page, plus library sha256, platform and full call transcripts |
| `junit.xml` | CI | `failures` = failed; `errors` = crashed checks and MUSTs that could not run; `skipped` = other blocked checks and every n/a (with the reason); passed = pass + warn |

Reports never contain the values of `--init-config` or of docker environment variables (values shorter than 3 characters excepted: the suite says so at the start), and directories are removed from every message
([details](https://github.com/OAAX-standard/conformance/blob/main/docs/reference.md#what-reports-keep-out)).

## Known findings on the reference implementation

The reference CPU runtime and toolchain have known issues ([#45](https://github.com/OAAX-standard/reference-implementation/issues/45) to
[#49](https://github.com/OAAX-standard/reference-implementation/issues/49)), so these results are **expected**:

| Subject | Check | Result | Why |
| ------- | ----- | ------ | --- |
| runtime | `RT-L-11` | WARN | exit without `cleanup` crashes |
| runtime | `RT-M-19`, `-21`, `-23`, `-25` | FAIL | requests that do not match the model are accepted and lost |
| runtime | `RT-M-20`, `-22`, `-24` | N/A | one input with this model: nothing to leave out |
| runtime | `RT-L-13`/`-14`/`-15`, `RT-M-26`, `-32` | WARN | |
| runtime | `RT-M-34` | WARN, sometimes PASS | `requests_in_flight` is read while the runtime drains its queue (it reports 1 or 2 where 3 requests are unretrieved): a race, about one run in five passes. The totals therefore vary between runs: 47 pass, 7 warn, 4 fail, 1 blocked, 3 n/a, or 48 pass, 6 warn |
| runtime | `RT-M-17` | BLOCKED | the tiny model answers faster than the suite can time |
| runtime | `RT-M-35` | varies | memory growth is an RSS measurement (#47): a 1 MB face detector grows 12 MB, other models do not, so a PASS is weaker than a FAIL |
| toolchain | `TC-30` | FAIL | an empty `.onnx` is accepted |
| toolchain | `TC-35` | FAIL | no log after a failed conversion |
| toolchain | `TC-04`, `TC-05` | WARN | it writes `model-simplified.onnx` into its input directory |
| toolchain | `TC-23` | WARN | no manifest (planned for 2.1) |

## More

- [docs/reference.md](https://github.com/OAAX-standard/conformance/blob/main/docs/reference.md): all options, what reports keep out, `--docker-arg` rules, how the toolchain tier works, disk use.
- [docs/design.md](https://github.com/OAAX-standard/conformance/blob/main/docs/design.md): architecture and direction for OAAX 2.1.
- [docs/tests/](https://github.com/OAAX-standard/conformance/blob/main/docs/tests/): the test specifications (under review).
- [CONTRIBUTING.md](https://github.com/OAAX-standard/conformance/blob/main/CONTRIBUTING.md), [SECURITY.md](https://github.com/OAAX-standard/conformance/blob/main/SECURITY.md), [CHANGELOG.md](https://github.com/OAAX-standard/conformance/blob/main/CHANGELOG.md).
- The suite tests itself against a mock runtime and a mock toolchain with injected faults: `uv run pytest`.

## License

Apache 2.0, see [LICENSE](https://github.com/OAAX-standard/conformance/blob/main/LICENSE).

# Reference

Everything the [README](../README.md) leaves out: options, what reports keep out, the `--docker-arg` rules and the details of the toolchain tier.

- [Options](#options)
- [Reports](#reports)
- [What reports keep out](#what-reports-keep-out)
- [`--docker-arg` rules](#--docker-arg-rules)
- [The toolchain tier](#the-toolchain-tier)

## Options

All of them are in `--help`.

### Both commands

| Option | What it does |
| ------ | ------------ |
| `--out DIR` | write `report.html`, `report.json` and `junit.xml` |
| `--init-config key=value` | a vendor key for the runtime's `init` (repeatable); never stored in reports |
| `--timeout SEC` | per scenario, default 300. Loops stop at 60 % of it; a call still running at the limit is `BLOCKED`, not failed |
| `--round-trip-timeout SEC` | also for slow devices (see `--help`) |
| `--skip ID` | leave a test out (a skipped MUST makes the run `INCOMPLETE`, exit 2) |
| `--only ID` | run only these tests. The console says "Only N of M approved checks were selected"; JSON records it as `summary.checks_selected` and JUnit as the `checks_selected` property. The exit code covers only the checks that ran |
| `--quiet` | hide the progress lines on stderr |

### `runtime`

| Option | What it does |
| ------ | ------------ |
| `--model FILE` | a model the runtime can load; enables the tests that send requests |
| `--input NAME:DTYPE:DIMS` | what the model takes (`?` = dynamic dimension). The suite otherwise reads an `oaax-manifest.json` next to the model, or the `.onnx` file itself; `--input` overrides both, which is also how to describe a runtime that takes another data type or layout than the ONNX model |
| `--hardware-safe` | for real devices: 3 cycles instead of the full count, smaller floods, a pause between scenarios, and no abusive tests (NULL pointers, exit without `cleanup`, `cleanup` during a blocked call, request flood) |
| `--workdir DIR` | run the harness in this directory |
| `--harness PATH` (or `OAAX_HARNESS`) | a C harness built elsewhere |

`--hardware-safe` and memory growth: a check that passes on fewer cycles than its title says carries a `note` (console Details, JSON, JUnit, HTML).
The two memory-growth checks (`RT-L-10`, `RT-M-35`) need at least 10 cycles and are `BLOCKED` with a hint otherwise, so a `--hardware-safe` run does not measure memory growth.

### `toolchain`

| Option | What it does |
| ------ | ------------ |
| `--onnx FILE` | the model to convert (default: a small built-in one); it must be one this toolchain can convert |
| `--runtime LIB` | enables the end-to-end tests; needs the C harness of a checkout |
| `--model-file NAME` | which output file is the model, relative to the output directory (see [model selection](#end-to-end-model-selection)) |
| `--docker-arg ARG` | added to every `docker run`, e.g. `--docker-arg=--gpus=all` (see [the rules](#--docker-arg-rules)) |

## Reports

| File | Contents |
| ---- | -------- |
| `report.html` | a self-contained page (works offline). Print it to PDF from the browser: the print layout opens every section. Verdict, expected behaviour and spec clause of each test, what was observed, every call the harness made |
| `report.json` | everything in the page, for tools and archiving: library sha256, platform, per-test result with evidence, full call transcripts |
| `junit.xml` | for CI. `failures` = failed; `errors` = crashed checks and MUSTs that could not run; `skipped` = the other blocked checks and every n/a (with the reason); passed = pass + warn. The properties `outcome_pass`, `outcome_warn`... give the same counts as the console and `report.json` |

## What reports keep out

- **Never stored**: the values of `--init-config`, and the values of docker environment variables in every spelling docker accepts:
  `-e NAME=V`, `-eNAME=V`, `-e=NAME=V`, `--env NAME=V`, `--env=NAME=V`, clusters such as `-te NAME=V`, and also `-e NAME` (the value is taken from your environment).
- **Removed from every message**: directory names.
- **Shown as given**: other docker flags (`--label`, `--add-host`...), so keep secrets out of them.
- **Refused**: `--env-file` (its values cannot be kept out of the reports), and a cluster of short flags the suite cannot read with certainty (exit 3).
- Secrets are removed before a text is cut to its tail, never after.
- **Values shorter than 3 characters are not masked** (`--init-config pin=42`, `-e A=1`): a one-character value would turn every such character of every message into `***`. The suite says so at the start ("note: the value of pin is too short to be hidden in reports"). Give a secret a longer value, or keep it out of the options.
- A refusal message never echoes an environment value: an argument that carries one is printed masked (`-dteTOKEN=***`), a cluster the reader cannot parse only by its flag letters.

## `--docker-arg` rules

The suite sets some flags itself, so a value from you would change what the tests test. These are refused, in every spelling (`-u0`, `-tu 0`, `-v/x:/y`, `--user=0`):

`--user`/`-u`, `-v`/`--volume`/`--mount`, `--network`/`--net`, `--entrypoint`, `--read-only`, `--privileged`, `--rm`, `--name`.

Flags that change how a foreground `docker run` behaves are refused too (exit 3, one line), because they gave false MUST FAILs: `-d`/`--detach` (docker returns the container id at once and the suite would read `/output` while the toolchain still runs), `-a`/`--attach`, `--restart` (docker: "conflicting options: cannot specify both --restart and --rm"), `--cidfile` (the second of the ~30 conversions fails with "container ID file found"), and `-i`/`--interactive` together with `-t`/`--tty` ("cannot attach stdin to a TTY-enabled container": the suite's stdin is `/dev/null`; either one alone is fine, and `-dit` clusters are read like `-d -i -t`). `--tmpfs`, `--init`, `--stop-signal` and `--sig-proxy` are accepted: docker runs fine with them.

The suite's own defaults on every conversion are `--cap-drop ALL`, `--security-opt no-new-privileges` and a process limit; `--docker-arg` may override those (a toolchain that needs a capability is worth reporting to its vendor).

## The toolchain tier

### What runs

1. The suite converts a small built-in model (or `--onnx FILE`) in temporary directories, running your image as your user id with all capabilities dropped.
2. It repeats the conversion as an arbitrary unprivileged user (TC-02), without network (TC-03), with a read-only input (TC-04), a read-only root file system (TC-06) and an unwritable output (TC-38). Network stays on in the other runs, because a toolchain that needs it is a finding, not a block.
3. It runs hostile inputs: empty, garbage, truncated, missing, and a zip with `../` members.
4. With `--runtime LIB` it loads the converted model in that runtime and compares the results with onnxruntime (`uv sync` has it; it needs the C harness of a checkout, so the wheel alone cannot run these tests).

### End-to-end model selection

A 2.0 toolchain has no manifest naming its model. If it writes several files besides its log, the suite picks the model in this order and says which file it tested (in a note on every end-to-end result):

1. `--model-file NAME`, relative to the output directory
2. the model named by a manifest
3. a file named `model.*`
4. the first file, in name order, that the runtime loads

All four end-to-end tests judge the same file.

### Disk use

- Each conversion works in a temporary directory under `$TMPDIR`: point it at a disk with room. A `TMPDIR` containing `:` is refused (docker cannot mount it).
- The big files of a conversion are deleted as soon as it is done, except those of the first valid conversion (the end-to-end tests load it). Plan for about two to three times the size of your model.
- The suite keeps only the tail of what a toolchain or runtime prints, but does not cap the files a toolchain writes to its output directory (compiled models can be several GB).
- Files a toolchain writes are read only when they are regular files, never through a link, and no more than 10 MB for logs and manifests.

### Cleanup

If the toolchain left files it owns that the suite cannot delete, the suite runs the same image once more as root with `--network none`, mounting only that workspace, to `chmod` them. An image without `chmod` leaves the directory, and the suite lists it.

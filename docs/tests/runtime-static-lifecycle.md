# Batch 1: runtime without a model

Spec version under test: **2.0**. Source = `interface.h` (normative) and the Runtime Specification page.
Probed = reference CPU, Intel, Hailo and PythonBindings CPU 2.0 runtimes, run by hand on a machine without an accelerator.

## Levels

The level says **how the spec text backs the test**, not how important the behaviour is.

| Level | Meaning | If the runtime fails it | Exit code |
| ----- | ------- | ----------------------- | --------- |
| **MUST** | The spec text of the version under test requires it. The Source column quotes the sentence. | `FAIL`: the runtime does not conform. | `1` |
| **SHOULD** | The text recommends it, or it follows from the text without being written (the Source column says "derived" and why). | `WARN`: worth fixing, does not fail the run. | `0` |
| **PROPOSED** | The text says nothing; it is the behaviour planned for the next spec version. | `WARN`: the suite is ahead of the spec. | `0` |

A MUST test that could not run (for example `init` fails without vendor keys) is `BLOCKED` and makes the run `INCOMPLETE` (exit code `2`). A test without a quoted source is never MUST.

## `RT-S`: static (library only, no `init`, no device)

| ID | Check | Pass when | Level | Source | Probed | Status |
| -- | ----- | --------- | ----- | ------ | ------ | ------ |
| RT-S-01 | Exports | the 9 `runtime_*` functions all resolve (`dlsym` / `GetProcAddress`) | MUST | header declares them; spec "implements the OAAX runtime C API" | 4/4 | approved |
| RT-S-02 | Only those exports | no other exported symbol (compiler-generated ones allow-listed; no `GNU_UNIQUE` on Linux) | PROPOSED | none: hygiene (leaked ORT/protobuf symbols collide with the host) | not probed | approved |
| RT-S-03 | Version string | `runtime_get_version()` before `init` is non-NULL and valid SemVer | MUST | header: "never NULL", "SemVer-compliant", "any time, even before init" | 4/4 | approved |
| RT-S-04 | Name string | `runtime_get_name()` before `init` is non-NULL and not empty | MUST | header: "never NULL", "valid null-terminated name string" | 4/4 | approved |
| RT-S-05 | Getters are stable | two calls of each getter return equal strings | MUST | header: "valid for the entire lifetime of the runtime" | not probed | approved |
| RT-S-06 | Info before `init` | `runtime_get_info()` is NULL | MUST | header: "Must return NULL if the runtime is not initialized" | 4/4 | approved |
| RT-S-07 | No error yet | in a fresh process `runtime_get_error()` is NULL | MUST | header: "NULL if no error has occurred" | 4/4 | approved |
| RT-S-08 | Cleanup before `init` | `runtime_cleanup()` returns SUCCESS, twice | MUST | header: "Must be idempotent (safe to call multiple times)" | 4/4 | approved |
| RT-S-09 | Calls before `init` | `load_models`, `enqueue_input`, `retrieve_output(timeout 0)` neither crash nor return SUCCESS; the status is `NOT_INITIALIZED`; a rejected input is freed by the harness | SHOULD | header: caller must `init` first (behaviour otherwise unspecified); `NOT_INITIALIZED` exists for this | 3/3 probed (reference CPU, Intel, Hailo) return `NOT_INITIALIZED` for all three calls | approved |

## `RT-L`: lifecycle (`init` must succeed: needs the device, or the CPU runtime)

Config for `init` is empty unless the user passes `--init-config key=value` (vendor keys). If `init` fails, the group is reported as **blocked**, not failed.

| ID | Check | Pass when | Level | Source | Probed | Status |
| -- | ----- | --------- | ----- | ------ | ------ | ------ |
| RT-L-01 | Unknown key | `init` with a key the runtime does not know returns SUCCESS | MUST | header: "Must ignore unknown configuration keys without error" | 4/4 | approved |
| RT-L-02 | Empty config | `init` with an empty `Config` returns SUCCESS | PROPOSED | none: "defaults work" (a user must not need vendor keys) | 4/4 | approved |
| RT-L-03 | Second `init` | returns `ALREADY_INITIALIZED`; the runtime keeps working (`get_info` non-NULL, `cleanup` SUCCESS) | MUST | header: "Must return ALREADY_INITIALIZED, and leave the existing state untouched" | 4/4 | approved |
| RT-L-04 | Info is JSON | after `init`, `get_info()` is non-NULL, valid UTF-8, parses as a JSON **object** | MUST | header: "valid null-terminated JSON object string when initialized" | 4/4 | approved |
| RT-L-05 | Error text | after the non-SUCCESS of RT-L-03, `get_error()` is non-NULL and not empty | MUST | header: "update the error message whenever a function returns an error status" | 4/4 | approved |
| RT-L-06 | Info after cleanup | `get_info()` is NULL after `cleanup` | MUST | header: NULL "if the runtime is not initialized" | 4/4 | approved |
| RT-L-07 | Re-init cycles | 50 x (`init` SUCCESS, `cleanup` SUCCESS) | MUST | header: runtime ready for `init` again after `cleanup` | 10 cycles, 4/4 | approved |
| RT-L-08 | Threads | thread count after `cleanup` returns to the count before the first `init` (settle up to 3 s) | SHOULD | derived: header "free all runtime-allocated memory and resources"; a surviving thread pins the DLL on Windows | 4/4 | approved |
| RT-L-09 | Unload | after `cleanup`, `dlclose` really unloads the library (it is gone from `/proc/self/maps`; Windows: module no longer loaded) | SHOULD | derived, same as RT-L-08; known pitfall: objects pinned by `NODELETE` / `GNU_UNIQUE` | not probed | approved |
| RT-L-10 | Memory | RSS after 50 cycles grows by less than 5 MB over RSS after the first cycle | SHOULD | derived, same as RT-L-08 | not probed | approved |
| RT-L-11 | Exit without cleanup | after `init`, the process exits with code 0 within 10 s (no crash or hang in static destructors) | PROPOSED | none: robustness rule | not probed | approved |

## Deliberately not tested in 2.0

The spec text is silent or declares the behaviour undefined, so a test would be guessing. Each is a candidate clarification for 2.1.

- Any call after `cleanup` except `init`, getters and `cleanup` (header: the caller "must not call any other runtime functions").
- `load_models` called twice in one session; exact status for a missing, corrupt or foreign model file (runtimes return `ERROR`, never `FILE_NOT_FOUND` / `INVALID_MODEL`).
- `get_error` after a later success (some runtimes keep the stale message).
- NULL pointers to `retrieve_output`.

# Batch 2: runtime with a model

Spec version under test: **2.0**. Source = `interface.h` (normative) and the Runtime Specification page. Levels:
see [runtime-static-lifecycle.md](runtime-static-lifecycle.md#levels).
Probed = the C++ CPU runtime (reference implementation build, 2.0.0), run by hand with the c-example face model (two inputs
`image-` FLOAT `[1,3,240,320]` and `nms_sensitivity-` FLOAT `[1]`, one output `bboxes-format:xyxysc;0:face`).

## What the tests need

`oaax-conformance runtime LIB --model FILE`. Without `--model` every test here is `BLOCKED`; if the runtime cannot load the model they are
`BLOCKED` too, never failed.

Tests that send inputs must know what the model takes (names, data types, shapes). The suite gets that from, in this order:

1. an `oaax-manifest.json` next to the model (the toolchain output described in the 2.1 manifest proposal), using the `native` view;
2. otherwise, for an `.onnx` model, the ONNX file itself (`pip install oaax-conformance[onnx]`);
3. otherwise the inference tests are `BLOCKED` ("cannot tell what the model takes").

Input data is deterministic (fixed seed, floats in `[0,1)`, integers small, booleans false). The tests check structure, routing and
behaviour, never accuracy (accuracy against the ONNX reference is the end-to-end tier).

## `RT-M`: loading and ownership

| ID | Check | Pass when | Level | Source | Probed | Status |
| -- | ----- | --------- | ----- | ------ | ------ | ------ |
| RT-M-01 | Second `load_models` | after a successful `load_models`, another call returns an error status (no crash); the runtime reports `ALREADY_INITIALIZED` today | PROPOSED | none: the header says loaded models are maintained "until runtime_cleanup()" but not what a later call does; models are replaced only by `cleanup` + `init` | pass | approved |
| RT-M-02 | Retry after a failed load | `load_models` with a missing file fails; a second call with the real model then succeeds without a new `init`, and inference works | MUST | header: "If a model fails to load, the caller can retry loading another config of models without having to re-call the runtime_init function"; "Must free all resources allocated during the call if any model fails to load" | pass | draft |
| RT-M-03 | Load from memory | `load_models` with `model_data` + `model_size` (no `file_path`) succeeds and a request is answered | MUST | header: "Either file_path or model_data/model_size must be provided to load the model"; spec: "from a file path or from memory" | pass | draft |
| RT-M-04 | Buffers are copied | after `load_models` the harness overwrites `model_data` and the path and config strings; a request is still answered | MUST | header: "Mustn't assume that the model_configs array, or any buffer it points to ... is accessible and valid after the call. Anything it needs later must be copied" | pass for `model_data`; path and config strings not probed | draft |
| RT-M-05 | Model id is the array index | two models loaded in one call (the same file twice): a request sent to id 0 and one to id 1 come back with `*model_id` 0 and 1 | MUST | header: "Models are identified by their index in the array (0, 1, 2, ...)"; "Must write the model that produced the output to *model_id" | pass | draft |
| RT-M-06 | Invalid model id | `enqueue_input` with id -1 and with id = number of loaded models returns an error status (no crash) | MUST | header: "Must validate model_id is within valid range and corresponds to a loaded model" | pass | draft |
| RT-M-07 | Invalid model id: code | the status of RT-M-06 is `INVALID_MODEL_ID` | SHOULD | the status exists in the header; the text does not say when to use it | pass (18) | draft |

## `RT-M`: one request, one answer

| ID | Check | Pass when | Level | Source | Probed | Status |
| -- | ----- | --------- | ----- | ------ | ------ | ------ |
| RT-M-08 | Round trip | a valid request is accepted; `retrieve_output` returns SUCCESS with the same `id`, the right `*model_id` and at least one tensor | MUST | header: "copying the id of the input Tensors that produced it"; "Must write the model that produced the output to *model_id" | pass | draft |
| RT-M-09 | Outputs are fully allocated | every output descriptor has a non-NULL, NUL-terminated `name`; `shape` non-NULL when `rank` > 0; `data` non-NULL when `data_size` > 0 | MUST | header: "Must allocate the tensors array and every descriptor's name, shape and data" | pass | draft |
| RT-M-10 | Outputs are separate allocations | all `name`, `shape` and `data` pointers of one output are pairwise distinct (the host frees each one) | MUST | header: "Each descriptor's name, shape, and data are separately allocated" | pass | draft |
| RT-M-11 | Output `data_size` | `data_size` equals the product of `shape` times the element size of `data_type` | SHOULD | derived: the header calls `data_size` the "Size of data in bytes; avoids recomputing from type+shape" | pass | draft |
| RT-M-12 | Output names | the output names equal the model's output names byte for byte (including characters such as `:` and `;`) | PROPOSED | none: the toolchain spec says the model keeps its I/O names; the runtime text does not repeat it | pass | draft |

## `RT-M`: `retrieve_output` timeouts (model loaded, nothing in flight)

| ID | Check | Pass when | Level | Source | Probed | Status |
| -- | ----- | --------- | ----- | ------ | ------ | ------ |
| RT-M-13 | Timeout 0 | returns `NO_OUTPUT_AVAILABLE` within 100 ms | MUST | header: "If zero, the function returns immediately if no output is available" | pass (0 ms) | draft |
| RT-M-14 | Timeout > 0 | `retrieve_output(200)` returns `NO_OUTPUT_AVAILABLE` and does not hang (back within the timeout + 1 s) | MUST | header: "Maximum time to wait for output in milliseconds"; "Returns RUNTIME_STATUS_NO_OUTPUT_AVAILABLE if no output is ready" | pass | draft |
| RT-M-15 | Timeout > 0 is honoured | the call of RT-M-14 took at least 90% of the timeout (a runtime that returns at once makes hosts spin) | SHOULD | derived: "maximum time to wait" | pass (200 ms) | draft |
| RT-M-16 | Timeout < 0 | with one request in flight, `retrieve_output(-1)` blocks until its output arrives and returns SUCCESS | MUST | header: "If negative, the function waits indefinitely until output is available" | pass | draft |
| RT-M-17 | Enqueue does not block | enqueueing 8 prepared requests back to back takes less than half the time of 8 sequential requests (the harness builds the tensors beforehand) | SHOULD | header: "Must process inference asynchronously, ie. it must not block the caller"; SHOULD because it is a timing heuristic | not probed (my first measurement included building the tensors) | draft |

## `RT-M`: input validation

Every case sends a request that does not match the model and expects an error status from `enqueue_input`, never SUCCESS followed by a request that
disappears.

| ID | Check | Pass when | Level | Source | Probed | Status |
| -- | ----- | --------- | ----- | ------ | ------ | ------ |
| RT-M-18 | NULL tensors | `enqueue_input(0, NULL)` returns an error status (no crash) | MUST | header: "Must validate input_tensors structure and all its fields" | pass (5) | draft |
| RT-M-19 | No tensors | a request with `num_tensors` = 0 is rejected | MUST | header: "Must validate input_tensors structure and all its fields"; "Must validate tensor shapes and types match the model's expected inputs" | **FAIL**: SUCCESS, the request is lost (`retrieve` keeps returning `NO_OUTPUT_AVAILABLE`) | draft |
| RT-M-20 | Missing input | a request without one of the model's inputs is rejected (models with at least two inputs; otherwise skipped) | MUST | same as RT-M-19 | **FAIL**: SUCCESS, the request is lost | draft |
| RT-M-21 | Wrong data type | a tensor with another `data_type` than the model's is rejected | MUST | header: "Must validate tensor shapes and types match the model's expected inputs" | **FAIL**: SUCCESS, the request is lost | draft |
| RT-M-22 | Wrong data type: code | the status of RT-M-21 is `TENSOR_TYPE_MISMATCH` | SHOULD | the status exists in the header; the text does not say when to use it | not applicable (accepted) | draft |
| RT-M-23 | Wrong shape | a tensor whose shape differs from the model's (for example `NHWC` for an `NCHW` input) is rejected | MUST | same as RT-M-21 | **FAIL**: SUCCESS, the request is lost | draft |
| RT-M-24 | Wrong shape: code | the status of RT-M-23 is `TENSOR_SHAPE_MISMATCH` | SHOULD | same as RT-M-22 | not applicable (accepted) | draft |
| RT-M-25 | Inconsistent `data_size` | a tensor whose `data_size` is smaller than shape times element size is rejected (otherwise the runtime reads past the buffer) | MUST | header: "Must validate input_tensors structure and all its fields"; the host must keep `data_size` "consistent with its data buffer" | **FAIL**: SUCCESS and an output; valgrind reports 257,702 invalid reads with a half-size `data_size` and 0 with the correct one: the runtime reads past the buffer | draft |
| RT-M-26 | Unknown tensor name | a tensor named like no input of the model is rejected (right type and shape) | PROPOSED | none: the 2.0 text does not say inputs are matched by name | **FAIL**: SUCCESS, the request is lost | draft |
| RT-M-27 | Rejected input is left intact | after any rejection of RT-M-06 and RT-M-18 to RT-M-26 the harness still reads the tensors it filled with a pattern, unchanged, and frees them | MUST | header: "Must not free input_tensors memory if function fails for any reason"; "the caller is responsible for freeing the input_tensors structure" | pass (model id rejection) | draft |
| RT-M-28 | Still usable | after all the rejections above, a valid request is answered | SHOULD | derived: nothing in the text allows a rejected call to break the runtime | pass | draft |

## `RT-M`: concurrency and lifecycle under load

| ID | Check | Pass when | Level | Source | Probed | Status |
| -- | ----- | --------- | ----- | ------ | ------ | ------ |
| RT-M-29 | Concurrent producers | 4 threads enqueue 25 requests each, across 2 model ids, with unique ids, while one thread retrieves: every request is answered exactly once, with its own `id` and `*model_id` | MUST | header: "Must be thread-safe. It must be safe to call this function from multiple threads for the same model or different ones"; id and `*model_id` echo | pass (1 model id) | draft |
| RT-M-30 | Concurrent results are intact | the outputs of RT-M-29 equal the output of the same input run alone (floats within 1e-4 relative) | SHOULD | derived: concurrency must not corrupt data | not probed | draft |
| RT-M-31 | Cleanup with requests in flight | `cleanup` with 20 unretrieved requests returns SUCCESS within 10 s; `init` works again | MUST | header: "Must clean up any pending input that hasn't been processed yet"; "Must clean up any pending output that hasn't been retrieved yet" | pass (7 ms) | draft |
| RT-M-32 | Blocked retrieve wakes up | a thread blocked in `retrieve_output(-1)` returns, with any status, within 5 s of `cleanup` called from another thread | PROPOSED | none: the 2.0 text does not say | not probed | draft |
| RT-M-33 | Flooding | 1000 `enqueue_input` calls without retrieving: every call returns promptly; each accepted request is answered exactly once; a rejected one stays the caller's | PROPOSED | none: the text has no queue capacity or "full" status | not probed | draft |
| RT-M-34 | Info follows the state | `get_info` fields that are present are right: `loaded_models` equals the number of loaded models, `requests_in_flight` equals the requests accepted and not yet retrieved and is 0 when idle | SHOULD | header: "Must update the returned information to reflect the current runtime state on each call" (the fields themselves are optional) | pass (`loaded_models` 2, `requests_in_flight` 0 idle) | draft |
| RT-M-35 | Cycles with a model | 20 x (`init`, `load_models`, 10 requests, `cleanup`): no failure, threads back to baseline, RSS grows by less than 5 MB after the first cycle | SHOULD | derived: "Must free all runtime-allocated memory and resources" | not probed | draft |
| RT-M-36 | Accepted inputs are freed | 500 sequential requests: RSS grows by less than 10 MB | SHOULD | derived: header: "Must free all memory associated with input_tensors after inference completes" | not probed | draft |

## Deliberately not tested in 2.0

- A request the runtime accepted but fails to run (a failed inference): the 2.0 API has no way to report it per request, so a runtime can only
  reject at `enqueue_input` (RT-M-18 to RT-M-26) or lose the request. The behaviour belongs to 2.1.
- A "foreign" model file (valid for another runtime): cannot be tested generically; an ONNX file is valid for ONNX-based runtimes.
- Accuracy against the ONNX reference: end-to-end tier.
- `get_error` pointer lifetime across threads: undefined in 2.0 (valid "until the next runtime function call", from any thread).

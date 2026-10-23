"""RT-M: tests that send requests (--model). Specs: docs/tests/runtime-model.md"""
import json

from ..harness import status_name
from ..registry import Blocked, Fail, NotApplicable, check

SUCCESS, NO_OUTPUT_AVAILABLE = 0, 12
TENSOR_SHAPE_MISMATCH, TENSOR_TYPE_MISMATCH, INVALID_MODEL_ID = 10, 11, 18
MAX_CYCLE_RSS_GROWTH = 5 * 1024 * 1024
MAX_REQUEST_RSS_GROWTH = 10 * 1024 * 1024


_name = status_name
REJECTED_VALID = (TENSOR_TYPE_MISMATCH, TENSOR_SHAPE_MISMATCH, 9)  # TYPE, SHAPE, INVALID_TENSOR: "this is not what I take"


def _scenario(ctx, name: str):
    """Run a scenario that sends requests: needs the model and to know what it takes."""
    ctx.model_io()
    return ctx.run(name)


def _answered(run, step: str) -> dict:
    """A round trip step that must have worked: the request was accepted and its output came back."""
    obs = run.step(step)
    enqueue = obs.get("enqueue", SUCCESS)
    if enqueue in REJECTED_VALID:
        # the suite cannot tell a runtime that takes another type/layout from one that is wrong: it cannot judge, so it does not blame
        raise Blocked(f"{step}: the runtime rejected the request built from the model's own I/O ({_name(enqueue)}): it probably takes another data type "
                      "or layout than the ONNX model. Give what it takes with --input NAME:DTYPE:DIMS, or an oaax-manifest.json next to the model")
    if enqueue != SUCCESS or obs["status"] != SUCCESS:
        got = _name(obs["status"]) if obs["status"] is not None else "not reached"
        raise Fail(f"{step}: the request was not answered (enqueue {_name(enqueue)}, retrieve {got})")
    return obs


def _established(ctx) -> None:
    """The model loads and answers a plain request; otherwise every later finding would be about the model, not the runtime."""
    run = _scenario(ctx, "roundtrip")
    run.step("load")


# ---- loading and ownership -----------------------------------------------------------------------------------


@check("RT-M-02")
def retry_after_failed_load(ctx):
    _established(ctx)
    run = ctx.run("model-load")
    if run.step("load_missing")["status"] == SUCCESS:
        raise Fail("load_models returned SUCCESS for a file that does not exist")
    retry = run.step("load_retry")["status"]
    if retry != SUCCESS:
        raise Fail(f"load_models after a failed load returned {_name(retry)}: the failed call left the runtime unusable")
    _answered(run, "infer_after_retry")


@check("RT-M-03")
def load_from_memory(ctx):
    _established(ctx)
    run = ctx.run("model-load")
    status = run.step("load_memory")["status"]
    if status != SUCCESS:
        raise Fail(f"load_models with model_data and model_size returned {_name(status)}")
    _answered(run, "infer_after_memory_load")


@check("RT-M-04")
def buffers_are_copied(ctx):
    _established(ctx)
    run = ctx.run("model-load")
    problems = []
    for load, infer, what in (("load_memory", "infer_after_memory_load", "model_data"), ("load_path_copy", "infer_after_path_overwritten", "file_path")):
        if run.step(load)["status"] != SUCCESS:
            continue  # RT-M-03 reports that
        try:
            _answered(run, infer)
        except Fail as e:
            problems.append(f"after {what} was overwritten: {e}")
    if problems:
        raise Fail("the runtime kept a pointer instead of a copy: " + "; ".join(problems))


@check("RT-M-05")
def model_id_is_the_index(ctx):
    _established(ctx)
    run = ctx.run("model-load")
    status = run.step("load_two")["status"]
    if status in (8, 14, 16):  # OUT_OF_MEMORY, NOT_IMPLEMENTED, DEVICE_ERROR: the device holds one model at a time
        raise NotApplicable(f"the runtime cannot hold two models at once ({_name(status)})")
    if status != SUCCESS:
        raise Fail(f"loading two models in one call returned {_name(status)}")
    bad = []
    for step, wanted, req in (("infer_model0", 0, 10), ("infer_model1", 1, 11)):
        obs = _answered(run, step)
        if obs["model_id"] != wanted or obs["id"] != req:
            bad.append(f"request {req} to model {wanted} came back as id {obs['id']} of model {obs['model_id']}")
    if bad:
        raise Fail("; ".join(bad))


SKIPPED = {"missing_input": "the model has a single input: there is no input to leave out",
           "wrong_shape": "every dimension of the model's inputs is dynamic (or the input is a scalar): no shape is wrong"}


def _rejected(run, step: str) -> dict:
    obs = run.step(step)
    if obs.get("skipped"):
        raise NotApplicable(SKIPPED.get(step, "not applicable to this model"))
    return obs


def _check_rejected(run, step: str, what: str) -> dict:
    obs = _rejected(run, step)
    if obs["accepted"]:
        fate = "and answered" if obs.get("answered") else "and then silently lost: no output ever came back"
        raise Fail(f"{what} was accepted ({_name(obs['status'])}) {fate}")
    return obs


@check("RT-M-06")
def invalid_model_id(ctx):
    run = _scenario(ctx, "validation")
    for step, what in (("invalid_id_negative", "model_id -1"), ("invalid_id_high", "a model_id past the last model")):
        _check_rejected(run, step, f"enqueue_input with {what}")


@check("RT-M-07")
def invalid_model_id_code(ctx):
    run = _scenario(ctx, "validation")
    for step in ("invalid_id_negative", "invalid_id_high"):
        obs = _rejected(run, step)
        if obs["accepted"]:
            raise NotApplicable("the request was accepted (see RT-M-06)")
        if obs["status"] != INVALID_MODEL_ID:
            raise Fail(f"an invalid model_id returned {_name(obs['status'])}, expected INVALID_MODEL_ID ({INVALID_MODEL_ID})")


# ---- one request, one answer ---------------------------------------------------------------------------------


@check("RT-M-08")
def round_trip(ctx):
    obs = _answered(_scenario(ctx, "roundtrip"), "roundtrip")
    problems = []
    if obs["id"] != 4242:
        problems.append(f"the output carries id {obs['id']}, expected the request's 4242")
    if obs["model_id"] != 0:
        problems.append(f"the output reports model {obs['model_id']}, expected 0")
    if obs["num"] < 1:
        problems.append("the output has no tensors")
    if problems:
        raise Fail("; ".join(problems))


@check("RT-M-09")
def outputs_fully_allocated(ctx):
    if not _answered(_scenario(ctx, "roundtrip"), "roundtrip")["alloc_ok"]:
        raise Fail("an output descriptor has a NULL name, a NULL shape for rank > 0, or NULL data for data_size > 0")


@check("RT-M-10")
def outputs_are_separate_allocations(ctx):
    if not _answered(_scenario(ctx, "roundtrip"), "roundtrip")["distinct"]:
        raise Fail("two name/shape/data pointers of one output are the same: the host would free one buffer twice")


@check("RT-M-11")
def output_data_size(ctx):
    if not _answered(_scenario(ctx, "roundtrip"), "roundtrip")["size_ok"]:
        raise Fail("an output data_size differs from the product of its shape times the element size")


@check("RT-M-12")
def output_names(ctx):
    obs = _answered(_scenario(ctx, "roundtrip"), "roundtrip")
    expected = [t.name for t in ctx.model_io().outputs]
    if not expected:
        raise Blocked("the output names of the model are not known (use an oaax-manifest.json or an .onnx model)")
    got = [bytes.fromhex(t["name"]).decode("utf-8", errors="replace") if t["name"] else None for t in obs["tensors"]]
    if sorted(map(str, got)) != sorted(expected):
        raise Fail(f"output names {got} differ from the model's {expected}")


# ---- timeouts ------------------------------------------------------------------------------------------------


@check("RT-M-13")
def timeout_zero(ctx):
    obs = _scenario(ctx, "timeouts").step("retrieve_0")
    if obs["status"] != NO_OUTPUT_AVAILABLE:
        raise Fail(f"retrieve_output(0) with nothing in flight returned {_name(obs['status'])}, expected NO_OUTPUT_AVAILABLE")
    if obs["ms"] > 100:
        raise Fail(f"retrieve_output(0) took {obs['ms']:.0f} ms instead of returning immediately")


@check("RT-M-14")
def timeout_positive(ctx):
    obs = _scenario(ctx, "timeouts").step("retrieve_200")
    if obs["status"] != NO_OUTPUT_AVAILABLE:
        raise Fail(f"retrieve_output(200) with nothing in flight returned {_name(obs['status'])}, expected NO_OUTPUT_AVAILABLE")
    if obs["ms"] > 200 + 1000:
        raise Fail(f"retrieve_output(200) took {obs['ms']:.0f} ms")


@check("RT-M-15")
def timeout_is_honoured(ctx):
    obs = _scenario(ctx, "timeouts").step("retrieve_200")
    if obs["status"] == NO_OUTPUT_AVAILABLE and obs["ms"] < 0.9 * 200:
        raise Fail(f"retrieve_output(200) returned after {obs['ms']:.0f} ms: it did not wait for the timeout, hosts polling it will spin")


@check("RT-M-16")
def timeout_negative(ctx):
    obs = _scenario(ctx, "timeouts").step("retrieve_negative")
    if obs["status"] != SUCCESS:
        raise Fail(f"retrieve_output(-1) with a request in flight returned {_name(obs['status'])} after {obs['ms']:.0f} ms instead of waiting for the output")


@check("RT-M-17")
def enqueue_does_not_block(ctx):
    obs = _scenario(ctx, "enqueue-latency").step("latency")
    if obs["accepted"] < 8:
        raise Fail(f"only {obs['accepted']} of 8 requests were accepted")
    if obs["sequential_ms"] < 4:
        raise Blocked("the model is too fast to time (8 requests run one after the other took under 4 ms)")
    if obs["enqueue_ms"] * 2 > obs["sequential_ms"]:
        raise Fail(f"enqueueing 8 requests took {obs['enqueue_ms']:.1f} ms, against {obs['sequential_ms']:.1f} ms to run 8 one by one: enqueue_input waits for the inference")


# ---- input validation ----------------------------------------------------------------------------------------


@check("RT-M-18")
def null_tensors(ctx):
    _check_rejected(_scenario(ctx, "validation"), "null_tensors", "enqueue_input with NULL tensors")


@check("RT-M-19")
def no_tensors(ctx):
    _check_rejected(_scenario(ctx, "validation"), "no_tensors", "a request with no tensors")


@check("RT-M-20")
def missing_input(ctx):
    _check_rejected(_scenario(ctx, "validation"), "missing_input", "a request without one of the model's inputs")


@check("RT-M-21")
def wrong_dtype(ctx):
    _check_rejected(_scenario(ctx, "validation"), "wrong_dtype", "a tensor with the wrong data type")


@check("RT-M-22")
def wrong_dtype_code(ctx):
    obs = _rejected(_scenario(ctx, "validation"), "wrong_dtype")
    if obs["accepted"]:
        raise NotApplicable("the request was accepted (see RT-M-21)")
    if obs["status"] != TENSOR_TYPE_MISMATCH:
        raise Fail(f"a wrong data type returned {_name(obs['status'])}, expected TENSOR_TYPE_MISMATCH ({TENSOR_TYPE_MISMATCH})")


@check("RT-M-23")
def wrong_shape(ctx):
    _check_rejected(_scenario(ctx, "validation"), "wrong_shape", "a tensor with the wrong shape")


@check("RT-M-24")
def wrong_shape_code(ctx):
    obs = _rejected(_scenario(ctx, "validation"), "wrong_shape")
    if obs["accepted"]:
        raise NotApplicable("the request was accepted (see RT-M-23)")
    if obs["status"] != TENSOR_SHAPE_MISMATCH:
        raise Fail(f"a wrong shape returned {_name(obs['status'])}, expected TENSOR_SHAPE_MISMATCH ({TENSOR_SHAPE_MISMATCH})")


@check("RT-M-25")
def inconsistent_data_size(ctx):
    _check_rejected(_scenario(ctx, "validation"), "short_data_size", "a tensor whose data_size is half of shape times element size")


@check("RT-M-26")
def unknown_tensor_name(ctx):
    _check_rejected(_scenario(ctx, "validation"), "unknown_name", "a tensor named like no input of the model")


REJECTION_CASES = ("invalid_id_negative", "invalid_id_high", "no_tensors", "missing_input", "wrong_dtype", "wrong_shape", "short_data_size", "unknown_name")


@check("RT-M-27")
def rejected_input_is_intact(ctx):
    run = _scenario(ctx, "validation")
    touched = []
    for step in REJECTION_CASES:
        try:
            obs = run.step(step)
        except (Fail, Blocked):
            continue  # a crash or a missing step belongs to the test of that case
        if not obs.get("skipped") and not obs["accepted"] and not obs["intact"]:
            touched.append(step)
    if touched:
        raise Fail("the runtime changed or freed the input it rejected (cases: " + ", ".join(touched) + ")")
    if run.returncode < 0 and not run.timed_out and not run.crashed_in and run.end is None:
        # the harness frees a rejected input itself: dying there, before it wrote its end marker, means the runtime had already freed it.
        # (A crash after the marker is something else: a static destructor, see RT-L-11.)
        raise Fail(f"the process died ({run.signal_name}) right after a rejected request: the runtime had freed an input that is the caller's")


@check("RT-M-28")
def usable_after_rejections(ctx):
    _answered(_scenario(ctx, "validation"), "still_usable")


# ---- concurrency and lifecycle under load --------------------------------------------------------------------


def _concurrency(ctx) -> dict:
    return _scenario(ctx, "concurrency").step("run")


@check("RT-M-29")
def concurrent_producers(ctx):
    obs = _concurrency(ctx)
    problems = []
    if obs["enqueue_failures"]:
        problems.append(f"{obs['enqueue_failures']} enqueue_input calls failed")
    if obs["received"] != obs["expected"]:
        problems.append(f"{obs['received']} of {obs['expected']} requests were answered" + (" (retrieve stalled)" if obs["stalled"] else ""))
    for key, what in (("duplicates", "answered twice"), ("wrong_model", "came back with the wrong model_id"), ("unknown_id", "came back with an id nobody sent")):
        if obs[key]:
            problems.append(f"{obs[key]} requests {what}")
    if problems:
        raise Fail("; ".join(problems))


@check("RT-M-30")
def concurrent_results_are_intact(ctx):
    obs = _concurrency(ctx)
    if obs["mismatch"]:
        raise Fail(f"{obs['mismatch']} outputs differ from the output of the same input run alone (floats compared within 1e-4 relative)")


@check("RT-M-31")
def cleanup_with_requests_in_flight(ctx):
    run = _scenario(ctx, "cleanup-inflight")
    if run.step("enqueued")["accepted"] == 0:
        raise Blocked("no request was accepted, so none was in flight")
    cleanup = run.step("cleanup")
    if cleanup["status"] != SUCCESS:
        raise Fail(f"cleanup with requests in flight returned {_name(cleanup['status'])}")
    if cleanup["ms"] > 10000:
        raise Fail(f"cleanup with requests in flight took {cleanup['ms']:.0f} ms")
    if run.step("init_again")["status"] != SUCCESS:
        raise Fail("init after such a cleanup failed")
    _answered(run, "infer_again")


@check("RT-M-32")
def blocked_retrieve_wakes_up(ctx):
    obs = _scenario(ctx, "cleanup-wakeup").step("wakeup")
    if not obs["woke"]:
        raise Fail("a thread blocked in retrieve_output(-1) was still blocked 5 s after cleanup ran in another thread")


@check("RT-M-33")
def flooding(ctx):
    obs = _scenario(ctx, "flood").step("flood")
    problems = []
    if obs["requests"] < 10:
        raise Blocked(f"too slow: only {obs['requests']} of {obs['requested']} requests fit the time budget (raise --timeout)")
    if obs["accepted"] + obs["rejected"] != obs["requests"]:
        problems.append("not every enqueue_input call returned")
    if obs["answered"] != obs["accepted"]:
        problems.append(f"{obs['accepted']} requests were accepted but only {obs['answered']} were answered")
    if obs["duplicates"]:
        problems.append(f"{obs['duplicates']} requests were answered twice")
    if obs["max_enqueue_ms"] > 1000:
        problems.append(f"one enqueue_input call took {obs['max_enqueue_ms']:.0f} ms")
    if problems:
        raise Fail("; ".join(problems))


def _info_field(run, step: str, key: str):
    raw = run.raw(step)
    if raw is None:
        raise Fail(f"runtime_get_info() returned NULL ({step})")
    try:
        return json.loads(raw.decode("utf-8")).get(key)
    except (UnicodeDecodeError, ValueError, AttributeError):
        raise Fail(f"runtime_get_info() is not a JSON object ({step})")


@check("RT-M-34")
def info_follows_the_state(ctx):
    run = _scenario(ctx, "info-state")
    expected = (("info_no_model", "loaded_models", 0), ("info_loaded", "loaded_models", 1), ("info_in_flight", "requests_in_flight", 3),
                ("info_idle", "requests_in_flight", 0))
    wrong, seen = [], 0
    for step, key, want in expected:
        got = _info_field(run, step, key)
        if got is None:
            continue  # optional field
        seen += 1
        if got != want:
            wrong.append(f"{key} is {got} at {step.removeprefix('info_')}, expected {want}")
    if wrong:
        raise Fail("; ".join(wrong))
    if not seen:
        raise NotApplicable("the runtime reports neither loaded_models nor requests_in_flight")


@check("RT-M-35")
def cycles_with_a_model(ctx):
    obs = _scenario(ctx, "model-cycles").step("model_cycles")
    if obs["completed"] + obs["failures"] < 3 and obs["cycles"] >= 3:
        raise Blocked(f"too slow to repeat: only {obs['completed']} of {obs['cycles']} cycles fit the time budget (raise --timeout)")
    problems = []
    if obs["failures"]:
        problems.append(obs["first_failure"])
    if obs["threads_after"] > obs["threads_before"]:
        problems.append(f"{obs['threads_after'] - obs['threads_before']} thread(s) left after {obs['cycles']} cycles")
    growth = obs["rss_last"] - obs["rss_first"]
    if obs["rss_first"] >= 0 and growth > MAX_CYCLE_RSS_GROWTH:
        problems.append(f"RSS grew {growth / 1048576:.1f} MB over {obs['cycles']} init/load/cleanup cycles (limit {MAX_CYCLE_RSS_GROWTH // 1048576} MB)")
    if problems:
        raise Fail("; ".join(problems))


@check("RT-M-36")
def accepted_inputs_are_freed(ctx):
    obs = _scenario(ctx, "leak-requests").step("leak")
    if obs["completed"] < 30 and obs["requests"] >= 30 and not obs["failures"]:
        raise Blocked(f"too slow to measure: only {obs['completed']} of {obs['requests']} requests fit the time budget (raise --timeout)")
    if obs["failures"]:
        raise Fail(f"{obs['failures']} of the requests failed")
    growth = obs["rss_after"] - obs["rss_before"]
    if obs["rss_before"] >= 0 and growth > MAX_REQUEST_RSS_GROWTH:
        raise Fail(f"RSS grew {growth / 1048576:.1f} MB over {obs['requests']} requests (limit {MAX_REQUEST_RSS_GROWTH // 1048576} MB)")


# ---- the first model test, which needs only the model --------------------------------------------------------


@check("RT-M-01")
def second_load_models(ctx):
    if not ctx.model:
        raise Blocked("needs a model: pass --model FILE (a model file this runtime can load)")
    status = ctx.run("load-twice").step("load_second")["status"]
    if status == SUCCESS:
        raise Fail("a second load_models call returned SUCCESS; it should return an error and leave the loaded models alone")

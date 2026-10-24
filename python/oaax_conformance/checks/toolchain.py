"""TC / E2E: toolchain images. Specs: docs/tests/toolchain.md"""
import json
import os
from pathlib import Path

from .. import manifest
from ..modelio import NoModelIO, discover
from ..registry import Blocked, Fail, NotApplicable, check
from ..toolchain import ARBITRARY_UID, LOG_NAMES, MANIFEST, follows_naming, make_hostile_zip

BAD_INPUTS = ("empty", "garbage", "truncated", "missing")


def _console_hint(run) -> str:
    lines = [line for line in run.console.splitlines() if line.strip()]
    return f": {lines[-1][:200]}" if lines else " (no console output)"


def _succeeded(run, what: str) -> None:
    if run.timed_out:
        raise Blocked(f"{what}: the toolchain did not finish within the time limit; the spec sets none, raise --timeout if the model is just big")
    if run.returncode != 0:
        raise Fail(f"{what}: exit code {run.returncode}{_console_hint(run)}")


def _valid(ctx):
    return ctx.convert("valid")


# ---- invocation and environment ------------------------------------------------------------------------------


@check("TC-01")
def valid_model(ctx):
    _succeeded(_valid(ctx), "converting a valid model")


@check("TC-02")
def any_non_root_user(ctx):
    _succeeded(ctx.convert("arbitrary", user=ARBITRARY_UID, out_path="/output"), f"converting as uid {ARBITRARY_UID}")


@check("TC-03")
def no_network(ctx):
    _succeeded(ctx.convert("no-network", network=False), "converting with --network none")


@check("TC-04")
def read_only_input(ctx):
    _succeeded(ctx.convert("ro-input", ro_input=True), "converting with the input directory mounted read-only")


@check("TC-05")
def writes_only_to_the_output(ctx):
    changed = _valid(ctx).input_changes()
    if changed:
        raise Fail("the toolchain changed the input directory: " + ", ".join(changed))


@check("TC-06")
def read_only_root_filesystem(ctx):
    _succeeded(ctx.convert("ro-rootfs", read_only_rootfs=True), "converting with --read-only --tmpfs /tmp")


@check("TC-07")
def output_directory_is_created(ctx):
    run = ctx.convert("nested", out_path="/output/a/b/c")
    _succeeded(run, "converting into /output/a/b/c, which does not exist")
    if not run.files():
        raise Fail("exit code 0, but /output/a/b/c was not created or is empty")


@check("TC-08")
def output_directory_may_exist(ctx):
    first = _valid(ctx)
    _succeeded(first, "the first conversion")
    _succeeded(ctx.convert("again", workspace=first.workspace), "a second conversion into the same output directory")


@check("TC-09")
def unusual_file_names(ctx):
    _succeeded(ctx.convert("unicode", filename="my modèl (1).onnx"), "converting 'my modèl (1).onnx'")


@check("TC-10")
def wrong_argument_count(ctx):
    cases = {"args-0": [], "args-1": ["/input/model.onnx"], "args-3": ["/input/model.onnx", "/output/a", "/output/b"]}
    bad = []
    for name, argv in cases.items():
        run = ctx.convert(name, argv=argv, content=ctx.model_bytes())
        if run.timed_out or run.returncode == 0:
            bad.append(f"{len(argv)} argument(s): exit code {run.returncode}")
        elif run.model_files():
            bad.append(f"{len(argv)} argument(s): wrote {run.model_files()[0]}")
    if bad:
        raise Fail("; ".join(bad))


@check("TC-11")
def parallel_runs(ctx):
    runs = ctx.convert_together("parallel", 2)
    failed = [f"#{i}: exit code {r.returncode}{_console_hint(r)}" for i, r in enumerate(runs) if r.returncode != 0 or r.timed_out]
    if failed:
        raise Fail("two conversions at once: " + "; ".join(failed))


# ---- what a successful conversion leaves ---------------------------------------------------------------------


@check("TC-20")
def a_model_file(ctx):
    run = _valid(ctx)
    _succeeded(run, "converting a valid model")
    if not run.model_files():
        raise Fail(f"no non-empty file besides the log in the output directory (found: {', '.join(run.files()) or 'nothing'})")


def _log_problem(run) -> str | None:
    logs = run.log_files()
    if not logs:
        return "no conversion-log.json or logs.json in the output directory"
    for name in logs:
        try:
            json.loads((run.result_dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return f"{name} is not valid JSON: {e}"
    return None


@check("TC-21")
def conversion_log(ctx):
    run = _valid(ctx)
    _succeeded(run, "converting a valid model")
    problem = _log_problem(run)
    if problem:
        raise Fail(problem)


@check("TC-22")
def readable_outputs(ctx):
    run = ctx.convert("arbitrary", user=ARBITRARY_UID, out_path="/output")
    _succeeded(run, f"converting as uid {ARBITRARY_UID}")
    unreadable = []
    for rel, entry in run.output.items():
        mode, need = int(entry["mode"], 8), (0o005 if entry["dir"] else 0o004)  # others: read (files), read + traverse (directories)
        if entry["uid"] != os.getuid() and mode & need != need:  # judged by the mode bits: access() says yes to everything for root
            unreadable.append(f"{rel} ({entry['mode']})")
    if unreadable:
        raise Fail("not readable by the user who started the container: " + ", ".join(unreadable[:5]))


def _manifest_doc(run):
    if MANIFEST not in run.files():
        return None
    try:
        return json.loads((run.result_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Fail(f"{MANIFEST} is not valid JSON: {e}")


@check("TC-23")
def bundle_manifest(ctx):
    run = _valid(ctx)
    _succeeded(run, "converting a valid model")
    if MANIFEST not in run.files():
        raise Fail(f"no {MANIFEST} in the output directory: 2.1 toolchains must write one")


@check("TC-24")
def manifest_is_valid(ctx):
    run = _valid(ctx)
    doc = _manifest_doc(run)
    if doc is None:
        raise NotApplicable("no manifest (see TC-23)")
    problems = manifest.validate(doc, str(run.result_dir))
    if problems:
        raise Fail("; ".join(problems[:4]))


@check("TC-25")
def manifest_matches_the_model(ctx):
    run = _valid(ctx)
    doc = _manifest_doc(run)
    if doc is None:
        raise NotApplicable("no manifest (see TC-23)")
    try:
        original = manifest.derive_io(ctx.onnx_path)
    except ImportError:
        raise Blocked("reading the ONNX model needs the onnx extra: pip install oaax-conformance[onnx]")
    problems = []
    for kind in ("inputs", "outputs"):
        want = {t["name"]: t["onnx"] for t in original[kind]}
        got = {t["name"]: t.get("onnx") for t in doc.get(kind, [])}
        if set(want) != set(got):
            problems.append(f"{kind}: names {sorted(got)} differ from the model's {sorted(want)}")
            continue
        problems += [f"{kind}[{name}]: onnx view {got[name]} differs from {view}" for name, view in want.items() if got[name] and got[name] != view]
    if problems:
        raise Fail("; ".join(problems))


# ---- what a failed conversion does ---------------------------------------------------------------------------


def _bad_input(ctx, kind: str):
    model = ctx.model_bytes()
    if kind == "empty":
        return ctx.convert("empty", filename="empty.onnx", content=b"")
    if kind == "garbage":
        return ctx.convert("garbage", filename="garbage.onnx", content=b"this is not an onnx model\n")
    if kind == "truncated":
        return ctx.convert("truncated", filename="truncated.onnx", content=model[: max(1, len(model) * 3 // 10)])
    if kind == "missing":
        return ctx.convert("missing", argv=["/input/does-not-exist.onnx", "/output/result"])
    return ctx.convert("directory", argv=["/input", "/output/result"])


def _must_fail(run, what: str) -> None:
    if run.timed_out:
        raise Blocked(f"{what}: the toolchain did not finish within the time limit; the spec sets none, raise --timeout if the model is just big")
    if run.returncode == 0:
        files = ", ".join(run.model_files()) or "no model"
        raise Fail(f"{what} was accepted: exit code 0 ({files} written)")


@check("TC-30")
def empty_file(ctx):
    _must_fail(_bad_input(ctx, "empty"), "an empty .onnx file")


@check("TC-31")
def not_an_onnx_file(ctx):
    _must_fail(_bad_input(ctx, "garbage"), "a file that is not ONNX")


@check("TC-32")
def truncated_onnx(ctx):
    _must_fail(_bad_input(ctx, "truncated"), "a truncated ONNX file")


@check("TC-33")
def missing_input(ctx):
    _must_fail(_bad_input(ctx, "missing"), "an input path that does not exist")


@check("TC-34")
def input_is_a_directory(ctx):
    _must_fail(_bad_input(ctx, "directory"), "a directory instead of a model file")


@check("TC-35")
def failure_log(ctx):
    missing = []
    for kind in ("garbage", "truncated", "missing"):
        run = _bad_input(ctx, kind)
        if run.returncode == 0 or run.timed_out:
            continue  # TC-31..33 report that
        problem = _log_problem(run)
        if problem:
            missing.append(f"{kind} input: {problem}")
    if missing:
        raise Fail("a failed conversion must leave a log: " + "; ".join(missing))


def _failed_runs(ctx):
    return [(kind, run) for kind in BAD_INPUTS for run in [_bad_input(ctx, kind)] if run.returncode != 0 and not run.timed_out]


@check("TC-36")
def no_model_after_failure(ctx):
    runs = _failed_runs(ctx)
    if not runs:
        raise NotApplicable("no bad input failed (see TC-30..33)")
    left = [f"{kind}: {run.model_files()[0]}" for kind, run in runs if run.model_files()]
    if left:
        raise Fail("a failed conversion left a model file behind (" + "; ".join(left) + ")")


@check("TC-37")
def failure_is_explained(ctx):
    runs = _failed_runs(ctx)
    if not runs:
        raise NotApplicable("no bad input failed (see TC-30..33)")
    silent = [kind for kind, run in runs if not run.console]
    if silent:
        raise Fail("a failed conversion printed nothing (" + ", ".join(silent) + ")")


@check("TC-38")
def unwritable_output(ctx):
    _must_fail(ctx.convert("unwritable", ro_output=True), "an output directory that cannot be written")


@check("TC-39")
def hostile_archive(ctx):
    run = ctx.convert("zip-slip", filename="evil.zip", content=make_hostile_zip(ctx.model_bytes()), keep_container=True)
    escaped = [k for k in run.output if "escape-" in k and not k.startswith("result/")] + [k for k in run.input_after if "escape-" in k]
    escaped += [f"{p} (inside the container)" for p in run.container_changes if "escape-" in p]
    if escaped:
        raise Fail("the archive wrote outside the directory the toolchain extracts to: " + ", ".join(sorted(escaped)))


# ---- the image -----------------------------------------------------------------------------------------------


@check("TC-50")
def image_name(ctx):
    if not follows_naming(ctx.image):
        raise Fail(f"{ctx.image!r} does not match oaax-<vendor>-toolchain[-<version>]:<tag>")


@check("TC-51")
def entry_point(ctx):
    if not ctx.image_info()["entrypoint"]:
        raise Fail("the image has no ENTRYPOINT: `docker run IMAGE <input> <output>` would try to run <input> as a command")


@check("TC-52")
def default_user(ctx):
    user = ctx.image_info()["user"]
    if user in ("", "root", "0") or user.startswith(("root:", "0:")):
        raise Fail(f"the image runs as root by default (USER {user or 'not set'})")


# ---- end to end ----------------------------------------------------------------------------------------------


def _converted(ctx):
    """(runtime Context for the converted model, path of that model) or Blocked."""
    run = _valid(ctx)
    if run.returncode != 0 or run.timed_out:
        raise Blocked("the conversion of TC-01 failed")
    if not ctx.runtime_lib:
        raise Blocked("needs --runtime LIB: the runtime that loads what this toolchain produces")
    doc = _manifest_doc(run)
    names = [doc["model_file"]] if doc and doc.get("model_file") in run.files() else run.model_files()
    if len(names) != 1:
        raise Blocked(f"cannot tell which output file is the model ({', '.join(names) or 'none'}); a 2.1 oaax-manifest.json names it")
    path = str(run.result_dir / names[0])
    return ctx.runtime_context(path), path


@check("E2E-01")
def runtime_loads_it(ctx):
    rctx, _ = _converted(ctx)
    run = rctx.run("load-twice")
    if run.end == "model":
        raise Fail("the runtime could not load the converted model" + run._why_missing("load_first").split("--model", 1)[-1].replace(")", "", 1).rstrip())
    run.step("load_first")


def _runtime_run(ctx, scenario: str):
    """The converted model in the runtime, after the checks every end-to-end test needs: the model loads, and what it takes is known."""
    from .runtime_model import _answered

    rctx, path = _converted(ctx)
    rctx.model_io()  # Blocked: the converted model's inputs cannot be read (no manifest, not ONNX): no request can be built
    run = rctx.run(scenario)
    if run.end == "model":
        raise Blocked("the converted model does not load (see E2E-01)")
    return rctx, path, run, _answered


@check("E2E-02")
def runtime_answers(ctx):
    _, _, run, answered = _runtime_run(ctx, "roundtrip")
    answered(run, "roundtrip")


def _io_views(model_path: str, doc: dict | None):
    """(inputs, outputs) of the converted model as {name: (dtype, shape)}: from its manifest's onnx views, or from the ONNX file itself."""
    if doc:
        if not all("onnx" in t for k in ("inputs", "outputs") for t in doc[k]):
            return None  # outputs without an onnx view: the manifest says the model is not equivalent (checked by the caller)
        return [{t["name"]: (t["onnx"]["dtype"], tuple(t["onnx"]["shape"])) for t in doc[k]} for k in ("inputs", "outputs")]
    if Path(model_path).suffix == ".onnx":
        io = manifest.derive_io(model_path)
        return [{t["name"]: (t["onnx"]["dtype"], tuple(t["onnx"]["shape"])) for t in io[k]} for k in ("inputs", "outputs")]
    return None


@check("E2E-03")
def io_is_retained(ctx):
    _, path = _converted(ctx)
    doc = _manifest_doc(_valid(ctx))
    if doc and doc.get("equivalent") is False:
        raise Fail("the manifest declares equivalent: false: the runtime does not return the original model's outputs "
                   f"(it returns {', '.join(t['name'] for t in doc['outputs'])}); the spec requires the names and shapes to be retained")
    try:
        converted = _io_views(path, doc)
        original = _io_views(ctx.onnx_path, None)
    except ImportError:
        raise Blocked("reading ONNX models needs the onnx extra: pip install oaax-conformance[onnx]")
    if converted is None:
        raise NotApplicable("the converted model is not ONNX and has no manifest: its I/O cannot be read")
    problems = []
    for kind, got, want in zip(("inputs", "outputs"), converted, original):
        if set(got) != set(want):
            problems.append(f"{kind}: names {sorted(got)} differ from the original's {sorted(want)}")
        else:
            problems += [f"{kind}[{n}]: {got[n]} differs from the original's {want[n]}" for n in want if got[n] != want[n]]
    if problems:
        raise Fail("; ".join(problems))


@check("E2E-04")
def same_results(ctx):
    try:
        import numpy as np
        import onnxruntime
    except ImportError:
        raise Blocked("needs the e2e extra: pip install oaax-conformance[e2e]")
    from ..reference import deterministic_input

    doc = _manifest_doc(_valid(ctx))
    if doc and doc.get("equivalent") is False:
        raise NotApplicable("the manifest declares equivalent: false: the outputs are not the original's, so there is nothing to compare")
    rctx, path, run, answered = _runtime_run(ctx, "roundtrip")
    obs = answered(run, "roundtrip")
    try:
        io = discover(ctx.onnx_path, [])
    except NoModelIO as e:
        raise Blocked(str(e))
    session = onnxruntime.InferenceSession(ctx.onnx_path, providers=["CPUExecutionProvider"])
    feeds = {t.name: deterministic_input(t.dtype, t.concrete_shape()) for t in io.inputs}
    expected = dict(zip([o.name for o in session.get_outputs()], session.run(None, feeds)))
    got = {}
    for i, tensor in enumerate(obs["tensors"]):
        name = bytes.fromhex(tensor["name"]).decode("utf-8", errors="replace")
        raw = Path(rctx.dump_dir, f"roundtrip_{i}.bin").read_bytes()
        dtype = {1: np.float32, 11: np.float64, 7: np.int64, 6: np.int32, 2: np.uint8, 3: np.int8, 9: np.bool_}.get(tensor["dtype"])
        if dtype is None:
            raise Blocked(f"output {name!r} has a data type the comparison does not read yet ({tensor['dtype']})")
        got[name] = np.frombuffer(raw, dtype=dtype).reshape(tensor["shape"])
    if set(got) != set(expected):
        raise Fail(f"the runtime returned {sorted(got)}, the original model has {sorted(expected)}: the outputs are not the same tensors")
    bad = []
    for name, want in expected.items():
        if want.shape != got[name].shape:
            bad.append(f"{name}: shape {tuple(got[name].shape)} instead of {tuple(want.shape)}")
        elif not np.allclose(got[name], want, rtol=1e-4, atol=1e-5, equal_nan=True):
            bad.append(f"{name}: values differ (max abs error {float(np.max(np.abs(got[name].astype(np.float64) - want.astype(np.float64)))):.3g})")
    if bad:
        raise Fail("; ".join(bad))

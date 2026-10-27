"""The toolchain tier tests itself: the clean mock toolchain passes everything, every check fails on its own fault.
Needs Docker (the mock toolchain is an image); skipped without it."""
import shutil
import subprocess
import time

import pytest

from oaax_conformance.registry import CHECKS
from oaax_conformance.suite import run_suite
from oaax_conformance.toolchain import ToolchainContext, default_model

CLEAN = "oaax-mock-toolchain:1.0"

# (OAAX_MOCK_FAULT of the toolchain, the check that must catch it, image)
CASES = [
    ("fail_always", "TC-01", CLEAN),
    ("needs_root", "TC-02", CLEAN),
    ("needs_network", "TC-03", CLEAN),
    ("writes_input_dir", "TC-04", CLEAN),
    ("writes_input_dir", "TC-05", CLEAN),
    ("rw_root", "TC-06", CLEAN),
    ("no_mkdir", "TC-07", CLEAN),
    ("not_idempotent", "TC-08", CLEAN),
    ("ascii_only", "TC-09", CLEAN),
    ("bad_args_ok", "TC-10", CLEAN),
    ("lock", "TC-11", CLEAN),
    ("no_model", "TC-20", CLEAN),
    ("no_log", "TC-21", CLEAN),
    ("bad_json_log", "TC-21", CLEAN),
    ("no_perm", "TC-22", CLEAN),
    ("no_manifest", "TC-23", CLEAN),
    ("manifest_invalid", "TC-24", CLEAN),
    ("manifest_wrong_names", "TC-25", CLEAN),
    ("accept_empty", "TC-30", CLEAN),
    ("accept_garbage", "TC-31", CLEAN),
    ("accept_garbage", "TC-32", CLEAN),
    ("accept_missing", "TC-33", CLEAN),
    ("accept_dir", "TC-34", CLEAN),
    ("no_log_on_failure", "TC-35", CLEAN),
    ("leave_model_on_failure", "TC-36", CLEAN),
    ("silent_failure", "TC-37", CLEAN),
    ("unwritable_ok", "TC-38", CLEAN),
    ("zip_slip", "TC-39", CLEAN),
    (None, "TC-50", "mocktoolchain:latest"),
    (None, "TC-51", "oaax-mocknoentry-toolchain:1.0"),
    (None, "TC-52", "oaax-mockroot-toolchain:1.0"),
    ("corrupt_model", "E2E-01", CLEAN),
    ("drop_all", "E2E-02", CLEAN),  # a runtime fault, see _runtime_fault
    ("manifest_wrong_names", "E2E-03", CLEAN),
    ("wrong_output", "E2E-04", CLEAN),  # a runtime fault
]
RUNTIME_FAULTS = {"drop_all", "wrong_output"}  # these faults are set in the runtime's environment, the others in the toolchain's


def _context(build_dir, mock_io_onnx, image, fault, tmp_path, with_runtime=True):
    docker_args = []
    if fault and fault not in RUNTIME_FAULTS:
        docker_args += ["-e", f"OAAX_MOCK_FAULT={fault}"]
    if fault == "lock":  # two containers meet at a directory they share
        shared = tmp_path / "lock"
        shared.mkdir(mode=0o777)
        shared.chmod(0o777)
        docker_args += ["-v", f"{shared}:/lock"]
    return ToolchainContext(image, mock_io_onnx if with_runtime else default_model(), docker_args, timeout=120,
                            runtime_lib=str(build_dir / "libmock_runtime.so") if with_runtime else None, harness=str(build_dir / "oaax-harness"),
                            init_config=["vendor_key=1"], runtime_timeout=60, warmup_ms=3000)


def _run(ctx, only=None):
    try:
        return {r.id: r for r in run_suite(ctx, only, prefixes=("TC-", "E2E-")).results}
    finally:
        ctx.close()


def test_clean_toolchain_passes_everything(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, None, tmp_path))
    assert {i: r.message for i, r in results.items() if r.outcome != "pass"} == {}


@pytest.mark.parametrize("fault,test_id,image", CASES, ids=[f"{c[1]}-{c[0] or 'image'}" for c in CASES])
def test_toolchain_fault_is_caught(build_dir, mock_io_onnx, toolchain_images, tmp_path, monkeypatch, fault, test_id, image):
    if fault in RUNTIME_FAULTS:
        monkeypatch.setenv("OAAX_MOCK_FAULT", fault)
    results = _run(_context(build_dir, mock_io_onnx, image, fault, tmp_path), [test_id])
    target = results[test_id]
    assert target.outcome == ("fail" if target.level == "MUST" else "warn"), target


def test_every_toolchain_check_has_a_fault_case():
    assert {c[1] for c in CASES} == {i for i in CHECKS if i.startswith(("TC-", "E2E-"))}


def test_e2e_is_blocked_without_a_runtime(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, None, tmp_path, with_runtime=False), ["E2E-01", "E2E-04"])
    assert {r.outcome for r in results.values()} == {"blocked"} and "--runtime" in results["E2E-01"].message


def test_a_toolchain_that_fails_blocks_the_end_to_end_tests(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "fail_always", tmp_path), ["E2E-01"])
    assert results["E2E-01"].outcome == "blocked"


def test_the_suite_removes_its_temporary_directories(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    from pathlib import Path

    ctx = _context(build_dir, mock_io_onnx, CLEAN, "zip_slip", tmp_path)
    run_suite(ctx, ["TC-39", "E2E-02"], prefixes=("TC-", "E2E-"))
    roots = [ws.root for ws in ctx.workspaces] + ([Path(ctx._dump)] if ctx._dump else [])
    assert roots
    ctx.close()
    assert not [r for r in roots if r.exists()] and ctx.leftovers == []


# ---- regression tests for the adversary findings (see also test_regressions.py) -------------------------------------------


def test_a_manifest_that_says_equivalent_false_is_explained_not_a_crash(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """A9: raw heads declared honestly: E2E-03 fails with the reason, E2E-04 has nothing to compare, nothing crashes."""
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "raw_heads", tmp_path), ["E2E-02", "E2E-03", "E2E-04", "TC-24", "TC-25"])
    assert results["E2E-03"].outcome == "fail" and "equivalent: false" in results["E2E-03"].message
    assert results["E2E-04"].outcome == "na"
    assert results["E2E-02"].outcome == "pass" and results["TC-24"].outcome == "pass"
    assert [i for i, r in results.items() if r.outcome == "error"] == []


def test_a_vendor_model_format_without_a_manifest_blocks_the_tests_that_need_its_inputs(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """A9: a .hef-like file and no manifest: the model loads, but nobody can say what it takes."""
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "vendor_format", tmp_path), ["E2E-01", "E2E-02", "E2E-03", "E2E-04"])
    assert results["E2E-01"].outcome == "pass"
    assert results["E2E-02"].outcome == "blocked" and "oaax-manifest.json" in results["E2E-02"].message or "take" in results["E2E-02"].message
    assert results["E2E-03"].outcome == "na" and results["E2E-04"].outcome == "blocked"


def test_subdirectories_created_by_the_toolchains_uid_are_removed(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """A13: the container runs as uid 12345 and creates sub/deep: the suite opens it up from a throwaway container and removes it."""
    ctx = _context(build_dir, mock_io_onnx, CLEAN, "subdirs", tmp_path)
    run_suite(ctx, ["TC-02"], prefixes=("TC-",))
    roots = [ws.root for ws in ctx.workspaces]
    assert any((root / "out" / "sub").exists() for root in roots)  # the toolchain did create them
    ctx.close()
    assert ctx.leftovers == [] and not any(root.exists() for root in roots)


def test_an_archive_that_escapes_inside_the_container_is_caught(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """A14: the members land in /tmp of the container, where no mount can see them: only `docker diff` does."""
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "zipslip_tmp", tmp_path), ["TC-39"])
    assert results["TC-39"].outcome == "warn" and "inside the container" in results["TC-39"].message


def test_docker_environment_values_never_reach_the_report(build_dir, toolchain_images, tmp_path):
    from oaax_conformance import cli

    out = tmp_path / "out"
    code = cli.main(["toolchain", CLEAN, "--docker-arg=-e", "--docker-arg=VENDOR_LICENSE=SECRET-LIC-42", "--only", "TC-01", "--only", "TC-50", "--quiet", "--out", str(out)])
    assert code == 0
    for name in ("report.json", "report.html", "junit.xml"):
        assert "SECRET-LIC-42" not in (out / name).read_text(), name
    assert "VENDOR_LICENSE=***" in (out / "report.json").read_text()


@pytest.mark.parametrize("only", ["TC-01", "TC-11"])  # TC-11 runs two containers from worker threads: D2
def test_ctrl_c_removes_the_running_container(build_dir, toolchain_images, tmp_path, only):
    """A12: an interrupted run must not leave its toolchain container running."""
    import os
    import signal
    import sys

    import uuid

    label = f"oaax-test={uuid.uuid4().hex}"  # parallel tests run containers too: find only this one
    proc = subprocess.Popen([sys.executable, "-m", "oaax_conformance.cli", "toolchain", CLEAN, "--docker-arg=-e", "--docker-arg=OAAX_MOCK_FAULT=hang",
                             f"--docker-arg=--label={label}", "--only", only, "--quiet", "--timeout", "120"], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    listing = lambda: subprocess.run(["docker", "ps", "-q", "--filter", f"label={label}"], capture_output=True, text=True).stdout.split()  # noqa: E731
    deadline = time.monotonic() + 40
    while not listing() and time.monotonic() < deadline:
        time.sleep(0.5)
    assert listing(), "the container never started"
    os.killpg(proc.pid, signal.SIGINT)
    assert proc.wait(timeout=60) == 130
    assert listing() == []
    assert b"Traceback" not in proc.stderr.read()


def test_b2_a_docker_failure_is_blocked_never_a_verdict(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """B2: a bad --docker-arg makes `docker run` exit 125; no negative test may pass and no MUST may fail because of it."""
    ctx = _context(build_dir, mock_io_onnx, CLEAN, None, tmp_path)
    ctx.docker_args = ["--bogus-flag"]
    results = _run(ctx, ["TC-01", "TC-10", "TC-11", "TC-30", "TC-35", "TC-38", "TC-39"])
    assert {r.outcome for r in results.values()} == {"blocked"}  # TC-11 runs in threads: no lost exception, no ERROR
    assert "docker could not start" in results["TC-10"].message


def test_b4_symlinks_are_listed_never_followed_nor_read(tmp_path):
    """B4: a link to a host file is not a model or a log, and is not hashed (it was read on the host)."""
    from oaax_conformance.toolchain import tree

    (tmp_path / "real.bin").write_bytes(b"x" * (3 << 20))
    (tmp_path / "outside.txt").write_text("host secret")
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.bin").symlink_to(tmp_path / "real.bin")
    (out / "logs.json").symlink_to(tmp_path / "outside.txt")
    entries = tree(out)
    assert all("link" in e and "sha256" not in e for e in entries.values())
    assert len(tree(tmp_path)["real.bin"]["sha256"]) == 64
    from types import SimpleNamespace

    from oaax_conformance.toolchain import Conversion

    run = SimpleNamespace(out_path="/output", output={**entries, "ok.onnx": {"dir": False, "size": 1, "reg": True}})
    assert list(Conversion.files(run)) == ["ok.onnx"]  # links are neither models nor logs


def test_b8_a_slow_toolchain_is_blocked_with_a_hint_not_failed(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    ctx = _context(build_dir, mock_io_onnx, CLEAN, "hang", tmp_path)
    ctx.timeout = 5
    results = _run(ctx, ["TC-01", "TC-30"])
    assert {r.outcome for r in results.values()} == {"blocked"} and "--timeout" in results["TC-01"].message


def test_d3_a_stalled_conversion_blocks_every_check_that_needed_it(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """D3: a run killed at --timeout used to make TC-35 (a MUST) and TC-39 PASS and TC-11 blame 'exit code -9'; nothing it left behind proves anything."""
    ctx = _context(build_dir, mock_io_onnx, CLEAN, "hang", tmp_path)
    ctx.timeout = 5
    results = _run(ctx, ["TC-11", "TC-35", "TC-39"])
    assert {i: r.outcome for i, r in results.items()} == {"TC-11": "blocked", "TC-35": "blocked", "TC-39": "blocked"}
    assert "--timeout" in results["TC-11"].message


def test_d3_tc35_does_not_pass_when_no_bad_input_failed(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "exit0_on_error", tmp_path), ["TC-35"])
    assert results["TC-35"].outcome == "na", results["TC-35"]


def test_no_toolchain_check_passes_when_every_conversion_stalls(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """The D3 class as a table: against an image that stalls past --timeout, no check that runs the toolchain may PASS, FAIL or WARN (only the static
    image checks TC-50..52 look at the image instead; TC-10 is observed for real: the mock rejects a wrong argument count before it reaches the stalling step)."""
    ctx = _context(build_dir, mock_io_onnx, CLEAN, "hang", tmp_path)
    ctx.timeout = 2
    results = _run(ctx)
    judged = {i: r.outcome for i, r in results.items() if not i.startswith(("TC-50", "TC-51", "TC-52")) and i != "TC-10" and r.outcome in ("pass", "fail", "warn", "error")}
    assert judged == {}


def test_disk_use_does_not_grow_with_the_number_of_conversions(build_dir, mock_io_onnx, toolchain_images, tmp_path, monkeypatch):
    """A7: every conversion had its own workspace until the end of the run (13x the model's size in the adversary's measurement); big files go as soon as
    the conversion is done, so the peak stays near one model (the 'valid' one, which E2E loads) plus the conversions in flight."""
    ctx = _context(build_dir, mock_io_onnx, CLEAN, "big_output", tmp_path, with_runtime=False)
    peaks = []
    real = ToolchainContext.convert

    def measured(self, *args, **kwargs):
        try:
            return real(self, *args, **kwargs)
        finally:
            peaks.append(sum(f.stat().st_size for ws in self.workspaces for f in ws.root.rglob("*") if f.is_file() and not f.is_symlink()))

    monkeypatch.setattr(ToolchainContext, "convert", measured)
    results = _run(ctx, [i for i in CHECKS if i.startswith("TC-") and i not in ("TC-50", "TC-51", "TC-52")])
    assert len(peaks) >= 12 and {r.outcome for r in results.values()} <= {"pass", "warn", "na", "blocked"}
    assert max(peaks) < 4 * (8 << 20), f"peak {max(peaks) >> 20} MiB over {len(peaks)} conversions"


# ---- round 5, A12: a toolchain that writes more than the model --------------------------------------------------------------------------

def test_vendor_metadata_next_to_the_model_does_not_block_the_end_to_end_tests(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """A12: a 2.0 toolchain has no manifest, and a second non-empty file (labels, calibration cache) made E2E-01 BLOCKED with no way out: the suite now
    tries each candidate and the one the runtime loads is the model (the labels file sorts first and the runtime refuses it)."""
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "second_file,no_manifest", tmp_path), ["E2E-01"])
    assert results["E2E-01"].outcome == "pass"
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "second_file", tmp_path), ["E2E-01", "E2E-02", "E2E-04"])  # with a manifest, as before
    assert {i: r.outcome for i, r in results.items()} == {"E2E-01": "pass", "E2E-02": "pass", "E2E-04": "pass"}


def test_no_output_file_loads_is_a_failure_that_names_the_candidates_and_the_way_out(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "corrupt_model,second_file,no_manifest", tmp_path), ["E2E-01", "E2E-02"])
    assert results["E2E-01"].outcome == "fail" and "a-labels.json, model.bin" in results["E2E-01"].message and "--model-file" in results["E2E-01"].message
    assert results["E2E-02"].outcome == "blocked"


def test_model_file_names_the_model_explicitly(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    ctx = _context(build_dir, mock_io_onnx, CLEAN, "second_file,no_manifest", tmp_path)
    ctx.model_file = "model.bin"
    assert _run(ctx, ["E2E-01"])["E2E-01"].outcome == "pass" and ctx.candidates == ["model.bin"]
    ctx = _context(build_dir, mock_io_onnx, CLEAN, "second_file,no_manifest", tmp_path)
    ctx.model_file = "nope.bin"
    blocked = _run(ctx, ["E2E-01"])["E2E-01"]
    assert blocked.outcome == "blocked" and "--model-file nope.bin" in blocked.message and "model.bin" in blocked.message


@pytest.mark.parametrize("name", ["../x", "/etc/passwd"])
def test_model_file_must_stay_inside_the_output_directory(build_dir, name):
    from oaax_conformance import cli

    assert cli.main(["toolchain", "x", "--model-file", name]) == 3


# ---- round 5, A13: files the toolchain writes are not trusted -----------------------------------------------------------------------------

def test_a_named_pipe_where_the_log_should_be_does_not_hang_the_suite(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """A13a: logs.json as a FIFO made read_text() block forever (only SIGTERM ended the run, no report); only regular files are listed and opened now."""
    import signal

    def stuck(*_):
        raise AssertionError("the suite blocked reading a named pipe")

    old = signal.signal(signal.SIGALRM, stuck)
    signal.alarm(60)
    try:
        result = _run(_context(build_dir, mock_io_onnx, CLEAN, "fifo_log", tmp_path, with_runtime=False), ["TC-21"])["TC-21"]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert result.outcome == "fail" and "no conversion-log.json or logs.json" in result.message


def test_a_huge_log_is_reported_not_read_into_memory(build_dir, mock_io_onnx, toolchain_images, tmp_path, monkeypatch):
    """A13b: a 3 GB logs.json made json.loads(read_text()) use 3 GB and end as 'TC-21 ERROR: MemoryError, a bug of the suite' (here: 2 MB over a 1 MB limit)."""
    from oaax_conformance.checks import toolchain as checks

    monkeypatch.setattr(checks, "MAX_JSON_BYTES", 1 << 20)
    result = _run(_context(build_dir, mock_io_onnx, CLEAN, "huge_log", tmp_path, with_runtime=False), ["TC-21"])["TC-21"]
    assert result.outcome == "warn" and "2 MB" in result.message and "not checked" in result.message


# ---- round 5, A8: one hung baseline conversion costs one timeout ------------------------------------------------------------------------------

def test_a_toolchain_that_hangs_on_a_valid_model_costs_one_timeout_not_one_per_conversion(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """A8: every one of the ~15 conversions was tried and timed out on its own (hours with the default 900 s); when the baseline hangs they are blocked at once."""
    ctx = _context(build_dir, mock_io_onnx, CLEAN, "hang", tmp_path, with_runtime=False)
    ctx.timeout = 4
    started = time.monotonic()
    results = _run(ctx, ["TC-01", "TC-02", "TC-03", "TC-05", "TC-30", "TC-31"])
    assert time.monotonic() - started < 12  # one 4 s timeout, not six
    assert {r.outcome for r in results.values()} == {"blocked"}
    assert "the other conversions are not tried" in results["TC-02"].message and "--timeout" in results["TC-30"].message


# ---- round 6 --------------------------------------------------------------------------------------------------------------------------

def test_a_toolchain_that_cleans_its_output_directory_does_not_break_the_end_to_end_tests(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """C1: TC-08 converted a second time into the baseline's own output directory; a toolchain that wipes and renames what it finds there made
    E2E-01 FAIL (MUST) on a model that had loaded fine. TC-08 now works on a copy."""
    ids = ["TC-01", "TC-08", "E2E-01", "E2E-02", "E2E-03", "E2E-04"]
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "wipe_second", tmp_path), ids)
    assert {i: (r.outcome, r.message) for i, r in results.items() if r.outcome != "pass"} == {}


def test_the_copy_of_an_output_directory_keeps_files_and_links_and_never_opens_a_fifo(tmp_path):
    import os

    from oaax_conformance.toolchain import Workspace

    first = Workspace.create()
    try:
        (first.out_dir / "result").mkdir()
        (first.out_dir / "result" / "model.bin").write_bytes(b"MODEL")
        os.symlink("model.bin", first.out_dir / "result" / "link")
        os.mkfifo(first.out_dir / "result" / "pipe")  # copying this by opening it would block forever
        copy = Workspace.with_output_of(first)
        try:
            assert (copy.out_dir / "result" / "model.bin").read_bytes() == b"MODEL"
            assert os.readlink(copy.out_dir / "result" / "link") == "model.bin"
            assert not (copy.out_dir / "result" / "pipe").exists()
        finally:
            shutil.rmtree(copy.root)
    finally:
        shutil.rmtree(first.root)


@pytest.mark.parametrize("fault", ["fifo_manifest", "link_manifest", "huge_manifest"])
def test_a_manifest_that_is_not_a_small_regular_file_never_hangs_the_end_to_end_tests(build_dir, mock_io_onnx, toolchain_images, tmp_path, fault):
    """C8: a named pipe called oaax-manifest.json hung the whole suite in E2E (modelio read it for every scenario)."""
    import signal

    def alarm(*_):
        raise TimeoutError("the suite hung")

    old = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(150)
    try:
        results = _run(_context(build_dir, mock_io_onnx, CLEAN, fault, tmp_path), ["TC-01", "E2E-01", "E2E-02"])
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert results["TC-01"].outcome == "pass"
    assert results["E2E-01"].outcome in ("pass", "blocked") and results["E2E-02"].outcome == "blocked"
    assert "manifest" in results["E2E-02"].message


def test_the_end_to_end_tests_prefer_model_dot_star_and_say_which_file_they_judged(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """C3: with two loadable output files and no manifest E2E silently tested the alphabetically first one and no report named it."""
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "two_models,no_manifest", tmp_path), ["E2E-01"])
    r = results["E2E-01"]
    assert r.outcome == "pass", r.message
    assert r.note.startswith("the end-to-end tests judge the output file model.bin (the first of 2 output files that the runtime loads (model.bin, a-copy.bin)")
    assert "--model-file" in r.note


def test_a_manifest_names_the_model_and_the_note_says_so(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "two_models", tmp_path), ["E2E-01", "E2E-02", "E2E-03", "E2E-04"])
    for test_id, r in results.items():
        assert r.outcome == "pass", (test_id, r.message)
        assert r.note == "the end-to-end tests judge the output file model.bin (named by the manifest)", (test_id, r.note)  # the same file in all four


def test_an_explicit_model_file_is_the_one_judged_and_the_note_says_so(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    from oaax_conformance.toolchain import ToolchainContext

    ctx = _context(build_dir, mock_io_onnx, CLEAN, "two_models", tmp_path)
    ctx.model_file = "a-copy.bin"
    results = _run(ctx, ["E2E-01"])
    assert results["E2E-01"].outcome == "pass" and "a-copy.bin (chosen with --model-file)" in results["E2E-01"].note


def test_every_dump_directory_of_every_candidate_is_removed_at_close(build_dir, mock_io_onnx, tmp_path):
    """C6: only the last candidate's dump directory was removed; the others stayed in /tmp."""
    import os

    ctx = _context(build_dir, mock_io_onnx, CLEAN, None, tmp_path)
    for name in ("a.bin", "b.bin", "c.bin"):
        ctx.runtime_context(str(tmp_path / name))
    dumps = list(ctx._dumps)
    assert len(dumps) == 3 and all(os.path.isdir(d) for d in dumps)
    ctx.close()
    assert not [d for d in dumps if os.path.exists(d)]


def test_a_secret_cut_by_the_tail_never_shows_half_of_it(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """C7: the last 3000 characters of the console were cut BEFORE the secrets were removed, so a secret straddling the cut left its tail in report.json."""
    token = "s3cretvalue123"
    ctx = _context(build_dir, mock_io_onnx, CLEAN, "echo_token", tmp_path)
    ctx.docker_args += ["-e", f"TOKEN={token}"]
    _run(ctx, ["TC-01"])
    obs = ctx._runs["valid"].observations()[0]  # the command line is masked later, in the report; what the toolchain printed must already be clean
    shown = obs.get("stdout_tail", "") + obs.get("stderr_tail", "")
    assert shown.count("B") > 2000 and token not in shown and len(shown) <= 3000
    assert all(token[-n:] not in shown for n in range(3, len(token) + 1)), shown[:60]  # not even its last 3 characters


def test_a_runtime_secret_cut_by_the_tail_never_shows_half_of_it(tmp_path):
    import os
    import stat

    from oaax_conformance.harness import TAIL, run_scenario

    token = "s3cretvalue123"
    script = tmp_path / "fake-harness"
    script.write_text(f"#!/usr/bin/env python3\nprint('A' * 5000 + '{token}' + 'B' * {TAIL - 5})\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    run = run_scenario(str(script), "static", "libx.so", [f"key={token}"], timeout=10)
    assert token not in run.stdout_tail and all(token[-n:] not in run.stdout_tail for n in range(3, len(token) + 1)), run.stdout_tail[:60]
    assert len(run.stdout_tail) <= TAIL

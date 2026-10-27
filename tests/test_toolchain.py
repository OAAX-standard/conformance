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
    roots = [ws.root for ws in ctx.workspaces] + [Path(d) for d in ctx._dumps]
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
    assert results["E2E-01"].outcome == "fail" and "model.bin, a-labels.json" in results["E2E-01"].message and "--model-file" in results["E2E-01"].message
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


@pytest.mark.parametrize("fault", ["fifo_manifest", "link_manifest", "huge_manifest", "nested_manifest"])
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


def test_the_tc08_copy_never_holds_the_big_model_a_second_time(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """C10(4): the workspace of TC-08's second conversion was not pruned like the others, so a multi-GB model existed up to three times."""
    from oaax_conformance.toolchain import KEEP_BYTES

    ctx = _context(build_dir, mock_io_onnx, CLEAN, "big_output", tmp_path)
    try:
        results = {r.id: r for r in run_suite(ctx, ["TC-01", "TC-08"], prefixes=("TC-", "E2E-")).results}
        assert results["TC-08"].outcome == "pass" and len(ctx.workspaces) >= 2
        baseline, others = ctx.workspaces[0], ctx.workspaces[1:]
        assert any(p.stat().st_size > KEEP_BYTES for p in baseline.out_dir.rglob("*") if p.is_file())  # E2E still loads the baseline's model
        assert [p for ws in others for p in ws.out_dir.rglob("*") if p.is_file() and p.stat().st_size > KEEP_BYTES] == []
    finally:
        ctx.close()


def test_the_copy_of_an_output_directory_does_not_copy_big_files(tmp_path):
    from oaax_conformance.toolchain import KEEP_BYTES, Workspace

    first = Workspace.create()
    try:
        with open(first.out_dir / "big.bin", "wb") as f:
            f.write(b"x" * 1000)
            f.truncate(4 * KEEP_BYTES)  # sparse stand-in for a compiled model
        (first.out_dir / "small.json").write_text("{}")
        copy = Workspace.with_output_of(first)
        try:
            placeholder = copy.out_dir / "big.bin"
            assert placeholder.stat().st_size == 4 * KEEP_BYTES and placeholder.stat().st_blocks * 512 < KEEP_BYTES  # same name and size, no disk
            assert (copy.out_dir / "small.json").read_text() == "{}"
        finally:
            shutil.rmtree(copy.root)
    finally:
        shutil.rmtree(first.root)


# ---- round 7 ---------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fault", ["manifest_list", "manifest_string", "manifest_number", "manifest_no_inputs"])
def test_a_manifest_that_is_not_a_usable_object_never_crashes_a_check(build_dir, mock_io_onnx, toolchain_images, tmp_path, fault):
    """D1: valid JSON that is no object (`[1]`, `"x"`, `7`) or lacks `inputs` made TC-25 and E2E-01..04 ERROR (exit 4, "a bug of the suite") on what a toolchain wrote."""
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, fault, tmp_path), ["TC-24", "TC-25", "E2E-01", "E2E-02", "E2E-03", "E2E-04"])
    assert [(i, r.message) for i, r in results.items() if r.outcome == "error"] == []
    assert results["TC-24"].outcome in ("warn", "fail") and results["TC-24"].message  # said once, where it belongs
    assert results["TC-25"].outcome == "na"
    assert results["E2E-01"].outcome == "pass"  # the file is judged without the manifest


def test_a_huge_model_never_makes_the_suite_read_it_into_memory(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """D3, end to end: a toolchain that writes a sparse 3 GiB `model.onnx` and no manifest."""
    import subprocess
    import sys

    code = ("import resource, sys\nfrom oaax_conformance import cli\nrc = cli.main(sys.argv[1:])\n"
            "print('MAXRSS_MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)\nsys.exit(rc)")
    argv = ["toolchain", CLEAN, "--docker-arg=-e", "--docker-arg=OAAX_MOCK_FAULT=huge_model", "--runtime", str(build_dir / "libmock_runtime.so"), "--harness",
            str(build_dir / "oaax-harness"), "--init-config", "vendor_key=1", "--only", "E2E-01", "--only", "E2E-04", "--quiet"]
    done = subprocess.run([sys.executable, "-c", code, *argv], capture_output=True, text=True, timeout=600)
    peak = int(done.stdout.rsplit("MAXRSS_MB", 1)[1])
    assert "E2E-04" in done.stdout and "ERROR" not in done.stdout.split("Details")[-1].split("Result")[0], done.stdout[-1500:]
    assert "GB: the suite does not read a model that big" in done.stdout.replace("\n", " ") or "GB" in done.stdout
    assert peak < 1500, f"peak memory {peak} MB"

# ---- round 7 (D4): a failed or impossible copy leaves nothing behind, a deep or huge output tree is a clean BLOCKED ------------------------------

def _deep_tree(root, levels):
    import os

    here = os.getcwd()
    os.chdir(root)
    try:
        for _ in range(levels):
            os.mkdir("d")
            os.chdir("d")
    finally:
        os.chdir(here)


@pytest.fixture
def private_tmp(tmp_path, monkeypatch):
    """tempfile creates the suite's workspaces here: what is left over after a run is visible (other tests running in parallel cannot interfere)."""
    import tempfile

    base = tmp_path / "tmp"
    base.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(base))
    return base


def test_an_unreadable_file_blocks_tc08_and_leaves_no_workspace_behind(build_dir, mock_io_onnx, toolchain_images, tmp_path, private_tmp):
    """D4: Workspace.with_output_of() created the copy, failed on a mode-000 file, and nobody ever removed it (it was registered with the context only in convert())."""
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "unreadable_file", tmp_path), ["TC-01", "TC-08"])
    assert results["TC-01"].outcome == "pass"
    assert results["TC-08"].outcome == "blocked" and "cannot copy" in results["TC-08"].message
    assert sorted(p.name for p in private_tmp.iterdir()) == []


def test_a_tree_of_1700_levels_is_a_clean_block_and_leaves_nothing_behind(build_dir, mock_io_onnx, toolchain_images, tmp_path, private_tmp):
    """D4: TC-08's recursive copy raised RecursionError (ERROR, exit 4, "a bug of the suite") and the half-copied tree stayed."""
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "deep_tree", tmp_path), ["TC-01", "TC-08"])
    assert [(i, r.message) for i, r in results.items() if r.outcome == "error"] == []
    assert results["TC-01"].outcome == "blocked" and "deeper than 100 levels" in results["TC-01"].message
    assert sorted(p.name for p in private_tmp.iterdir()) == []


def test_the_walkers_are_iterative_and_capped(tmp_path):
    from oaax_conformance import toolchain
    from oaax_conformance.toolchain import TreeTooBig, Workspace, remove_tree, tree

    ws = Workspace.create()
    try:
        _deep_tree(ws.out_dir, 1700)
        with pytest.raises(TreeTooBig, match="deeper than 100 levels"):
            tree(ws.out_dir)
        toolchain._drop_big_files(ws.out_dir)  # best effort, never an exception
        with pytest.raises(TreeTooBig):
            Workspace.with_output_of(ws)
    finally:
        remove_tree(ws.root)  # 1700 levels, without recursion
    assert not ws.root.exists()


def test_a_normal_deep_tree_is_copied_and_the_entry_cap_applies(tmp_path, monkeypatch):
    from oaax_conformance import toolchain
    from oaax_conformance.toolchain import TreeTooBig, Workspace, remove_tree, tree

    ws = Workspace.create()
    try:
        _deep_tree(ws.out_dir, 90)
        (ws.out_dir / "a.json").write_text("{}")
        copy = Workspace.with_output_of(ws)
        try:
            assert set(tree(copy.out_dir)) == set(tree(ws.out_dir)) and len(tree(copy.out_dir)) == 91
        finally:
            remove_tree(copy.root)
        monkeypatch.setattr(toolchain, "MAX_TREE_ENTRIES", 50)
        with pytest.raises(TreeTooBig, match="more than 50 entries"):
            tree(ws.out_dir)
    finally:
        remove_tree(ws.root)


def test_e2e_03_does_not_take_the_manifest_of_another_file_for_the_judged_ones_io(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """D5: with --model-file a-copy.bin the manifest (which names model.bin) was used for E2E-03 while E2E-02/04 correctly ignored it: a MUST PASSED on I/O it never read."""
    ctx = _context(build_dir, mock_io_onnx, CLEAN, "two_models", tmp_path)
    ctx.model_file = "a-copy.bin"
    results = _run(ctx, ["E2E-01", "E2E-02", "E2E-03", "E2E-04"])
    assert results["E2E-01"].outcome == "pass"
    assert results["E2E-03"].outcome == "blocked" and "the manifest describes model.bin, not a-copy.bin" in results["E2E-03"].message
    assert results["E2E-02"].outcome == "blocked" and results["E2E-04"].outcome == "blocked"  # the same file judged by all four, with the same information
    for r in results.values():
        assert "a-copy.bin" in r.note


def test_a_manifest_that_names_the_judged_file_still_counts(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, None, tmp_path), ["E2E-03"])
    assert results["E2E-03"].outcome == "pass"


# ---- round 7 (D3 addendum, D2) ------------------------------------------------------------------------------------------------------------------

def test_big_files_are_not_hashed_and_a_change_is_still_seen(tmp_path):
    """D3 addendum: tree() hashed every regular file: a sparse 16 GiB file cost 31 s, for each of ~15 conversions per run. Above 512 MiB size and mtime stand in."""
    import os
    import time

    from oaax_conformance.toolchain import HASH_LIMIT, tree

    with open(tmp_path / "big.bin", "wb") as f:
        f.truncate(8 << 30)  # sparse: 8 GiB that hashing would read whole (about 15 s)
    (tmp_path / "small.json").write_text("{}")
    started = time.monotonic()
    first = tree(tmp_path)
    assert time.monotonic() - started < 5, "the sparse file was hashed"
    assert first["big.bin"]["sha256"] is None and first["big.bin"]["size"] > HASH_LIMIT and "mtime_ns" in first["big.bin"]
    assert len(first["small.json"]["sha256"]) == 64 and "mtime_ns" not in first["small.json"]
    os.utime(tmp_path / "big.bin", ns=(1, 1))
    assert tree(tmp_path)["big.bin"] != first["big.bin"]  # touched: input_changes() compares these dicts
    with open(tmp_path / "big.bin", "ab") as f:
        f.truncate(9 << 30)
    assert tree(tmp_path)["big.bin"]["size"] == 9 << 30


def test_a_manifest_that_names_a_link_is_not_valid_and_the_end_to_end_tests_agree(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """D2: validate() followed the link (TC-24 PASS) while the suite's listing does not: E2E ignored the manifest and judged another file."""
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "manifest_names_link", tmp_path), ["TC-24", "E2E-01", "E2E-02", "E2E-03"])
    assert results["TC-24"].outcome in ("warn", "fail") and "model.lnk" in results["TC-24"].message and "a link" in results["TC-24"].message
    assert results["E2E-01"].outcome == "pass"
    assert [i for i, r in results.items() if r.outcome == "error"] == []
    assert "(the only file)" in results["E2E-01"].note  # the unusable manifest is not used to choose, nor to describe, the file
    assert results["E2E-03"].outcome != "pass"  # not judged against a manifest that does not describe the judged file


# ---- round 7 (D9): the cleanup container keeps the sandbox ----------------------------------------------------------------------------------

def test_the_cleanup_container_runs_with_the_sandbox_and_still_removes_what_another_uid_made(build_dir, mock_io_onnx, toolchain_images, tmp_path, monkeypatch):
    """D9: the container that opens up undeletable output ran as root WITHOUT the flags the README says every run has."""
    import os
    import subprocess

    from oaax_conformance.toolchain import Workspace

    ctx = _context(build_dir, mock_io_onnx, CLEAN, None, tmp_path)
    ws = Workspace.create()
    ctx.workspaces.append(ws)
    made = subprocess.run(["docker", "run", "--rm", "--user", "12345:12345", "-v", f"{ws.out_dir}:/out", "--entrypoint", "sh", CLEAN, "-c",
                           "mkdir -m 700 /out/sub && mkdir -m 000 /out/sub/locked && touch /out/sub/f && chmod 000 /out/sub/f && chmod 500 /out/sub"],
                          capture_output=True, text=True)
    assert made.returncode == 0, made.stderr
    assert (ws.out_dir / "sub").exists() and not os.access(ws.out_dir / "sub", os.W_OK)  # the host user cannot remove it
    seen = []
    real = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: (seen.append(list(cmd)), real(cmd, *a, **k))[1])
    ctx.close()
    assert not ws.root.exists() and ctx.leftovers == []  # removed, with only FOWNER and DAC_OVERRIDE
    cleanup = next(c for c in seen if "--entrypoint" in c and "chmod" in c)
    joined = " ".join(cleanup)
    assert "--cap-drop ALL" in joined and "--cap-add FOWNER" in joined and "--cap-add DAC_OVERRIDE" in joined
    assert "--security-opt no-new-privileges" in joined and "--pids-limit 4096" in joined and "--network none" in joined


def test_a_deeply_nested_manifest_is_a_finding_not_a_crash(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    """F1: 200 KB of "[" is far under the 10 MB cap but over the recursion limit of the JSON parser: seven checks ended in ERROR (exit 4, "a bug of the suite")."""
    results = _run(_context(build_dir, mock_io_onnx, CLEAN, "nested_manifest", tmp_path), ["TC-24", "TC-25", "E2E-01", "E2E-02", "E2E-03", "E2E-04"])
    assert {i: r.outcome for i, r in results.items() if r.outcome == "error"} == {}
    assert results["TC-24"].outcome in ("fail", "warn")  # the manifest is not valid, said as a finding


def test_the_entry_cap_fires_while_listing_not_after_holding_the_whole_directory(tmp_path, monkeypatch):
    """F3: walk() sorted the complete scandir listing before checking the cap (1.5M files: 419 MB and 6 s before TreeTooBig)."""
    from oaax_conformance import toolchain

    monkeypatch.setattr(toolchain, "MAX_TREE_ENTRIES", 50)
    for i in range(200):
        (tmp_path / f"f{i}").write_text("x")
    seen = []
    real_scandir = toolchain.os.scandir

    class Counting:
        def __init__(self, path):
            self.inner = real_scandir(path)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.inner.close()

        def __iter__(self):
            for entry in self.inner:
                seen.append(entry.name)
                yield entry

    monkeypatch.setattr(toolchain.os, "scandir", Counting)
    with pytest.raises(toolchain.TreeTooBig):
        list(toolchain.walk(tmp_path))
    assert len(seen) <= 51  # stopped at the cap, did not read the other 150

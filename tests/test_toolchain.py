"""The toolchain tier tests itself: the clean mock toolchain passes everything, every check fails on its own fault.
Needs Docker (the mock toolchain is an image); skipped without it."""
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


def test_ctrl_c_removes_the_running_container(build_dir, toolchain_images, tmp_path):
    """A12: an interrupted run must not leave its toolchain container running."""
    import os
    import signal
    import sys

    import uuid

    label = f"oaax-test={uuid.uuid4().hex}"  # parallel tests run containers too: find only this one
    proc = subprocess.Popen([sys.executable, "-m", "oaax_conformance.cli", "toolchain", CLEAN, "--docker-arg=-e", "--docker-arg=OAAX_MOCK_FAULT=hang",
                             f"--docker-arg=--label={label}", "--only", "TC-01", "--quiet", "--timeout", "120"], start_new_session=True,
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

    run = SimpleNamespace(out_path="/output", output={**entries, "ok.onnx": {"dir": False, "size": 1}})
    assert list(Conversion.files(run)) == ["ok.onnx"]  # links are neither models nor logs


def test_b8_a_slow_toolchain_is_blocked_with_a_hint_not_failed(build_dir, mock_io_onnx, toolchain_images, tmp_path):
    ctx = _context(build_dir, mock_io_onnx, CLEAN, "hang", tmp_path)
    ctx.timeout = 5
    results = _run(ctx, ["TC-01", "TC-30"])
    assert {r.outcome for r in results.values()} == {"blocked"} and "--timeout" in results["TC-01"].message

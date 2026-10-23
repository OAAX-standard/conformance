"""The toolchain tier tests itself: the clean mock toolchain passes everything, every check fails on its own fault.
Needs Docker (the mock toolchain is an image); skipped without it."""
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
                            init_config=["vendor_key=1"], runtime_timeout=60)


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

"""One regression test per finding of the adversary reviews (A1.. = round 1), each driven by a fault of the mock runtime or toolchain."""
import json
import subprocess
import time
import xml.etree.ElementTree as ET

import helpers
import pytest
from oaax_conformance import cli, report
from oaax_conformance.harness import run_scenario

INPUTS = [("x", "FLOAT", (1, 4)), ("scale", "FLOAT", (1,))]


def _run(build_dir, fault, only, monkeypatch, init_config=("vendor_key=1",), **overrides):
    if fault:
        monkeypatch.setenv("OAAX_MOCK_FAULT", fault)
    return helpers.run(helpers.context(build_dir, init_config=init_config, **overrides), only)


def _scenario(build_dir, scenario, extra=(), config=("vendor_key=1",), timeout=60):
    lib, harness = str(build_dir / "libmock_runtime.so"), str(build_dir / "oaax-harness")
    return run_scenario(harness, scenario, lib, list(config), timeout, ["-m", str(build_dir / "mock.model"), *extra])


def test_every_check_runs_on_the_clean_mock_without_a_suite_error(build_dir):
    """A24: the fault tests only look at the faulted run; here nothing at all may crash."""
    results = helpers.run(helpers.context(build_dir))
    assert [i for i, r in results.items() if r.outcome == "error"] == []


# ---- A1: safe on a user's machine -------------------------------------------------------------------------


def test_a_flood_of_big_requests_is_capped_by_bytes(build_dir, monkeypatch):
    monkeypatch.setenv("OAAX_MOCK_FAULT", "any_shape")  # takes any size: 4 MB per request here
    run = _scenario(build_dir, "flood", ["--flood", "1000", "-i", "x:FLOAT:262144,4", "-i", "scale:FLOAT:1"])
    flood = run.step("flood")
    assert flood["requested"] == 1000 and flood["requests"] <= 70  # 256 MB / 4 MB = 64


def test_skip_and_hardware_safe_leave_tests_out(build_dir, tmp_path):
    out = tmp_path / "out"
    base = ["runtime", str(build_dir / "libmock_runtime.so"), "--harness", str(build_dir / "oaax-harness"), "--model", str(build_dir / "mock.model"),
            "--init-config", "vendor_key=1", "--quiet", "--pause", "0", "--only", "RT-M-33", "--only", "RT-M-08", "--only", "RT-L-11", "--out", str(out)]
    assert cli.main(base + ["--skip", "RT-M-08"]) == 0
    outcomes = {r["id"]: r["outcome"] for r in json.loads((out / "report.json").read_text())["results"]}
    assert outcomes["RT-M-08"] == "skipped" and outcomes["RT-M-33"] == "pass"
    assert cli.main(base + ["--hardware-safe"]) == 0
    doc = json.loads((out / "report.json").read_text())
    assert {r["id"]: r["outcome"] for r in doc["results"]} == {"RT-M-33": "skipped", "RT-M-08": "pass", "RT-L-11": "skipped"}
    assert "hardware-safe" in next(r["message"] for r in doc["results"] if r["id"] == "RT-M-33")


# ---- A2: a slow device is not a failing one ---------------------------------------------------------------


def test_a_slow_first_inference_does_not_fail_the_round_trip(build_dir, monkeypatch):
    # the first request takes 3 s, the round-trip budget is 1.5 s: only the warm-up calibration lets this pass
    results = _run(build_dir, "slow_first", ["RT-M-08", "RT-M-02", "RT-M-09"], monkeypatch, warmup_ms=20000)
    assert {i: r.outcome for i, r in results.items()} == {"RT-M-08": "pass", "RT-M-02": "pass", "RT-M-09": "pass"}


def test_a_scenario_that_cannot_finish_in_time_is_blocked_not_failed(build_dir, monkeypatch):
    results = _run(build_dir, "slow_init", ["RT-L-01"], monkeypatch, timeout=5)  # five inits of 2.5 s do not fit
    assert results["RT-L-01"].outcome == "blocked" and "--timeout" in results["RT-L-01"].message


def test_loops_stop_at_the_time_budget_and_the_check_says_so(build_dir, monkeypatch):
    results = _run(build_dir, "slow_init", ["RT-L-07"], monkeypatch, timeout=10, cycles=50)  # budget 6 s: 2 cycles of 2.5 s
    assert results["RT-L-07"].outcome == "blocked" and "too slow to repeat" in results["RT-L-07"].message


# ---- A3: the vendor key --------------------------------------------------------------------------------------


def test_a_runtime_that_needs_a_vendor_key_blocks_instead_of_failing(build_dir, monkeypatch):
    results = _run(build_dir, "require_config", ["RT-L-01", "RT-L-03"], monkeypatch, init_config=("other=1",))  # a config, but not the key it needs
    assert {r.outcome for r in results.values()} == {"blocked"} and "--init-config" in results["RT-L-01"].message


# ---- A4: native views ------------------------------------------------------------------------------------------


def test_a_runtime_that_takes_another_type_blocks_with_a_hint_and_passes_with_input(build_dir, monkeypatch):
    results = _run(build_dir, "u8_only", ["RT-M-08"], monkeypatch)
    assert results["RT-M-08"].outcome == "blocked" and "--input" in results["RT-M-08"].message
    ctx = helpers.context(build_dir, manual_inputs=["x:UINT8:1,4", "scale:FLOAT:1"])
    assert helpers.run(ctx, ["RT-M-08", "RT-M-02"]) == {i: r for i, r in helpers.run(ctx, ["RT-M-08", "RT-M-02"]).items()}
    assert {i: r.outcome for i, r in helpers.run(ctx, ["RT-M-08", "RT-M-02"]).items()} == {"RT-M-08": "pass", "RT-M-02": "pass"}


# ---- A5: dynamic dimensions ------------------------------------------------------------------------------------


def test_a_wrong_shape_is_only_built_from_a_fixed_dimension(tmp_path, build_dir, monkeypatch):
    monkeypatch.setenv("OAAX_MOCK_FAULT", "dynamic_dim0")  # accepts any first dimension
    ctx = helpers.context(build_dir)
    ctx.model = helpers.model_variant(tmp_path, [("x", "FLOAT", (None, 4)), ("scale", "FLOAT", (1,))])
    assert helpers.run(ctx, ["RT-M-23", "RT-M-08"]) and {i: r.outcome for i, r in helpers.run(ctx, ["RT-M-23", "RT-M-08"]).items()} == {"RT-M-23": "pass", "RT-M-08": "pass"}


def test_when_every_dimension_is_dynamic_no_shape_is_wrong(tmp_path, build_dir, monkeypatch):
    monkeypatch.setenv("OAAX_MOCK_FAULT", "any_shape")
    ctx = helpers.context(build_dir)
    ctx.model = helpers.model_variant(tmp_path, [("x", "FLOAT", (None, None)), ("scale", "FLOAT", (None,))])
    assert helpers.run(ctx, ["RT-M-23"])["RT-M-23"].outcome == "na"


# ---- A6 / M3: a crash after the work is done -------------------------------------------------------------------


def test_a_crash_in_a_static_destructor_is_not_blamed_on_a_rejected_input(build_dir, monkeypatch):
    results = _run(build_dir, "crash_after_end", ["RT-M-27", "RT-L-11"], monkeypatch)
    assert results["RT-M-27"].outcome == "pass"  # the process died, but after its last word
    assert results["RT-L-11"].outcome != "pass"  # that is RT-L-11's business


# ---- A7: a runtime that forks -----------------------------------------------------------------------------------


def test_a_runtime_that_forks_a_helper_does_not_hang_the_suite(build_dir, monkeypatch):
    import uuid

    marker = f"marker{uuid.uuid4().hex}"  # on the command line of this test's processes only: parallel tests are not confused with it
    started = time.monotonic()
    results = _run(build_dir, "daemonize", ["RT-S-01", "RT-L-03", "RT-L-11"], monkeypatch, init_config=("vendor_key=1", f"{marker}=1"), timeout=30)
    assert time.monotonic() - started < 20 and results["RT-L-03"].outcome == "pass"
    assert subprocess.run(["pgrep", "-f", marker], capture_output=True).returncode == 1  # no helper left behind


# ---- A8: suite errors are not vendor verdicts ---------------------------------------------------------------------


def test_a_status_outside_the_header_is_shown_not_a_crash(build_dir, monkeypatch):
    result = _run(build_dir, "weird_status", ["RT-L-13"], monkeypatch)["RT-L-13"]
    assert result.outcome == "warn" and "unknown status 99" in result.message


def test_thread_names_are_bytes(build_dir, monkeypatch):
    results = _run(build_dir, "bad_thread_name", ["RT-L-01", "RT-L-08"], monkeypatch)
    assert results["RT-L-01"].outcome == "pass" and results["RT-L-08"].outcome == "warn" and "bad" in results["RT-L-08"].message


def test_an_unreadable_model_blocks(tmp_path, build_dir, monkeypatch):
    garbage = tmp_path / "bad.onnx"
    garbage.write_bytes(b"\x00\x01 not protobuf")
    ctx = helpers.context(build_dir)
    ctx.model = str(garbage)
    results = helpers.run(ctx, ["RT-M-08"])
    assert results["RT-M-08"].outcome == "blocked" and "--input" in results["RT-M-08"].message


def test_a_suite_error_has_its_own_exit_code():
    result = lambda o, lv="MUST": report.Result("X", "t", lv, "s", o)  # noqa: E731
    assert report.exit_code([result("error", "PROPOSED")]) == 4
    assert report.exit_code([result("error"), result("fail")]) == 1  # a real MUST failure stands
    assert report.exit_code([result("skipped"), result("pass")]) == 0


# ---- A10: one model at a time ---------------------------------------------------------------------------------------


def test_a_runtime_that_holds_one_model_still_gets_the_concurrency_tests(build_dir, monkeypatch):
    results = _run(build_dir, "one_model", ["RT-M-05", "RT-M-29", "RT-M-30"], monkeypatch)
    assert results["RT-M-05"].outcome == "na" and results["RT-M-29"].outcome == "pass" and results["RT-M-30"].outcome == "pass"


# ---- A11 / A18: what reaches a report ----------------------------------------------------------------------------


def test_secrets_and_control_characters_never_reach_a_report(build_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OAAX_MOCK_FAULT", "echo_config")
    out = tmp_path / "out"
    cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--harness", str(build_dir / "oaax-harness"), "--init-config", "vendor_key=SECRET123",
              "--only", "RT-L-13", "--only", "RT-L-05", "--quiet", "--out", str(out)])
    assert "SECRET123" not in capsys.readouterr().out
    for name in ("report.json", "report.html", "junit.xml"):
        assert "SECRET123" not in (out / name).read_text(), name


def test_control_characters_do_not_break_the_xml_report(build_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OAAX_MOCK_FAULT", "ctrl_err")
    out = tmp_path / "out"
    cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--harness", str(build_dir / "oaax-harness"), "--init-config", "vendor_key=1",
              "--only", "RT-L-05", "--only", "RT-S-09", "--quiet", "--out", str(out)])
    ET.parse(out / "junit.xml")  # well-formed
    assert "\x1b" not in capsys.readouterr().out


# ---- A15 / A25: the command line ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [
    ["runtime", "/nonexistent.so"],
    ["runtime", "{lib}", "--only", "RT-X-99"],
    ["runtime", "{lib}", "--skip", "TC-01"],
    ["runtime", "{lib}", "--init-config", "novalue"],
    ["runtime", "{lib}", "--harness", "/bad"],
    ["runtime", "{lib}", "--model", "/nonexistent.onnx"],
    ["runtime", "{lib}", "--input", "x:FLOAT:abc"],
    ["runtime", "{lib}", "--workdir", "/nonexistent"],
    ["runtime", "{lib}", "--timeout", "1"],
    ["runtime", "{lib}", "--bogus"],
    ["toolchain", "x", "--docker-arg=--user=0"],
    ["toolchain", "x", "--docker-arg=--network=host"],
])
def test_a_bad_command_line_exits_with_3_before_running_anything(build_dir, tmp_path, argv):
    argv = [a.replace("{lib}", str(build_dir / "libmock_runtime.so")) for a in argv]
    if argv[0] == "runtime" and "--harness" not in argv:
        argv += ["--harness", str(build_dir / "oaax-harness")]
    with pytest.raises(SystemExit) as stop:
        code = cli.main(argv)
        raise SystemExit(code)
    assert stop.value.code == 3


def test_out_that_is_a_file_is_refused(build_dir, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--out", str(blocker)]) == 3


def test_a_library_that_is_not_elf_does_not_pass_the_export_check(tmp_path, build_dir):
    fake = tmp_path / "libfake.so"
    fake.write_text("not a library")
    results = helpers.run(helpers.context(build_dir, model=False), ["RT-S-02", "RT-S-01"])
    ctx = helpers.context(build_dir, model=False)
    ctx.lib = str(fake)
    results = helpers.run(ctx, ["RT-S-02", "RT-S-01"])
    assert results["RT-S-02"].outcome == "blocked" and "ELF" in results["RT-S-02"].message
    assert results["RT-S-01"].outcome == "blocked" and "LD_LIBRARY_PATH" in results["RT-S-01"].message


# ---- A16: what the runtime itself says ------------------------------------------------------------------------------------


def test_the_runtimes_own_output_is_kept_for_the_report(build_dir, monkeypatch):
    monkeypatch.setenv("OAAX_MOCK_FAULT", "chatty")
    run = _scenario(build_dir, "lifecycle")
    assert any("noisy line" in o.get("value", "") for o in run.observations() if str(o["call"]).startswith("(the runtime's own"))


# ---- A17: a runtime that calls exit() --------------------------------------------------------------------------------------


def test_exit_inside_a_call_is_blamed_on_the_call_not_on_loading(build_dir, monkeypatch):
    result = _run(build_dir, "exit_in_call", ["RT-L-13"], monkeypatch)["RT-L-13"]
    assert result.outcome == "warn" and "ended the process (exit code 3)" in result.message and "does not load" not in result.message


# ---- A19 / A20: the harness survives what a runtime returns ------------------------------------------------------------------


@pytest.mark.parametrize("fault", ["huge_name", "big_size"])
def test_the_harness_does_not_crash_on_oversized_names_or_sizes(build_dir, monkeypatch, fault):
    monkeypatch.setenv("OAAX_MOCK_FAULT", fault)
    run = _scenario(build_dir, "roundtrip", ["-i", "x:FLOAT:1,4", "-i", "scale:FLOAT:1"])
    assert run.end == "ok" and run.step("roundtrip")["status"] == 0


# ---- A21: tolerance ----------------------------------------------------------------------------------------------------------------


def test_floats_that_differ_below_the_tolerance_are_the_same_result(build_dir, monkeypatch):
    assert _run(build_dir, "jitter", ["RT-M-30"], monkeypatch)["RT-M-30"].outcome == "pass"
    assert _run(build_dir, "corrupt_every_7th", ["RT-M-30"], monkeypatch)["RT-M-30"].outcome == "warn"


# ---- M2: a failing cycle is not a completed one ----------------------------------------------------------------------------------------


def test_cycles_that_fail_are_not_counted_as_completed(build_dir, monkeypatch):
    monkeypatch.setenv("OAAX_MOCK_FAULT", "no_reinit")  # init fails after the first cleanup
    cycles = _scenario(build_dir, "cycles", ["--cycles", "5"]).step("cycles")
    assert cycles["completed"] == 0 and cycles["init_failures"] == 5


# ---- A22: the docs -----------------------------------------------------------------------------------------------------------------------


def test_the_test_readme_lists_every_specification_file():
    from conftest import ROOT

    readme = (ROOT / "docs" / "tests" / "README.md").read_text()
    for path in (ROOT / "docs" / "tests").glob("*.md"):
        assert path.name == "README.md" or path.name in readme, path.name


def test_the_clean_mock_passes_at_the_default_sizes(build_dir):
    """A24: the other tests use small sizes; this is what a user runs (1000-request flood, 50 cycles, 4 x 25 concurrent). CI repeats it under ASAN/UBSAN."""
    ctx = helpers.context(build_dir, settle_ms=3000, cycles=50, flood=1000, requests=500, producers=4, per_producer=25, model_cycles=20, lost_after_ms=3000,
                          retrieve_timeout_ms=10000, warmup_ms=120000, timeout=300)
    assert {i: r.outcome for i, r in helpers.run(ctx).items() if r.outcome != "pass"} == {}

"""One regression test per finding of the adversary reviews (A1.. = round 1), each driven by a fault of the mock runtime or toolchain."""
import json
import re
import subprocess
import time
import xml.etree.ElementTree as ET

import helpers
import pytest
from oaax_conformance import cli, report
from oaax_conformance.harness import Run, run_scenario

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
    assert cli.main(base + ["--skip", "RT-M-08"]) == 2  # B5: a skipped MUST is never a clean verdict
    outcomes = {r["id"]: r["outcome"] for r in json.loads((out / "report.json").read_text())["results"]}
    assert outcomes["RT-M-08"] == "skipped" and outcomes["RT-M-33"] == "pass"
    assert cli.main(base + ["--hardware-safe"]) == 0  # the tests it skips are not MUSTs
    doc = json.loads((out / "report.json").read_text())
    assert {r["id"]: r["outcome"] for r in doc["results"]} == {"RT-M-33": "skipped", "RT-M-08": "pass", "RT-L-11": "skipped"}
    assert "hardware-safe" in next(r["message"] for r in doc["results"] if r["id"] == "RT-M-33")


# ---- A2: a slow device is not a failing one ---------------------------------------------------------------


def test_a_slow_first_inference_does_not_fail_the_round_trip(build_dir, monkeypatch):
    # the first request takes 3 s, the round-trip budget is 1.5 s: only the warm-up calibration lets this pass
    results = _run(build_dir, "slow_first", ["RT-M-08", "RT-M-02", "RT-M-09"], monkeypatch, warmup_ms=20000)
    assert {i: r.outcome for i, r in results.items()} == {"RT-M-08": "pass", "RT-M-02": "pass", "RT-M-09": "pass"}


def test_a_device_that_is_slow_on_every_request_gets_a_timeout_that_matches(build_dir, monkeypatch):
    # every request takes 3 s, the round-trip budget is 1.5 s: the first answered request must raise the timeout of the later ones
    results = _run(build_dir, "slow_every", ["RT-M-08", "RT-M-09"], monkeypatch, warmup_ms=20000)
    assert {i: r.outcome for i, r in results.items()} == {"RT-M-08": "pass", "RT-M-09": "pass"}


def test_a_call_that_stalls_is_blocked_not_failed(build_dir, monkeypatch):
    results = _run(build_dir, "hang_init", ["RT-L-01"], monkeypatch, timeout=5)
    assert results["RT-L-01"].outcome == "blocked" and "--timeout" in results["RT-L-01"].message


def test_a_slow_init_repeated_in_one_scenario_is_not_cut_off(build_dir, monkeypatch):
    """The timeout is per call: five inits of 2.5 s (12.5 s) under --timeout 5 used to BLOCK every lifecycle check (a 76 s init hit the 300 s default)."""
    results = _run(build_dir, "slow_init", ["RT-L-01", "RT-L-03"], monkeypatch, timeout=5)
    assert {r.outcome for r in results.values()} == {"pass"}


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
    assert report.exit_code([result("skipped"), result("pass")]) == 2  # B5: a skipped MUST
    assert report.exit_code([result("skipped", "SHOULD"), result("pass")]) == 0


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
    for name in ("junit.xml", "report.json", "report.html"):  # the message really carried control characters; every format shows them escaped
        text = (out / name).read_text()
        assert "\x1b" not in text and "\x01" not in text, name
        assert "\\x1b" in text or name == "junit.xml", name  # junit carries messages of failed and blocked tests only


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
def test_a_bad_command_line_exits_with_3_before_running_anything(build_dir, tmp_path, argv, capsys):
    argv = [a.replace("{lib}", str(build_dir / "libmock_runtime.so")) for a in argv]
    if argv[0] == "runtime" and "--harness" not in argv:
        argv += ["--harness", str(build_dir / "oaax-harness")]
    with pytest.raises(SystemExit) as stop:
        code = cli.main(argv)
        raise SystemExit(code)
    assert stop.value.code == 3
    if "--docker-arg" in " ".join(argv):  # refused for the flag, not because image "x" does not exist
        assert "the suite sets this itself" in capsys.readouterr().err


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


@pytest.mark.parametrize("fault", ["huge_name", "big_size", "many_huge_outputs"])
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


# ---- adversary round 2 (B1..) -----------------------------------------------------------------------------


def test_b1_a_real_time_signal_is_named_not_a_traceback(build_dir, monkeypatch):
    assert Run("x", -40, False).signal_name == "signal 40" and Run("x", -11, False).signal_name == "SIGSEGV" and Run("x", 3, False).signal_name is None
    assert "40" in Run("x", -40, False).how_it_ended()
    monkeypatch.setattr(cli, "_runtime", lambda args: 1 / 0)  # any other bug of the suite: exit 4, never 1
    assert cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--quiet"]) == 4


def test_b3_a_crash_inside_cleanup_fails_the_cleanup_check(build_dir, monkeypatch):
    """B3: the first init/cleanup cycle is a throw-away one; a crash in it was only BLOCKED, so the run ended INCOMPLETE instead of FAILED."""
    results = _run(build_dir, "crash_in_cleanup", ["RT-L-17"], monkeypatch)
    assert results["RT-L-17"].outcome == "fail" and "SIGSEGV" in results["RT-L-17"].message


def test_b5_hardware_safe_skipping_a_must_is_incomplete(build_dir):
    assert cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--harness", str(build_dir / "oaax-harness"), "--model", str(build_dir / "mock.model"),
                     "--init-config", "vendor_key=1", "--quiet", "--pause", "0", "--hardware-safe", "--only", "RT-M-18"]) == 2


def test_b6_an_unwritable_out_is_refused_before_the_run(build_dir, capsys):
    assert cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--quiet", "--out", "/proc/nope/out"]) == 3
    assert "--out" in capsys.readouterr().err


def test_b7_an_unknown_input_dtype_is_blocked_not_a_crashed_scenario(build_dir, tmp_path, monkeypatch):
    ctx = helpers.context(build_dir)
    ctx.model = helpers.model_variant(tmp_path, [("x", "STRING", (1, 4))])
    results = helpers.run(ctx, ["RT-M-08"])
    assert results["RT-M-08"].outcome == "blocked" and "STRING" in results["RT-M-08"].message


def test_b9_a_timeout_before_a_call_says_how_to_raise_it(build_dir):
    run = Run("lifecycle", -9, True, lines=[{"begin": "init_plain"}], timeout=5)
    assert "--timeout" in run._why_missing("init_unknown_key")


def test_b10_the_scenarios_scratch_dir_and_huge_texts_do_not_reach_reports():
    import tempfile

    text = report.clean_text(f"cannot open {tempfile.gettempdir()}/tmpkqeophgp/dev.cfg " + "x" * 50000)
    assert "tmpkqeophgp" not in text and "<tmp>/dev.cfg" in text and len(text) < 20100 and "more characters" in text


@pytest.mark.parametrize("fault", ["null_tensors", "negative_tensors"])
def test_c1_malformed_output_tensors_fail_the_check_and_do_not_crash_the_harness(build_dir, monkeypatch, fault):
    results = _run(build_dir, fault, ["RT-M-08", "RT-M-09", "RT-M-10"], monkeypatch)
    assert results["RT-M-09"].outcome == "fail"
    assert not {r.outcome for r in results.values()} & {"blocked", "error"}, {i: r.message for i, r in results.items()}


def test_c2_the_html_report_is_written_when_a_check_crashed(build_dir, tmp_path, monkeypatch):
    """C2: exit code 4 raised a KeyError in the HTML renderer, so the evidence for the bug report was the one file missing."""
    from oaax_conformance import registry

    def crash(ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(registry.CHECKS, "RT-S-03", crash)
    out = tmp_path / "out"
    code = cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--harness", str(build_dir / "oaax-harness"), "--init-config", "vendor_key=1",
                     "--only", "RT-S-03", "--quiet", "--out", str(out)])
    assert code == 4 and {p.name for p in out.iterdir()} == {"report.html", "report.json", "junit.xml"}


def test_h1_what_a_runtime_prints_is_captured_bounded(build_dir, monkeypatch):
    """H1: stdout/stderr went to temp files nobody capped (9 GB in 5 s in the adversary's repro); now a ring of the last bytes."""
    import os
    import subprocess
    import sys

    from oaax_conformance.harness import TailCapture

    cap = TailCapture(100)
    child = subprocess.Popen([sys.executable, "-c", "import sys\nfor i in range(200000): sys.stdout.write('x' * 100 + '%d\\n' % i)"], stdout=cap.write_fd)
    cap.child_started()
    child.wait()
    tail = cap.value()
    assert len(tail) <= 100 and tail.endswith(b"199999\n")
    import stat

    from oaax_conformance import harness

    kinds, real = [], subprocess.Popen

    def spy(*args, **kwargs):
        kinds.append(stat.S_ISFIFO(os.fstat(kwargs["stdout"]).st_mode) if isinstance(kwargs.get("stdout"), int) else False)
        return real(*args, **kwargs)

    monkeypatch.setattr(harness.subprocess, "Popen", spy)
    monkeypatch.setenv("OAAX_MOCK_FAULT", "spam_init")
    run = _scenario(build_dir, "lifecycle")
    assert kinds == [True], "the runtime's output must go to a pipe we read, not to a file"
    assert 0 < len(run.stdout_tail) <= 2000 and run.end is not None


# ---- adversary round 3: tests that stayed green when their fix was undone ---------------------------------------------


def test_the_console_hides_secrets_too(build_dir, tmp_path, monkeypatch, capsys):
    """Round 2's secret test never put a secret on the console (every check passed): here a failed model load echoes the config into the message."""
    bad = tmp_path / "bad.model"
    bad.write_text("garbage")
    monkeypatch.setenv("OAAX_MOCK_FAULT", "echo_config")
    cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--harness", str(build_dir / "oaax-harness"), "--init-config", "vendor_key=SECRET123",
              "--model", str(bad), "--input", "x:FLOAT:1,4", "--input", "scale:FLOAT:1", "--only", "RT-M-08", "--quiet"])
    console = capsys.readouterr().out
    assert "config=***" in console and "SECRET123" not in console


def test_hardware_safe_shrinks_every_load_and_pauses(build_dir):
    ctx, safe = helpers.context(build_dir), helpers.context(build_dir)
    ctx.cycles, ctx.model_cycles, ctx.flood, ctx.requests, ctx.producers, ctx.per_producer, ctx.pause = 20, 20, 1000, 1000, 8, 100, 0
    safe.__dict__.update(ctx.__dict__)
    safe.hardware_safe()
    assert (safe.cycles, safe.model_cycles, safe.flood, safe.requests, safe.producers, safe.per_producer) == (3, 3, 50, 50, 2, 10)
    assert safe.pause >= 3 and ctx.pause == 0


def test_b4_hashing_reads_in_chunks(tmp_path, monkeypatch):
    from pathlib import Path

    from oaax_conformance.toolchain import tree

    (tmp_path / "big.bin").write_bytes(b"y" * ((1 << 20) * 3 + 5))
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("a whole file was read into memory"))
    import hashlib

    assert tree(tmp_path)["big.bin"]["sha256"] == hashlib.sha256(b"y" * ((1 << 20) * 3 + 5)).hexdigest()


def test_junit_does_not_show_green_for_an_incomplete_run(tmp_path):
    results = [report.Result("A", "t", "MUST", "s", "pass"), report.Result("B", "t", "MUST", "s", "blocked", "cannot run"),
               report.Result("C", "t", "SHOULD", "s", "blocked", "cannot run"), report.Result("D", "t", "MUST", "s", "skipped", "by request")]
    from dataclasses import asdict

    path = tmp_path / "junit.xml"
    report.write_junit(str(path), {"spec_version": "2.0", "results": [asdict(r) for r in results]})
    suite = ET.parse(path).getroot()
    assert (suite.get("tests"), suite.get("failures"), suite.get("errors"), suite.get("skipped")) == ("4", "0", "2", "1")
    assert [c.find("error") is not None for c in suite.iter("testcase")] == [False, True, False, True]


# ---- round 4: the tests that stayed green when their fix was undone (round 3 mutation check) -----------------------------


def test_the_warmup_wait_is_capped_after_an_unanswered_first_request(build_dir, monkeypatch):
    from oaax_conformance import suite
    from oaax_conformance.harness import Run

    ctx = helpers.context(build_dir, warmup_ms=120000)
    monkeypatch.setattr(suite, "run_scenario", lambda *a, **k: Run("roundtrip", 0, False, [{"step": "warmup", "status": 6}]))
    ctx.run("roundtrip")
    assert ctx.warmup_ms == 10000, "the next scenario must not wait the full warm-up again"


def test_the_lost_after_window_follows_the_first_answered_request(build_dir, monkeypatch):
    """Accepted-and-answered must not be reported as 'silently lost' because the harness's own window was shorter than the device's pace."""
    monkeypatch.setenv("OAAX_MOCK_FAULT", "any_shape")
    ctx = helpers.context(build_dir, lost_after_ms=1)
    result = helpers.run(ctx, ["RT-M-23"])["RT-M-23"]
    assert result.outcome == "fail" and "and answered" in result.message, result


def test_the_byte_cap_holds_in_the_concurrency_scenario_too(build_dir, monkeypatch):
    monkeypatch.setenv("OAAX_MOCK_FAULT", "any_shape")  # takes any size: 4 MB per request
    run = _scenario(build_dir, "concurrency", ["--producers", "50", "--per-producer", "100", "-i", "x:FLOAT:262144,4", "-i", "scale:FLOAT:1"])
    assert run.step("run")["expected"] <= 70  # 256 MB / 4 MB = 64 (5000 were asked for)


def test_a_run_that_selected_no_test_is_not_a_clean_pass(build_dir, monkeypatch, capsys):
    from types import SimpleNamespace

    monkeypatch.setattr(cli, "run_suite", lambda *a, **k: SimpleNamespace(results=[], not_approved=[], leftovers=[]))
    assert cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--harness", str(build_dir / "oaax-harness"), "--quiet"]) == 3
    assert "no test ran" in capsys.readouterr().err


def test_only_with_a_draft_test_id_exits_3_not_0(build_dir, tmp_path, monkeypatch, capsys):
    """The real path, no stub: a draft test is a known id that runs nothing, and "0 checks" must not look like a pass."""
    from oaax_conformance import specdocs

    specs = tmp_path / "specs"
    specs.mkdir()
    for f in (specdocs.spec_dir()).glob("*.md"):
        (specs / f.name).write_text(f.read_text())
    text = (specs / "runtime-static-lifecycle.md").read_text()
    row = next(l for l in text.splitlines() if l.startswith("| RT-S-01 |"))
    (specs / "runtime-static-lifecycle.md").write_text(text.replace(row, row.rsplit("approved", 1)[0] + "draft |"))
    monkeypatch.setattr(specdocs, "spec_dir", lambda: specs)
    argv = ["runtime", str(build_dir / "libmock_runtime.so"), "--harness", str(build_dir / "oaax-harness"), "--quiet", "--only", "RT-S-01"]
    assert cli.main(argv) == 3
    assert "no test ran" in capsys.readouterr().err


@pytest.mark.parametrize("sig,code", [("SIGTERM", 143), ("SIGHUP", 129)])  # D4: a closed terminal is the same hole as a CI timeout
def test_d1_a_signal_to_the_suite_stops_the_harness(build_dir, tmp_path, monkeypatch, sig, code):
    """D1: a CI timeout (SIGTERM) used to kill only the suite and leave the harness, and the device it holds, running."""
    import shutil
    import signal
    import sys
    import uuid

    lib = tmp_path / f"libmock-{uuid.uuid4().hex}.so"  # a unique path to find our own harness by
    shutil.copy(build_dir / "libmock_runtime.so", lib)
    monkeypatch.setenv("OAAX_MOCK_FAULT", "hang_init")
    pgrep = lambda: subprocess.run(["pgrep", "-f", str(lib)], capture_output=True, text=True).stdout.split()  # noqa: E731
    suite = subprocess.Popen([sys.executable, "-m", "oaax_conformance.cli", "runtime", str(lib), "--harness", str(build_dir / "oaax-harness"),
                              "--init-config", "vendor_key=1", "--only", "RT-L-01", "--quiet", "--timeout", "60"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 20
    while len(pgrep()) < 2 and time.monotonic() < deadline:  # the suite itself and its harness both carry the path
        time.sleep(0.2)
    assert len(pgrep()) >= 2, "the harness never started"
    suite.send_signal(getattr(signal, sig))
    assert suite.wait(timeout=30) == code
    deadline = time.monotonic() + 15
    while pgrep() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert pgrep() == [], "the harness outlived the suite"
    assert b"Traceback" not in suite.stderr.read()


def test_a_hangup_that_nohup_ignored_stays_ignored(build_dir, monkeypatch):
    import signal

    seen = {}
    monkeypatch.setattr(cli, "_runtime", lambda args: seen.update(hup=signal.getsignal(signal.SIGHUP), term=signal.getsignal(signal.SIGTERM)) or 0)
    old = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        assert cli.main(["runtime", str(build_dir / "libmock_runtime.so")]) == 0
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN  # restored as it was
    finally:
        signal.signal(signal.SIGHUP, old)
    assert seen["hup"] is signal.SIG_IGN and seen["term"] is cli._terminate


def test_d5_a_duplicated_or_absurd_input_is_a_usage_error_not_a_verdict(build_dir, capsys, tmp_path):
    lib = str(build_dir / "libmock_runtime.so")
    for extra, expect in ((["--input", "x:FLOAT:1,4", "--input", "x:FLOAT:1,4"], "listed twice"), (["--input", "x:FLOAT:100000,100000"], "GiB")):
        assert cli.main(["runtime", lib, "--harness", str(build_dir / "oaax-harness"), "--quiet", *extra]) == 3
        assert expect in capsys.readouterr().err
    # the same two mistakes inside a manifest block the model checks instead of failing them
    ctx = helpers.context(build_dir)
    ctx.model = helpers.model_variant(tmp_path, [("x", "FLOAT", (1, 4)), ("x", "FLOAT", (1, 4))])
    result = helpers.run(ctx, ["RT-M-20"])["RT-M-20"]
    assert result.outcome == "blocked" and "twice" in result.message


@pytest.mark.parametrize("option", ["--pause", "--round-trip-timeout"])
@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_pause_and_round_trip_timeout_must_be_finite_seconds(build_dir, option, value, capsys):
    assert cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--harness", str(build_dir / "oaax-harness"), "--quiet", option, value]) == 3
    assert option in capsys.readouterr().err


def test_version_is_printed_and_comes_from_one_place(capsys):
    from oaax_conformance import __version__

    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0 and f"oaax-conformance {__version__}" in capsys.readouterr().out


# ---- round 5 ------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("args", [["-e", "T=s3cretvalue"], ["-eT=s3cretvalue"], ["-e=T=s3cretvalue"], ["--env", "T=s3cretvalue"], ["--env=T=s3cretvalue"],
                                  ["--gpus=all", "-e", "T=s3cretvalue"]])
def test_every_docker_env_form_registers_its_bare_value_as_a_secret(args):
    """A1/A3: `--env=NAME=VALUE` and `-e=NAME=VALUE` registered "NAME=VALUE" as the secret, so the value alone leaked into every report."""
    secrets = report.secret_values([], args)
    assert secrets == ["s3cretvalue"]
    shown = " ".join(report.mask_docker_args(args))
    assert "s3cretvalue" not in shown and "T=***" in shown
    assert "s3cretvalue" not in report.clean_text("failed: token=s3cretvalue", [], secrets)


def test_a_value_containing_equals_signs_is_kept_whole():
    assert report.secret_values([], ["-e", "T=a=b=c"]) == ["a=b=c"]


@pytest.mark.parametrize("args", [["-e", "T"], ["--env", "T"], ["--env=T"], ["-eT"], ["-e=T"]])
def test_a_variable_docker_takes_from_this_environment_is_a_secret_too(args, monkeypatch):
    """A3: `-e NAME` has no value in the arguments, but docker injects ours into the container and the toolchain may echo it."""
    monkeypatch.setenv("T", "s3cretvalue")
    assert report.secret_values([], args) == ["s3cretvalue"]
    assert report.mask_docker_args(args) == args  # nothing to hide in the arguments themselves
    monkeypatch.delenv("T")
    assert report.secret_values([], args) == []  # an unset variable has no value to leak


def test_an_env_file_is_refused_because_its_values_cannot_be_kept_out_of_the_reports(capsys):
    assert cli.main(["toolchain", "x", "--docker-arg=--env-file=/etc/hostname"]) == 3
    assert "-e NAME=VALUE" in capsys.readouterr().err


def test_text_that_xml_cannot_hold_is_escaped_so_junit_stays_parseable(tmp_path):
    """A2: U+FFFE/U+FFFF and lone surrogates are valid in a Python string and in UTF-8 but not in XML 1.0, and a toolchain's last output line is quoted in a message."""
    from dataclasses import asdict

    nasty = "fatal: ￾￿ \ud800 \x9b[31m ‮ rest"
    doc = {"spec_version": "2.0", "results": [asdict(report.Result("A", "t", "MUST", "s", "fail", nasty))]}
    path = tmp_path / "junit.xml"
    report.write_junit(str(path), report.scrub(doc, [], []))
    failure = ET.parse(path).getroot().find("testcase/failure")  # raises on malformed XML
    assert failure.get("message") == "fatal: \\ufffe\\uffff \\ud800 \\x9b[31m \\u202e rest"


# ---- round 5, A10: a timing the spec does not bound is never a failure ---------------------------------------------------------------

def test_a_slow_cleanup_with_requests_in_flight_is_a_warning_not_a_failure(build_dir, monkeypatch):
    """RT-M-31 failed any cleanup over 10 s although the spec sets no bound (the mock takes 1.2 s, the hint threshold is lowered to 0.5 s)."""
    from oaax_conformance.checks import runtime_model

    monkeypatch.setattr(runtime_model, "SLOW_CLEANUP_MS", 500)
    result = _run(build_dir, "slow_cleanup", ["RT-M-31"], monkeypatch)["RT-M-31"]
    assert result.outcome == "warn" and "SUCCESS but took 1." in result.message
    assert _run(build_dir, "slow_cleanup", ["RT-M-31"], monkeypatch)["RT-M-31"].outcome == "warn"  # the same on a second run: no hidden state
    monkeypatch.setattr(runtime_model, "SLOW_CLEANUP_MS", 10000)
    assert _run(build_dir, "slow_cleanup", ["RT-M-31"], monkeypatch)["RT-M-31"].outcome == "pass"


def test_one_slow_retrieve_zero_call_on_a_loaded_host_is_not_a_failure(build_dir, monkeypatch):
    """RT-M-13 judged ONE sample; it now takes the fastest of 5, so a single delayed call passes and a runtime that is always slow still fails."""
    assert _run(build_dir, "retrieve0_hiccup", ["RT-M-13"], monkeypatch)["RT-M-13"].outcome == "pass"
    slow = _run(build_dir, "retrieve0_blocks", ["RT-M-13"], monkeypatch)["RT-M-13"]
    assert slow.outcome == "fail" and "in each of 5 calls" in slow.message


def test_the_timeout_steps_report_the_fastest_and_the_slowest_sample(build_dir, monkeypatch):
    monkeypatch.setenv("OAAX_MOCK_FAULT", "retrieve0_hiccup")
    run = _scenario(build_dir, "timeouts", ["-i", "x:FLOAT:1,4", "-i", "scale:FLOAT:1"], timeout=20)
    assert run.step("retrieve_0")["ms"] < 100 < run.step("retrieve_0")["max_ms"]  # one call took 300 ms, the fastest did not
    assert run.step("retrieve_200")["max_ms"] >= 190


def test_an_init_slower_than_the_budget_blocks_the_checks_that_need_it(build_dir, monkeypatch):
    """RT-L-07 and RT-M-02 run init many times: a runtime that takes minutes to start is BLOCKED (raise --timeout), never FAILED."""
    results = _run(build_dir, "hang_init", ["RT-L-07", "RT-M-02"], monkeypatch, timeout=5)
    assert {i: r.outcome for i, r in results.items()} == {"RT-L-07": "blocked", "RT-M-02": "blocked"}


# ---- round 5, A4-A6: install instructions the user can follow --------------------------------------------------------------------------

def test_no_message_or_doc_tells_the_user_to_pip_install():
    """A6: the package is not on PyPI, so `pip install oaax-conformance` would install whatever squats that name, and uv tool environments have no pip."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    files = [*(root / "python").rglob("*.py"), root / "README.md", root / "CONTRIBUTING.md", root / "docs" / "design.md"]
    assert [str(f.relative_to(root)) for f in files if "pip install" in f.read_text()] == []


def test_the_readme_gives_the_python_the_wheel_needs():
    """A4: `uv tool install` of the wheel fails on a default Python 3.11 unless told `--python 3.12` (requires-python is >=3.12)."""
    from pathlib import Path

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert "uv tool install --python 3.12 oaax_conformance" in readme and "[e2e]" not in readme


# ---- round 5, A11: a PASS that rests on less than the specified work says so -----------------------------------------------------------------

def test_a_slow_init_is_not_a_pass_with_an_empty_message_after_a_few_cycles(build_dir, monkeypatch):
    """A11: with a slow init RT-L-07 (50 cycles), RT-L-10 and RT-M-35 passed after 3 cycles, silently; a leak of 1 MB per cycle passes 3 cycles."""
    results = _run(build_dir, "slow_init", ["RT-L-07", "RT-L-10", "RT-M-35"], monkeypatch, timeout=20)
    assert results["RT-L-07"].outcome == "pass" and re.search(r"only \d+ of 50 init/cleanup cycles ran \(the time budget ran out: raise --timeout\)", results["RT-L-07"].note)
    for check in ("RT-L-10", "RT-M-35"):
        assert results[check].outcome == "blocked" and "at least 10" in results[check].message and "--timeout" in results[check].message


def test_hardware_safe_cycles_are_blocked_for_growth_and_noted_for_repeatability(build_dir):
    ctx = helpers.context(build_dir)
    ctx.hardware_safe()
    results = helpers.run(ctx, ["RT-L-07", "RT-L-10"])
    assert results["RT-L-07"].outcome == "pass" and results["RT-L-07"].note == "only 3 of 50 init/cleanup cycles ran (--hardware-safe runs few cycles by design)"
    assert results["RT-L-10"].outcome == "blocked" and "--hardware-safe" in results["RT-L-10"].message


def test_a_note_is_printed_on_the_console_and_kept_in_junit(tmp_path):
    from dataclasses import asdict

    r = report.Result("A", "t", "MUST", "s", "pass", note="only 3 of 50 cycles ran")
    text = report.console([r], 0)
    assert "Details" in text and "A [MUST] PASS" in text and "note: only 3 of 50 cycles ran" in text
    path = tmp_path / "junit.xml"
    report.write_junit(str(path), {"spec_version": "2.0", "results": [asdict(r)]})
    assert "NOTE [MUST] only 3 of 50 cycles ran" in path.read_text()


# ---- round 5, A9 ------------------------------------------------------------------------------------------------------------------------------

def _cli_run(build_dir, tmp_path, *extra):
    out = tmp_path / "out"
    argv = ["runtime", str(build_dir / "libmock_runtime.so"), "--harness", str(build_dir / "oaax-harness"), "--model", str(build_dir / "mock.model"), "--quiet", "--pause", "0",
            "--out", str(out), *extra]
    return cli.main(argv), out


def test_a_secret_that_is_a_common_word_does_not_rewrite_the_specification_text(build_dir, tmp_path):
    """A9c: scrub() ran over the whole document, so --init-config device=load turned "two models loaded in one call" into "two models ***ed" in the permanent record."""
    code, out = _cli_run(build_dir, tmp_path, "--init-config", "vendor_key=1", "--init-config", "device=load", "--only", "RT-M-02")
    assert code == 0
    result = json.loads((out / "report.json").read_text())["results"][0]
    assert "load_models" in result["pass_when"] and "***" not in result["pass_when"] + result["source"] + result["title"]


def test_a_limited_selection_says_so_everywhere(build_dir, tmp_path, capsys):
    """A9d: `--only X --only Y` printed "Result: no MUST check failed" and recorded nothing about the 60 checks that did not run."""
    code, out = _cli_run(build_dir, tmp_path, "--init-config", "vendor_key=1", "--only", "RT-M-33", "--only", "RT-M-08")
    assert code == 0
    assert re.search(r"Only 2 of \d+ approved checks were selected", capsys.readouterr().out)
    doc = json.loads((out / "report.json").read_text())
    assert doc["summary"]["checks_selected"] == 2 and doc["summary"]["checks_approved"] > 50
    prop = ET.parse(out / "junit.xml").getroot().find("properties/property")
    assert prop.get("name") == "checks_selected" and prop.get("value").startswith("2 of ")


def test_a_full_run_does_not_claim_to_be_partial(build_dir, tmp_path, capsys):
    _cli_run(build_dir, tmp_path, "--init-config", "vendor_key=1", "--only", "RT-S-01")
    assert "approved checks were selected" in capsys.readouterr().out
    results = helpers.run(helpers.context(build_dir))
    assert "approved checks were selected" not in report.console(list(results.values()), 0, selected=(len(results), len(results)))


def test_a_tmpdir_docker_cannot_mount_is_refused_naming_the_cause(tmp_path, monkeypatch, capsys):
    """A9e: with ':' in TMPDIR every `docker run -v` failed and the message blamed the Docker setup and --docker-arg."""
    import tempfile

    bad = tmp_path / "a:b"
    bad.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(bad))
    assert cli.main(["toolchain", "x"]) == 3
    assert "contains ':'" in capsys.readouterr().err

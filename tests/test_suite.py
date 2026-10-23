"""The suite tests itself: the clean mock runtime passes everything, and every check fails on its own fault."""
import pytest

import json
import re
from pathlib import Path

from oaax_conformance import cli, html_report, report
from oaax_conformance.harness import STATUS_NAMES
from oaax_conformance.registry import CHECKS
from oaax_conformance.specdocs import load
from oaax_conformance.suite import Context, run_suite

# (OAAX_MOCK_FAULT, mock library variant, the test that must catch it)
CASES = [
    (None, "mock_missing_export", "RT-S-01"),
    (None, "mock_extra_export", "RT-S-02"),
    ("bad_version", "mock_runtime", "RT-S-03"),
    ("null_name", "mock_runtime", "RT-S-04"),
    ("unstable_getters", "mock_runtime", "RT-S-05"),
    ("info_before_init", "mock_runtime", "RT-S-06"),
    ("error_at_start", "mock_runtime", "RT-S-07"),
    ("cleanup_not_idempotent", "mock_runtime", "RT-S-08"),
    ("preinit_success", "mock_runtime", "RT-S-09"),
    ("crash_preinit", "mock_runtime", "RT-S-09"),
    ("reject_unknown_key", "mock_runtime", "RT-L-01"),
    ("require_config", "mock_runtime", "RT-L-02"),
    ("init_twice_ok", "mock_runtime", "RT-L-03"),
    ("info_not_json", "mock_runtime", "RT-L-04"),
    ("no_error_text", "mock_runtime", "RT-L-05"),
    ("info_after_cleanup", "mock_runtime", "RT-L-06"),
    ("no_reinit", "mock_runtime", "RT-L-07"),
    ("leak_thread", "mock_runtime", "RT-L-08"),
    ("pin_library", "mock_runtime", "RT-L-09"),
    ("leak_memory", "mock_runtime", "RT-L-10"),
    ("crash_at_exit", "mock_runtime", "RT-L-11"),
    ("postcleanup_success", "mock_runtime", "RT-L-12"),
    ("load_error_generic", "mock_runtime", "RT-L-13"),
    ("load_error_generic", "mock_runtime", "RT-L-14"),
    ("stale_error", "mock_runtime", "RT-L-15"),
    ("null_ptr_ignored", "mock_runtime", "RT-L-16"),
    ("null_ptr_crash", "mock_runtime", "RT-L-16"),
    ("load_twice_ok", "mock_runtime", "RT-M-01"),
    ("load_sticky_failure", "mock_runtime", "RT-M-02"),
    ("no_memory_load", "mock_runtime", "RT-M-03"),
    ("retain_buffer", "mock_runtime", "RT-M-04"),
    ("wrong_model_id", "mock_runtime", "RT-M-05"),
    ("accept_bad_id", "mock_runtime", "RT-M-06"),
    ("bad_id_generic", "mock_runtime", "RT-M-07"),
    ("no_id_echo", "mock_runtime", "RT-M-08"),
    ("output_null_name", "mock_runtime", "RT-M-09"),
    ("output_aliased", "mock_runtime", "RT-M-10"),
    ("bad_data_size", "mock_runtime", "RT-M-11"),
    ("rename_output", "mock_runtime", "RT-M-12"),
    ("retrieve0_blocks", "mock_runtime", "RT-M-13"),
    ("retrieve_timeout_long", "mock_runtime", "RT-M-14"),
    ("retrieve_returns_early", "mock_runtime", "RT-M-15"),
    ("negative_no_wait", "mock_runtime", "RT-M-16"),
    ("enqueue_blocks", "mock_runtime", "RT-M-17"),
    ("null_tensors_crash", "mock_runtime", "RT-M-18"),
    ("accept_empty", "mock_runtime", "RT-M-19"),
    ("accept_missing", "mock_runtime", "RT-M-20"),
    ("accept_wrong_dtype", "mock_runtime", "RT-M-21"),
    ("type_code_generic", "mock_runtime", "RT-M-22"),
    ("accept_wrong_shape", "mock_runtime", "RT-M-23"),
    ("shape_code_generic", "mock_runtime", "RT-M-24"),
    ("accept_bad_size", "mock_runtime", "RT-M-25"),
    ("accept_unknown_name", "mock_runtime", "RT-M-26"),
    ("free_rejected", "mock_runtime", "RT-M-27"),
    ("break_after_reject", "mock_runtime", "RT-M-28"),
    ("concurrency_drop", "mock_runtime", "RT-M-29"),
    ("corrupt_every_7th", "mock_runtime", "RT-M-30"),
    ("cleanup_fails_inflight", "mock_runtime", "RT-M-31"),
    ("no_wakeup", "mock_runtime", "RT-M-32"),
    ("flood_lost", "mock_runtime", "RT-M-33"),
    ("info_stale", "mock_runtime", "RT-M-34"),
    ("leak_per_load", "mock_runtime", "RT-M-35"),
    ("leak_per_request", "mock_runtime", "RT-M-36"),
]


def _context(build_dir, variant, init_config=("vendor_key=1",), model=True):
    """Small load sizes: the self-tests run every fault, and the mock needs 5 ms per request."""
    return Context(str(build_dir / f"lib{variant}.so"), str(build_dir / "oaax-harness"), list(init_config), str(build_dir / "mock.model") if model else None,
                   settle_ms=500, timeout=60, cycles=5, flood=100, requests=60, producers=3, per_producer=10, model_cycles=3, lost_after_ms=400, retrieve_timeout_ms=1500)


def _run(build_dir, variant, init_config=("vendor_key=1",)):
    return {r.id: r for r in run_suite(_context(build_dir, variant, init_config)).results}


def test_clean_mock_passes_everything(build_dir):
    results = _run(build_dir, "mock_runtime")
    assert {i: r.outcome for i, r in results.items() if r.outcome != "pass"} == {}


@pytest.mark.parametrize("fault,variant,test_id", CASES, ids=[f"{c[2]}-{c[0] or c[1]}" for c in CASES])
def test_fault_is_caught(build_dir, monkeypatch, fault, variant, test_id):
    if fault:
        monkeypatch.setenv("OAAX_MOCK_FAULT", fault)
    results = _run(build_dir, variant)
    target = results[test_id]
    assert target.outcome == ("fail" if target.level == "MUST" else "warn"), target
    assert not [r for r in results.values() if r.outcome == "error"], "a check crashed instead of judging"


def test_version_node_is_not_an_extra_export(build_dir):
    assert _run(build_dir, "mock_versioned")["RT-S-02"].outcome == "pass"


def test_every_approved_test_is_implemented_and_vice_versa():
    approved = {i for i, s in load().items() if s.status == "approved"}
    assert approved == set(CHECKS)


def test_every_check_has_a_fault_case():
    assert {c[2] for c in CASES} == set(CHECKS)


def test_exit_codes(build_dir, monkeypatch, tmp_path):
    lib = str(build_dir / "libmock_runtime.so")
    base = ["runtime", lib, "--harness", str(build_dir / "oaax-harness"), "--model", str(build_dir / "mock.model"), "--hardware-safe", "--pause", "0",
            "--init-config", "vendor_key=1"]
    assert cli.main(base) == 0
    monkeypatch.setenv("OAAX_MOCK_FAULT", "bad_version")
    assert cli.main(base + ["--out", str(tmp_path / "out")]) == 1
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"report.html", "report.json", "junit.xml"}
    monkeypatch.setenv("OAAX_MOCK_FAULT", "require_config")  # init needs a key and none is given: lifecycle MUSTs cannot run
    assert cli.main(base[:-2]) == 2  # without the vendor key init fails: the MUST checks cannot run


def test_model_tests_are_blocked_without_a_model(build_dir):
    result = {r.id: r for r in run_suite(_context(build_dir, "mock_runtime", model=False)).results}
    assert result["RT-M-01"].outcome == "blocked" and "--model" in result["RT-M-01"].message
    assert {r.outcome for i, r in result.items() if i.startswith("RT-M")} == {"blocked"}


def test_a_model_the_runtime_cannot_load_blocks_instead_of_failing(build_dir, tmp_path):
    bad = tmp_path / "bad.model"
    bad.write_bytes(b"not a mock model")
    ctx = _context(build_dir, "mock_runtime")
    ctx.model = str(bad)
    results = {r.id: r for r in run_suite(ctx).results}
    assert results["RT-M-01"].outcome == "blocked" and "could not load the model" in results["RT-M-01"].message
    assert results["RT-M-08"].outcome == "blocked"  # no manifest next to this model: inputs unknown


def test_exit_code_function():
    r = lambda outcome, level="MUST": report.Result("X", "t", level, "s", outcome)  # noqa: E731
    assert report.exit_code([r("pass")]) == 0
    assert report.exit_code([r("warn", "SHOULD"), r("blocked", "SHOULD")]) == 0
    assert report.exit_code([r("blocked")]) == 2
    assert report.exit_code([r("fail"), r("blocked")]) == 1


def test_console_is_a_table():
    results = [
        report.Result("RT-S-01", "Exports", "MUST", "header", "pass"),
        report.Result("RT-S-03", "Version `string`", "MUST", "header: SemVer", "fail", "'v1' is not a SemVer version"),
        report.Result("RT-L-08", "Threads", "SHOULD", "derived", "blocked", "init failed"),
    ]
    lines = report.console(results, 0).splitlines()
    assert lines[0].split() == ["ID", "Check", "Level", "Status"]
    assert lines[2].split() == ["RT-S-01", "Exports", "MUST", "PASS"]
    assert lines[3].split() == ["RT-S-03", "Version", "string", "MUST", "FAIL"]  # backticks are dropped
    assert "RT-S-03 [MUST] FAIL: 'v1' is not a SemVer version" in "\n".join(lines)
    assert "\033" not in "\n".join(lines)  # no colour unless asked


def test_status_names_match_the_pinned_header():
    header = (Path(__file__).resolve().parents[1] / "spec" / "2.0" / "interface.h").read_text()
    enum = header[header.index("typedef enum RuntimeStatus"):header.index("} RuntimeStatus")]
    pairs = re.findall(r"RUNTIME_STATUS_(\w+) = (\d+)", enum)
    assert [name for name, _ in sorted(pairs, key=lambda p: int(p[1]))] == STATUS_NAMES


def test_report_document_and_html(build_dir, monkeypatch):
    monkeypatch.setenv("OAAX_MOCK_FAULT", "bad_version")
    lib = str(build_dir / "libmock_runtime.so")
    ctx = Context(lib, str(build_dir / "oaax-harness"), ["secret_key=hunter2"], settle_ms=500, timeout=30)
    doc = report.build(run_suite(ctx), lib, "2.0", ["secret_key=hunter2"], [])
    text = json.dumps(doc)
    assert "hunter2" not in text and doc["options"]["init_config_keys"] == ["secret_key"]  # values are never stored
    assert doc["verdict"]["exit_code"] == 1 and doc["library"]["runtime_name"] == "OAAX Mock Runtime"
    assert "path" not in doc["library"] and str(build_dir) not in text  # no directories in a report that gets shared
    assert doc["summary"]["counts"]["fail"] == 1 and doc["summary"]["by_level"]["MUST"]["fail"] == 1
    calls = {o["call"]: o for o in doc["scenarios"]["static"]["observations"]}
    assert "value" in calls["get_error_fresh"] and calls["get_error_fresh"]["value"] is None  # NULL is shown, not left blank
    assert calls["version"]["value"] == "v1" and calls["preinit_load_models"]["status_name"] == "NOT_INITIALIZED"
    failing = next(r for r in doc["results"] if r["outcome"] == "fail")
    assert failing["id"] == "RT-S-03" and failing["scenarios"] == ["static"] and failing["pass_when"]
    page = html_report.render(doc)
    assert "RT-S-03" in page and "FAILED" in page and "NOT_INITIALIZED" in page


def test_html_escapes_everything_it_shows():
    doc = {"suite_version": "x", "spec_version": "2.0", "generated_at": "t", "platform": "p",
           "verdict": {"exit_code": 1, "text": "FAILED", "note": "n"},
           "library": {"name": "<img src=x onerror=alert(1)>", "sha256": "0", "runtime_name": "<script>", "runtime_version": None},
           "options": {"init_config_keys": ["<k>"], "ignore_threads": []},
           "summary": {"total": 1, "counts": {"pass": 0, "warn": 0, "fail": 1, "error": 0, "blocked": 0, "skipped": 0},
                       "by_level": {"MUST": {}}, "draft_tests_not_run": 0},
           "results": [{"id": "RT-X-01", "title": "<i>t</i>", "level": "MUST", "source": "<s>", "outcome": "fail", "message": "<script>alert(1)</script>",
                       "pass_when": "<p>", "scenarios": ["static"]}],
           "scenarios": {"static": {"returncode": 0, "signal": None, "timed_out": False, "duration_ms": 1,
                                    "observations": [{"call": "<c>", "status": 0, "status_name": "SUCCESS", "value": "<script>", "extra": "<e>"}]}}}
    page = html_report.render(doc)
    assert "<script>alert" not in page and "<img src=x" not in page and "<c>" not in page


def test_report_redacts_directories_runtimes_echo_in_messages():
    doc = {"results": [{"message": "Failed to extract model archive: /home/someone/models/model.onnx"}], "library": {"name": "x"}}
    clean = report.redact(doc, ["/home/someone/models"])
    assert clean["results"][0]["message"] == "Failed to extract model archive: model.onnx"
    assert report.redact(doc, ["/"])["results"][0]["message"] == doc["results"][0]["message"]  # never redact the root


def test_only_does_not_count_unselected_tests_as_drafts(build_dir):
    ctx = Context(str(build_dir / "libmock_runtime.so"), str(build_dir / "oaax-harness"), ["vendor_key=1"], settle_ms=500, timeout=30)
    suite = run_suite(ctx, ["RT-S-01"])
    drafts = sum(spec.status != "approved" for spec in load().values())
    assert len(suite.results) == 1 and suite.not_approved == drafts  # unselected approved tests are not drafts

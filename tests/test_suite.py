"""The suite tests itself: the clean mock runtime passes everything, and every check fails on its own fault."""
import pytest

from oaax_conformance import cli, report
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
]


def _run(build_dir, variant, init_config=("vendor_key=1",)):
    ctx = Context(str(build_dir / f"lib{variant}.so"), str(build_dir / "oaax-harness"), list(init_config), settle_ms=500, timeout=30)
    return {r.id: r for r in run_suite(ctx)[0]}


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
    base = ["runtime", lib, "--harness", str(build_dir / "oaax-harness"), "--init-config", "vendor_key=1"]
    assert cli.main(base) == 0
    monkeypatch.setenv("OAAX_MOCK_FAULT", "bad_version")
    assert cli.main(base + ["--report", str(tmp_path / "r.json"), "--junit", str(tmp_path / "r.xml")]) == 1
    assert (tmp_path / "r.json").stat().st_size > 0 and (tmp_path / "r.xml").stat().st_size > 0
    monkeypatch.setenv("OAAX_MOCK_FAULT", "require_config")  # init needs a key and none is given: lifecycle MUSTs cannot run
    assert cli.main(base[:-2]) == 2


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

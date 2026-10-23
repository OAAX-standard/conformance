"""RT-L: init/cleanup rules (init must succeed). Specs: docs/tests/runtime-static-lifecycle.md"""
import json
from collections import Counter

from ..registry import Blocked, Fail, check

SUCCESS, ALREADY_INITIALIZED = 0, 3
MAX_RSS_GROWTH = 5 * 1024 * 1024


def _status(run, step: str) -> int:
    return run.step(step)["status"]


@check("RT-L-01")
def unknown_key(ctx):
    status = _status(ctx.run("lifecycle"), "init_unknown_key")
    if status != SUCCESS:
        raise Fail(f"init with an unknown key returned {status}")


@check("RT-L-02")
def empty_config(ctx):
    status = _status(ctx.run("lifecycle"), "init_empty")
    if status != SUCCESS:
        raise Fail(f"init with an empty config returned {status}: the runtime needs vendor keys to start")


@check("RT-L-03")
def second_init(ctx):
    run = ctx.run("lifecycle")
    status = _status(run, "init_second")
    if status != ALREADY_INITIALIZED:
        raise Fail(f"second init returned {status}, expected ALREADY_INITIALIZED (3)")
    if run.raw("info_after_second_init") is None:
        raise Fail("the runtime stopped answering after the rejected second init (get_info is NULL)")


@check("RT-L-04")
def info_is_json(ctx):
    raw = ctx.run("lifecycle").raw("info")
    if raw is None:
        raise Fail("runtime_get_info() returned NULL after init")
    try:
        info = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise Fail(f"runtime_get_info() is not valid JSON: {e}")
    if not isinstance(info, dict):
        raise Fail(f"runtime_get_info() is a JSON {type(info).__name__}, expected an object")


@check("RT-L-05")
def error_text(ctx):
    run = ctx.run("lifecycle")
    if _status(run, "init_second") == SUCCESS:
        raise Blocked("the second init did not fail (see RT-L-03)")
    if not (run.step("init_second").get("err") or ""):
        raise Fail("runtime_get_error() is NULL or empty after a failed call")


@check("RT-L-06")
def info_after_cleanup(ctx):
    if ctx.run("lifecycle").raw("info_after_cleanup") is not None:
        raise Fail("runtime_get_info() is not NULL after cleanup")


@check("RT-L-07")
def reinit_cycles(ctx):
    cycles = ctx.run("cycles").step("cycles")
    if cycles["init_failures"] or cycles["cleanup_failures"]:
        raise Fail(cycles["first_failure"])


@check("RT-L-08")
def threads_after_cleanup(ctx):
    run = ctx.run("lifecycle")
    before, after = Counter(run.step("threads_baseline")["threads"]), Counter(run.step("threads_after_cleanup")["threads"])
    left = {n: c for n, c in (after - before).items() if n not in ctx.ignore_threads}
    if left:
        raise Fail(f"{sum(left.values())} thread(s) still running after cleanup: {', '.join(sorted(left))} (excuse driver threads with --ignore-thread)")


@check("RT-L-09")
def library_unloads(ctx):
    if ctx.run("unload").step("dlclose")["mapped"]:
        raise Fail("the library is still mapped after cleanup + dlclose (pinned by NODELETE/GNU_UNIQUE, or a thread still runs)")


@check("RT-L-10")
def no_memory_growth(ctx):
    cycles = ctx.run("cycles").step("cycles")
    if cycles["rss_first"] < 0 or cycles["rss_last"] < 0:
        raise Blocked("RSS not readable on this platform")
    growth = cycles["rss_last"] - cycles["rss_first"]
    if growth > MAX_RSS_GROWTH:
        raise Fail(f"RSS grew {growth / 1048576:.1f} MB over {cycles['cycles']} init/cleanup cycles (limit {MAX_RSS_GROWTH // 1048576} MB)")


@check("RT-L-11")
def exit_without_cleanup(ctx):
    run = ctx.run("exit-without-cleanup")
    run.step("exiting")
    if run.timed_out:
        raise Fail("the process hung at exit")
    if run.returncode != 0:
        raise Fail(f"the process exited abnormally ({run.signal_name or 'exit code ' + str(run.returncode)}) without cleanup")

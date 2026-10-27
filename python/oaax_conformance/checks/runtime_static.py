"""RT-S: the library alone, no init, no device. Specs: docs/tests/runtime-static-lifecycle.md"""
import re
import shutil
import subprocess

from ..registry import Blocked, Fail, check

NOT_INITIALIZED = 2
SYMBOLS = ["init", "load_models", "enqueue_input", "retrieve_output", "cleanup", "get_error", "get_version", "get_name", "get_info"]
LINKER_SYMBOLS = {"_init", "_fini", "_edata", "_end", "__bss_start"}  # added by the toolchain, not by the runtime
SEMVER = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?")


def _text(raw: bytes, what: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise Fail(f"{what} is not valid UTF-8")


@check("RT-S-01")
def exports(ctx):
    missing = ctx.run("static").step("symbols")["missing"]
    if missing:
        raise Fail("missing exports: " + ", ".join(missing))


@check("RT-S-02")
def only_the_nine_exports(ctx):
    nm, readelf = shutil.which("nm"), shutil.which("readelf")
    if not nm or not readelf:
        raise Blocked("nm/readelf not found")
    with open(ctx.lib, "rb") as f:
        if f.read(4) != b"\x7fELF":
            raise Blocked("the library is not an ELF file: its exports cannot be listed here")
    nm_run = subprocess.run([nm, "-D", "--defined-only", ctx.lib], capture_output=True, text=True)
    if nm_run.returncode != 0:  # an empty listing is not "no extra symbols": a library nm cannot read has not been judged
        raise Blocked(f"nm could not read the library ({(nm_run.stderr.strip().splitlines() or ['no message'])[-1][:150]}): its exports cannot be listed")
    listing = nm_run.stdout
    # "A" = absolute: the version node of a linker version script (e.g. OAAX_V2), not an exported function
    exported = {parts[-1].split("@")[0] for parts in map(str.split, listing.splitlines()) if len(parts) == 3 and parts[1] != "A"}
    extra = sorted(exported - {"runtime_" + s for s in SYMBOLS} - LINKER_SYMBOLS)
    readelf_run = subprocess.run([readelf, "--dyn-syms", "-W", ctx.lib], capture_output=True, text=True)
    if readelf_run.returncode != 0:
        raise Blocked(f"readelf could not read the library ({(readelf_run.stderr.strip().splitlines() or ['no message'])[-1][:150]}): its symbols cannot be listed")
    unique = sum(" UNIQUE " in line for line in readelf_run.stdout.splitlines())
    problems = []
    if extra:
        problems.append(f"{len(extra)} extra exported symbol(s), e.g. {', '.join(extra[:5])}")
    if unique:
        problems.append(f"{unique} GNU_UNIQUE symbol(s): shared by every copy loaded in the process")
    if problems:
        raise Fail("; ".join(problems))


@check("RT-S-03")
def version_string(ctx):
    raw = ctx.run("static").raw("version")
    if raw is None:
        raise Fail("runtime_get_version() returned NULL")
    version = _text(raw, "the version")
    if not SEMVER.fullmatch(version):
        raise Fail(f"{version!r} is not a SemVer version")


@check("RT-S-04")
def name_string(ctx):
    raw = ctx.run("static").raw("name")
    if raw is None:
        raise Fail("runtime_get_name() returned NULL")
    if not _text(raw, "the name").strip():
        raise Fail("runtime_get_name() returned an empty string")


@check("RT-S-05")
def getters_are_stable(ctx):
    run = ctx.run("static")
    changed = [n for n in ("version", "name") if run.raw(n) != run.raw(n + "_again")]
    if changed:
        raise Fail("changed between two calls: " + ", ".join(changed))


@check("RT-S-06")
def info_before_init(ctx):
    if ctx.run("static").raw("info_before_init") is not None:
        raise Fail("runtime_get_info() is not NULL before init")


@check("RT-S-07")
def no_error_yet(ctx):
    if ctx.run("static").raw("get_error_fresh") is not None:
        raise Fail("runtime_get_error() is not NULL although no call failed")


@check("RT-S-08")
def cleanup_before_init(ctx):
    run = ctx.run("static")
    bad = [f"call {i}: {run.step(n)['status']}" for i, n in enumerate(("cleanup_before_init", "cleanup_before_init_again"), 1) if run.step(n)["status"] != 0]
    if bad:
        raise Fail("runtime_cleanup() before init did not return SUCCESS (" + "; ".join(bad) + ")")


@check("RT-S-09")
def calls_before_init(ctx):
    run = ctx.run("static")
    bad = []
    for name in ("preinit_load_models", "preinit_enqueue_input", "preinit_retrieve_output"):
        status = run.step(name)["status"]
        if status != NOT_INITIALIZED:
            bad.append(f"{name.removeprefix('preinit_')} returned {status}")
    if bad:
        raise Fail("expected NOT_INITIALIZED (2): " + "; ".join(bad))

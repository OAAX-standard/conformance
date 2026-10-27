"""oaax-conformance runtime|toolchain ...

Exit codes: 0 no MUST check failed; 1 a MUST check failed; 2 nothing failed but a MUST check could not run or was skipped (INCOMPLETE);
3 bad command line, missing file/Docker/image; 4 a check of the suite itself crashed (a bug of the suite); 130 / 143 / 129 stopped by Ctrl-C / SIGTERM / SIGHUP."""
import argparse
import os
import signal
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

from . import __version__, html_report, report
from .harness import find_harness
from .modelio import NoModelIO, check_inputs, parse_arg
from .specdocs import load
from .suite import Context, run_suite
from .toolchain import inspect

SPEC_VERSION = "2.0"
EXIT_USAGE = 3
# Tests that abuse a runtime on purpose (NULL pointers, exit without cleanup, cleanup racing a blocked call, request floods). A fragile device may not
# survive them: --hardware-safe leaves them out.
ABUSIVE = {
    "RT-L-11": "the process exits without cleanup",
    "RT-L-16": "it passes NULL pointers",
    "RT-M-18": "it passes NULL tensors",
    "RT-M-32": "it runs cleanup while another thread waits in retrieve_output",
    "RT-M-33": "it floods the runtime with requests",
}
# docker options the suite sets itself: a user value would silently defeat the tests that vary them
FORBIDDEN_DOCKER_ARGS = ("--user", "-u", "--network", "--net", "--read-only", "--entrypoint", "--rm", "--name", "-v", "--volume", "--mount", "--privileged")


class Parser(argparse.ArgumentParser):
    def error(self, message: str):  # argparse exits with 2, which is the code of INCOMPLETE
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        sys.exit(EXIT_USAGE)


def usage_error(message: str) -> int:
    print(f"oaax-conformance: error: {message}", file=sys.stderr)
    return EXIT_USAGE


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--only", action="append", metavar="ID", help="run only this test id (repeatable)")
    p.add_argument("--skip", action="append", default=[], metavar="ID", help="do not run this test id (repeatable)")
    p.add_argument("--timeout", type=int, default=None, metavar="SECONDS", help="time allowed per scenario (runtime, default 300) or per conversion (toolchain, default 900); "
                   "loops stop at 60%% of it; a call still running at the limit is BLOCKED (a slow device is not a failing one), raise it for slow hardware")
    p.add_argument("--init-config", action="append", default=[], metavar="KEY=VALUE", help="runtime_init key (repeatable); values are never stored in reports")
    p.add_argument("--harness", help="path to oaax-harness")
    p.add_argument("--out", metavar="DIR", help="also write report.html (open it, or print it to PDF), report.json (everything) and junit.xml")
    p.add_argument("--quiet", action="store_true", help="do not print each scenario to stderr as it starts")


def build_parser() -> argparse.ArgumentParser:
    parser = Parser(prog="oaax-conformance", description="OAAX conformance suite")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=Parser)
    rt = sub.add_parser("runtime", help="check a runtime library (.so)")
    rt.add_argument("library")
    _add_common(rt)
    rt.add_argument("--model", metavar="FILE", help="a model file this runtime can load: enables the tests that need one")
    rt.add_argument("--input", action="append", default=[], metavar="NAME:DTYPE:DIMS", help="an input of the model (repeatable), when it cannot be read from an "
                    "oaax-manifest.json or an .onnx file, or when the runtime takes another data type or layout than the ONNX model; use ? for a dynamic dimension")
    rt.add_argument("--hardware-safe", action="store_true", help="gentle on shared or fragile devices: few cycles, small floods, a pause between scenarios, "
                    "and none of the deliberately abusive tests (" + ", ".join(ABUSIVE) + ")")
    rt.add_argument("--pause", type=float, default=None, metavar="SECONDS", help="wait between scenarios (default 0, or 3 with --hardware-safe)")
    rt.add_argument("--round-trip-timeout", type=float, default=None, metavar="SECONDS", help="how long one request may take (default 10; scaled up automatically from the "
                    "first answered request, so only a device that is slow from the very first request needs this)")
    rt.add_argument("--workdir", metavar="DIR", help="run the harness in this directory (default: a fresh temporary one), for runtimes that need files next to them")
    rt.add_argument("--ignore-thread", action="append", default=[], metavar="NAME", help="driver thread name to excuse in RT-L-08")
    tc = sub.add_parser("toolchain", help="check a toolchain image (Docker), and with --runtime the model it produces")
    tc.add_argument("image", help="the toolchain image, e.g. oaax-cpu-toolchain:2.0.0")
    _add_common(tc)
    tc.add_argument("--onnx", metavar="FILE", help="a valid ONNX model this toolchain can convert (default: a small convolution model shipped with the suite)")
    tc.add_argument("--model-file", metavar="NAME", help="which file of the output directory is the model, if the toolchain writes several (default: the suite tries each "
                    "and uses the one the runtime loads); relative to the output directory")
    tc.add_argument("--runtime", metavar="LIB", help="the runtime that loads what the toolchain produces: enables the E2E tests")
    tc.add_argument("--docker-arg", action="append", default=[], metavar="ARG", help="an argument added to every `docker run` (repeatable), e.g. --docker-arg=--gpus=all; "
                    "the suite sets --user, --network, --read-only and the mounts itself")
    return parser


def _check_ids(ids: list[str] | None, prefixes: tuple, what: str) -> str | None:
    known = [i for i in load() if i.startswith(prefixes)]  # in test order
    unknown = sorted(set(ids or []) - set(known))
    return f"{what}: unknown test id(s) {', '.join(unknown)} (they run from {known[0]} to {known[-1]}, see docs/tests)" if unknown else None


def _validate(args) -> str | None:
    prefixes = ("RT-",) if args.command == "runtime" else ("TC-", "E2E-")
    for ids, what in ((args.only, "--only"), (args.skip, "--skip")):
        if (problem := _check_ids(ids, prefixes, what)):
            return problem
    bad = [p for p in args.init_config if "=" not in p]
    if bad:
        return f"--init-config {bad[0]!r}: expected KEY=VALUE"
    if args.out and Path(args.out).exists() and not Path(args.out).is_dir():
        return f"--out {args.out}: exists and is not a directory"
    if args.out:  # fail now, not after a run of minutes
        try:
            Path(args.out).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return f"--out {args.out}: cannot create it ({e.strerror})"
        if not os.access(args.out, os.W_OK):
            return f"--out {args.out}: not writable"
    if args.harness and not Path(args.harness).is_file():
        return f"--harness {args.harness}: no such file"
    if args.command == "runtime":
        for value, what in ((args.pause, "--pause"), (args.round_trip_timeout, "--round-trip-timeout")):
            if value is not None and not (0 <= value < 86400):  # nan and inf pass a plain comparison and crash later
                return f"{what} must be a number of seconds between 0 and 86400"
    if args.timeout is not None and args.timeout < 5:
        return "--timeout must be at least 5 seconds"
    if args.command == "runtime":
        if not Path(args.library).is_file():
            return f"{args.library}: no such library"
        if args.model and not Path(args.model).is_file():
            return f"--model {args.model}: no such file"
        if args.workdir and not Path(args.workdir).is_dir():
            return f"--workdir {args.workdir}: no such directory"
        try:
            check_inputs([parse_arg(text) for text in args.input])
        except NoModelIO as e:
            return str(e) if str(e).startswith("--input") else f"--input: {e}"
    else:
        if args.onnx and not Path(args.onnx).is_file():
            return f"--onnx {args.onnx}: no such file"
        if args.runtime and not Path(args.runtime).is_file():
            return f"--runtime {args.runtime}: no such library"
        if args.model_file and (Path(args.model_file).is_absolute() or ".." in Path(args.model_file).parts):
            return f"--model-file {args.model_file}: must be a path inside the output directory"
        if ":" in tempfile.gettempdir():  # every `docker run -v <workspace>:/input` would read the colon as a separator
            return f"TMPDIR {tempfile.gettempdir()} contains ':' which docker cannot mount: set TMPDIR to another directory"
        for arg in args.docker_arg:
            if arg.split("=", 1)[0] == "--env-file":  # its values cannot be known, so they could not be kept out of the reports
                return f"--docker-arg {arg}: pass variables as -e NAME=VALUE; the values of an env file cannot be kept out of the reports"
            if arg.split("=", 1)[0] in FORBIDDEN_DOCKER_ARGS:
                return f"--docker-arg {arg}: the suite sets this itself (a value here would change what the tests test)"
    return None


def _print_and_write(suite, args, title: str, document, paths: list[str], secrets: list[str]) -> int:
    if not suite.results:  # e.g. --only names only draft tests: "0 checks, exit 0" must never look like a pass
        return usage_error("no test ran: the selection matches no approved test (see docs/tests)")
    shown = report.scrub_results(suite.results, paths, secrets)
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    print(report.console(shown, suite.not_approved, color, title, (suite.selected, suite.approved)))
    if suite.leftovers:
        print("\nCould not remove (owned by another user, left alone): " + ", ".join(suite.leftovers))
    if args.out:
        doc = document()
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        report.write_json(str(out / "report.json"), doc)
        report.write_junit(str(out / "junit.xml"), doc)
        (out / "report.html").write_text(html_report.render(doc), encoding="utf-8")
        print(f"\nReports: {out / 'report.html'}  {out / 'report.json'}  {out / 'junit.xml'}")
    else:
        print("\nAdd --out DIR for the HTML report and the full JSON.")
    return report.exit_code(suite.results)


# What dlerror() says for a file that is not a shared library for this machine (a missing dependency says "cannot open shared object file" and is the suite's business).
NOT_A_LIBRARY = ("file too short", "invalid ELF header", "not a dynamic executable", "cannot dynamically load", "wrong ELF class", "ELF load command", "not an ELF")


def _not_a_library(ctx) -> str | None:
    run = ctx.run("static")  # RT-S-01's own scenario: it only loads the library and looks up the nine exports
    text = run.dlopen_error() if run.end == "dlopen" else ""
    if any(pattern in text for pattern in NOT_A_LIBRARY):
        return f"{ctx.lib} is not a loadable runtime library ({text.replace(f"{ctx.lib}: ", "")}): pass the libRuntimeLibrary.so built for this machine"
    return None


def _runtime(args) -> int:
    lib = str(Path(args.library).resolve())
    model = str(Path(args.model).resolve()) if args.model else None
    try:
        ctx = Context(lib, find_harness(args.harness), args.init_config, model, args.ignore_thread, args.timeout or 300, manual_inputs=args.input,
                      workdir=str(Path(args.workdir).resolve()) if args.workdir else None, progress=not args.quiet)
    except FileNotFoundError as e:
        return usage_error(str(e))
    problem = _not_a_library(ctx)
    if problem:  # 60 BLOCKED rows and a wrong hint would hide this one line
        return usage_error(problem)
    skip = {i: "skipped by request (--skip)" for i in args.skip}
    if args.hardware_safe:
        ctx.hardware_safe()
        skip.update({i: f"skipped by --hardware-safe: {why}" for i, why in ABUSIVE.items() if i not in skip})
    if args.pause is not None:
        ctx.pause = args.pause
    if args.round_trip_timeout is not None:
        ctx.retrieve_timeout_ms = int(args.round_trip_timeout * 1000)
    paths = [str(Path(lib).parent)] + ([str(Path(model).parent)] if model else []) + ([ctx.workdir] if ctx.workdir else [])
    secrets = report.secret_values(args.init_config, [])
    suite = run_suite(ctx, args.only, skip=skip)
    title = f"OAAX {SPEC_VERSION} runtime conformance: {Path(lib).name}\n"
    return _print_and_write(suite, args, title, lambda: report.build(suite, lib, SPEC_VERSION, args.init_config, args.ignore_thread, paths), paths, secrets)


def _toolchain(args) -> int:
    from .toolchain import ToolchainContext, default_model, docker_available

    if not docker_available():
        return usage_error("Docker is not available: a toolchain is a Docker image, so the toolchain tests need `docker` on this machine")
    try:
        info = inspect(args.image)
        harness = find_harness(args.harness) if args.runtime else None
    except FileNotFoundError as e:
        return usage_error(str(e))
    onnx = str(Path(args.onnx).resolve()) if args.onnx else default_model()
    runtime = str(Path(args.runtime).resolve()) if args.runtime else None
    ctx = ToolchainContext(args.image, onnx, args.docker_arg, args.timeout or 900, runtime, harness, args.init_config)
    ctx.progress = not args.quiet
    ctx.model_file = args.model_file
    skip = {i: "skipped by request (--skip)" for i in args.skip}
    try:
        suite = run_suite(ctx, args.only, prefixes=("TC-", "E2E-"), skip=skip)
    finally:
        ctx.close()
    suite.leftovers = ctx.leftovers
    paths = [str(Path(onnx).parent)] + ([str(Path(runtime).parent)] if runtime else [])
    secrets = report.secret_values(args.init_config, args.docker_arg)
    title = f"OAAX {SPEC_VERSION} toolchain conformance: {args.image}\n"
    return _print_and_write(suite, args, title, lambda: report.build_toolchain(suite, args.image, info, SPEC_VERSION, args.docker_arg, runtime, paths, args.init_config),
                            paths, secrets)


class Terminated(BaseException):
    """SIGTERM (a CI timeout, `timeout N`, `kill`) or SIGHUP (a closed terminal, a dropped ssh session): unwinds like Ctrl-C so the harness and the
    containers are stopped first. The exit code is 128 + the signal, as a shell would report it."""

    def __init__(self, signum: int):
        self.signum = signum


def _terminate(signum, frame):
    raise Terminated(signum)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    problem = _validate(args)
    if problem:
        return usage_error(problem)
    main_thread = threading.current_thread() is threading.main_thread()
    # `nohup oaax-conformance ...` ignores SIGHUP on purpose: keep it ignored
    caught = [signal.SIGTERM] + ([] if signal.getsignal(signal.SIGHUP) is signal.SIG_IGN else [signal.SIGHUP])
    previous = {s: signal.signal(s, _terminate) for s in caught} if main_thread else {}
    try:
        return _runtime(args) if args.command == "runtime" else _toolchain(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Terminated as e:
        print(f"\nterminated ({signal.Signals(e.signum).name})", file=sys.stderr)
        return 128 + e.signum
    except Exception as e:  # a bug of the suite must not look like "a MUST failed" (exit 1)
        print(f"internal error of the conformance suite ({type(e).__name__}: {e}); please report it with the command line you used", file=sys.stderr)
        return 4
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


if __name__ == "__main__":
    sys.exit(main())

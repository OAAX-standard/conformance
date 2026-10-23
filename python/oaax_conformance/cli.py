import argparse
import os
import sys
from pathlib import Path

from . import report
from .harness import find_harness
from .suite import Context, run_suite

SPEC_VERSION = "2.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oaax-conformance", description="OAAX conformance suite")
    sub = parser.add_subparsers(dest="command", required=True)
    rt = sub.add_parser("runtime", help="check a runtime library (.so)")
    rt.add_argument("library")
    rt.add_argument("--init-config", action="append", default=[], metavar="KEY=VALUE", help="runtime_init key (repeatable)")
    rt.add_argument("--ignore-thread", action="append", default=[], metavar="NAME", help="driver thread name to excuse in RT-L-08")
    rt.add_argument("--only", action="append", metavar="ID", help="run only this test id (repeatable)")
    rt.add_argument("--timeout", type=int, default=60, help="seconds per scenario")
    rt.add_argument("--harness", help="path to oaax-harness")
    rt.add_argument("--report", metavar="FILE.json")
    rt.add_argument("--junit", metavar="FILE.xml")
    args = parser.parse_args(argv)

    lib = str(Path(args.library).resolve())
    ctx = Context(lib, find_harness(args.harness), args.init_config, args.ignore_thread, args.timeout)
    results, not_approved = run_suite(ctx, args.only)
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    print(report.console(results, not_approved, color, f"OAAX {SPEC_VERSION} runtime conformance: {Path(lib).name}\n"))
    info = report.meta(lib, SPEC_VERSION)
    if args.report:
        report.write_json(args.report, results, info)
    if args.junit:
        report.write_junit(args.junit, results, info)
    return report.exit_code(results)


if __name__ == "__main__":
    sys.exit(main())

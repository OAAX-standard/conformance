import argparse
import os
import sys
from pathlib import Path

from . import html_report, report
from .harness import find_harness
from .suite import Context, run_suite

SPEC_VERSION = "2.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oaax-conformance", description="OAAX conformance suite")
    sub = parser.add_subparsers(dest="command", required=True)
    rt = sub.add_parser("runtime", help="check a runtime library (.so)")
    rt.add_argument("library")
    rt.add_argument("--init-config", action="append", default=[], metavar="KEY=VALUE", help="runtime_init key (repeatable)")
    rt.add_argument("--model", metavar="FILE", help="a model file this runtime can load: enables the tests that need one")
    rt.add_argument("--input", action="append", default=[], metavar="NAME:DTYPE:DIMS", help="an input of the model, when it cannot be read from an oaax-manifest.json or an .onnx file (repeatable)")
    rt.add_argument("--hardware-safe", action="store_true", help="gentle on shared or fragile devices: few cycles, small floods, a pause between scenarios")
    rt.add_argument("--pause", type=float, default=None, metavar="SECONDS", help="wait between scenarios (default 0, or 3 with --hardware-safe)")
    rt.add_argument("--ignore-thread", action="append", default=[], metavar="NAME", help="driver thread name to excuse in RT-L-08")
    rt.add_argument("--only", action="append", metavar="ID", help="run only this test id (repeatable)")
    rt.add_argument("--timeout", type=int, default=60, help="seconds per scenario")
    rt.add_argument("--harness", help="path to oaax-harness")
    rt.add_argument("--out", metavar="DIR", help="also write report.html (open it, or print it to PDF), report.json (everything) and junit.xml")
    args = parser.parse_args(argv)

    lib = str(Path(args.library).resolve())
    model = str(Path(args.model).resolve()) if args.model else None
    ctx = Context(lib, find_harness(args.harness), args.init_config, model, args.ignore_thread, args.timeout, manual_inputs=args.input)
    if args.hardware_safe:
        ctx.hardware_safe()
    if args.pause is not None:
        ctx.pause = args.pause
    suite = run_suite(ctx, args.only)
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    print(report.console(suite.results, suite.not_approved, color, f"OAAX {SPEC_VERSION} runtime conformance: {Path(lib).name}\n"))
    if args.out:
        doc = report.build(suite, lib, SPEC_VERSION, args.init_config, args.ignore_thread, [str(Path(model).parent)] if model else [])
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        report.write_json(str(out / "report.json"), doc)
        report.write_junit(str(out / "junit.xml"), doc)
        (out / "report.html").write_text(html_report.render(doc), encoding="utf-8")
        print(f"\nReports: {out / 'report.html'}  {out / 'report.json'}  {out / 'junit.xml'}")
    else:
        print("\nAdd --out DIR for the HTML report and the full JSON.")
    results = suite.results
    return report.exit_code(results)


if __name__ == "__main__":
    sys.exit(main())

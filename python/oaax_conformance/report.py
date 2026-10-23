"""Console, JSON and JUnit output."""
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from . import __version__


@dataclass
class Result:
    id: str
    title: str
    level: str
    source: str
    outcome: str  # pass | fail | warn | blocked | error | skipped
    message: str = ""


def exit_code(results: list[Result]) -> int:
    """1 = a MUST failed or a check errored; 2 = nothing failed but a MUST could not run; 0 = clean."""
    if any(r.outcome in ("fail", "error") for r in results):
        return 1
    return 2 if any(r.outcome in ("blocked", "skipped") and r.level == "MUST" for r in results) else 0


COLORS = {"pass": "32", "warn": "33", "fail": "31", "error": "31", "blocked": "36", "skipped": "2"}
MESSAGE_WIDTH = 160  # the JSON report keeps the full message


def _paint(outcome: str, text: str, color: bool) -> str:
    return f"\033[{COLORS[outcome]}m{text}\033[0m" if color else text


def console(results: list[Result], not_approved: int, color: bool = False, title: str = "") -> str:
    rows = [(r.id, r.title.replace("`", ""), r.level, r.outcome.upper()) for r in results]
    header = ("ID", "Check", "Level", "Status")
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(4)]
    line = lambda cells, outcome=None: "  ".join(  # noqa: E731
        (_paint(outcome, c.ljust(w), color) if outcome and i == 3 else c.ljust(w)) for i, (c, w) in enumerate(zip(cells, widths))
    ).rstrip()

    out = [title] if title else []
    out += [line(header), "  ".join("-" * w for w in widths)]
    out += [line(row, r.outcome) for row, r in zip(rows, results)]

    details = [r for r in results if r.outcome != "pass"]
    if details:
        out += ["", "Details"]
        for r in details:
            out.append(f"  {r.id} [{r.level}] {r.outcome.upper()}: {r.message[:MESSAGE_WIDTH]}{'...' if len(r.message) > MESSAGE_WIDTH else ''}")
            if r.outcome in ("fail", "warn"):
                out.append(f"      spec: {r.source}")

    counts = {o: sum(r.outcome == o for r in results) for o in COLORS}
    out += ["", ", ".join(f"{n} {o}" for o, n in counts.items() if n) + f"  ({not_approved} draft tests not run)"]
    verdict = {0: "no MUST check failed", 1: "FAILED", 2: "INCOMPLETE: some MUST checks could not run"}[exit_code(results)]
    out.append(f"Result: {verdict}. This run covers only the approved tests; it is not a full conformance claim.")
    return "\n".join(out)


def meta(lib: str, spec_version: str) -> dict:
    return {
        "suite_version": __version__,
        "spec_version": spec_version,
        "library": {"path": lib, "sha256": hashlib.sha256(Path(lib).read_bytes()).hexdigest()},
        "platform": platform.platform(),
    }


def write_json(path: str, results: list[Result], info: dict) -> None:
    Path(path).write_text(json.dumps({**info, "results": [asdict(r) for r in results]}, indent=2) + "\n")


def write_junit(path: str, results: list[Result], info: dict) -> None:
    suite = ET.Element("testsuite", name=f"oaax-conformance {info['spec_version']}", tests=str(len(results)))
    for r in results:
        case = ET.SubElement(suite, "testcase", classname=r.level, name=f"{r.id} {r.title}")
        if r.outcome in ("fail", "error"):
            ET.SubElement(case, "failure", message=r.message).text = f"spec: {r.source}"
        elif r.outcome in ("blocked", "skipped"):
            ET.SubElement(case, "skipped", message=r.message)
        elif r.outcome == "warn":
            ET.SubElement(case, "system-out").text = f"WARNING [{r.level}] {r.message}"
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)

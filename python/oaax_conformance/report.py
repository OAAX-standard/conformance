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


def console(results: list[Result], not_approved: int) -> str:
    lines = []
    for r in results:
        head = f"{r.outcome.upper():8} {r.id}  {r.title}"
        lines.append(head + (f"  [{r.level}]" if r.outcome != "pass" else ""))
        if r.message:
            lines.append(f"           {r.message}")
        if r.outcome in ("fail", "warn"):
            lines.append(f"           spec: {r.source}")
    counts = {o: sum(r.outcome == o for r in results) for o in ("pass", "warn", "fail", "blocked", "error", "skipped")}
    lines.append("")
    lines.append(", ".join(f"{n} {o}" for o, n in counts.items() if n) + f"  ({not_approved} draft tests not run)")
    verdict = {0: "no MUST check failed", 1: "FAILED", 2: "INCOMPLETE: some MUST checks could not run"}[exit_code(results)]
    lines.append(f"Result: {verdict}. This run covers only the approved tests; it is not a full conformance claim.")
    return "\n".join(lines)


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

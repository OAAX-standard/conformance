"""Console, JSON and JUnit output."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from . import __version__


@dataclass
class Result:
    id: str
    title: str
    level: str
    source: str
    outcome: str  # pass | fail | warn | blocked | error | skipped | na (not applicable)
    message: str = ""
    pass_when: str = ""
    scenarios: list = field(default_factory=list)  # scenarios the check read: its evidence


def exit_code(results: list[Result]) -> int:
    """1 = a MUST failed or a check errored; 2 = nothing failed but a MUST could not run; 0 = clean."""
    if any(r.outcome in ("fail", "error") for r in results):
        return 1
    return 2 if any(r.outcome in ("blocked", "skipped") and r.level == "MUST" for r in results) else 0


COLORS = {"pass": "32", "warn": "33", "fail": "31", "error": "31", "blocked": "36", "skipped": "2", "na": "2"}
MESSAGE_WIDTH = 160  # the JSON report keeps the full message


def _paint(outcome: str, text: str, color: bool) -> str:
    return f"\033[{COLORS[outcome]}m{text}\033[0m" if color else text


def console(results: list[Result], not_approved: int, color: bool = False, title: str = "") -> str:
    rows = [(r.id, r.title.replace("`", ""), r.level, "N/A" if r.outcome == "na" else r.outcome.upper()) for r in results]
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
    out += ["", ", ".join(f"{n} {'n/a' if o == 'na' else o}" for o, n in counts.items() if n) + f"  ({not_approved} draft tests not run)"]
    out.append(f"Result: {VERDICTS[exit_code(results)]}. {DISCLAIMER}")
    return "\n".join(out)


VERDICTS = {0: "no MUST check failed", 1: "FAILED", 2: "INCOMPLETE: some MUST checks could not run"}
DISCLAIMER = "This run covers only the approved tests; it is not a full conformance claim."


def redact(doc: dict, private_paths: list[str]) -> dict:
    """Runtimes put the paths they were given into their messages; a report that gets shared must not carry them."""
    text = json.dumps(doc, ensure_ascii=False)
    for path in sorted({p.rstrip("/") for p in private_paths if p and p != "/"}, key=len, reverse=True):
        text = text.replace(json.dumps(path)[1:-1] + "/", "")
    return json.loads(text)


def build(suite, lib: str, spec_version: str, init_config: list[str], ignore_threads: list[str], private_paths: list[str] = ()) -> dict:
    """The complete report: the JSON file is this document, the HTML page is rendered from it."""
    results = suite.results
    levels = sorted({r.level for r in results}, key=["MUST", "SHOULD", "PROPOSED"].index)
    static = suite.runs.get("static")
    identity = {}
    for key, step in (("runtime_name", "name"), ("runtime_version", "version")):
        try:
            raw = static.raw(step) if static else None
        except Exception:  # the static scenario may not have got that far; the report still has to be written
            raw = None
        identity[key] = raw.decode("utf-8", errors="replace") if raw else None
    code = exit_code(results)
    doc = {
        "suite_version": __version__,
        "spec_version": spec_version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": {"exit_code": code, "text": VERDICTS[code], "note": DISCLAIMER},
        "library": {"name": Path(lib).name,  # no directory: reports get shared
                     "sha256": hashlib.sha256(Path(lib).read_bytes()).hexdigest(), **identity},
        "platform": platform.platform(),
        # values are not stored: vendor keys can hold licences or paths
        "options": {"init_config_keys": [pair.split("=", 1)[0] for pair in init_config], "ignore_threads": ignore_threads},
        "summary": {
            "total": len(results),
            "counts": {o: sum(r.outcome == o for r in results) for o in COLORS},
            "by_level": {lv: {o: sum(r.outcome == o and r.level == lv for r in results) for o in COLORS} for lv in levels},
            "draft_tests_not_run": suite.not_approved,
        },
        "results": [asdict(r) for r in results],
        "scenarios": {
            name: {
                "returncode": run.returncode,
                "signal": run.signal_name,
                "timed_out": run.timed_out,
                "duration_ms": run.duration_ms,
                "observations": run.observations(),
            }
            for name, run in suite.runs.items()
        },
    }
    return redact(doc, [str(Path(lib).parent), *private_paths, str(Path.home())])


def write_json(path: str, doc: dict) -> None:
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


def write_junit(path: str, doc: dict) -> None:
    results = [Result(**{k: v for k, v in r.items()}) for r in doc["results"]]
    suite = ET.Element("testsuite", name=f"oaax-conformance {doc['spec_version']}", tests=str(len(results)))
    for r in results:
        case = ET.SubElement(suite, "testcase", classname=r.level, name=f"{r.id} {r.title}")
        if r.outcome in ("fail", "error"):
            ET.SubElement(case, "failure", message=r.message).text = f"spec: {r.source}"
        elif r.outcome in ("blocked", "skipped"):
            ET.SubElement(case, "skipped", message=r.message)
        elif r.outcome == "warn":
            ET.SubElement(case, "system-out").text = f"WARNING [{r.level}] {r.message}"
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)

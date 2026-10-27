"""Console, JSON and JUnit output."""
import hashlib
import json
import platform
import os
import re
import tempfile
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from xml.etree import ElementTree as ET

from . import __version__
from .dockerargs import env_options


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
    note: str = ""  # a pass that rests on less than the specified work says so (for example: only 3 of 50 cycles fit the time budget)


def exit_code(results: list[Result]) -> int:
    """0 = clean; 1 = a MUST failed; 2 = nothing failed but a MUST could not run or was skipped (--skip, --hardware-safe); 4 = a check of the suite itself crashed (3 is used by the CLI
    for a bad command line or a missing Docker). A failed MUST wins over a suite error: that verdict stands."""
    if any(r.outcome == "fail" for r in results):
        return 1
    if any(r.outcome == "error" for r in results):
        return 4
    return 2 if any(r.outcome in ("blocked", "skipped") and r.level == "MUST" for r in results) else 0


COLORS = {"pass": "32", "warn": "33", "fail": "31", "error": "31", "blocked": "36", "skipped": "2", "na": "2"}
MESSAGE_WIDTH = 160  # the JSON report keeps the full message


def _paint(outcome: str, text: str, color: bool) -> str:
    return f"\033[{COLORS[outcome]}m{text}\033[0m" if color else text


def console(results: list[Result], not_approved: int, color: bool = False, title: str = "", selected: tuple[int, int] | None = None) -> str:
    rows = [(r.id, r.title.replace("`", ""), r.level, "N/A" if r.outcome == "na" else r.outcome.upper()) for r in results]
    header = ("ID", "Check", "Level", "Status")
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(4)]
    line = lambda cells, outcome=None: "  ".join(  # noqa: E731
        (_paint(outcome, c.ljust(w), color) if outcome and i == 3 else c.ljust(w)) for i, (c, w) in enumerate(zip(cells, widths))
    ).rstrip()

    out = [title] if title else []
    out += [line(header), "  ".join("-" * w for w in widths)]
    out += [line(row, r.outcome) for row, r in zip(rows, results)]

    details = [r for r in results if r.outcome != "pass" or r.note]
    if details:
        out += ["", "Details"]
        for r in details:
            text = r.message if r.outcome != "pass" else ""
            out.append(f"  {r.id} [{r.level}] {r.outcome.upper()}: {text[:MESSAGE_WIDTH]}{'...' if len(text) > MESSAGE_WIDTH else ''}".rstrip(": "))
            if r.note:
                out.append(f"      note: {r.note[:MESSAGE_WIDTH]}")
            if r.outcome in ("fail", "warn"):
                out.append(f"      spec: {r.source}")

    counts = {o: sum(r.outcome == o for r in results) for o in COLORS}
    out += ["", ", ".join(f"{n} {'n/a' if o == 'na' else o}" for o, n in counts.items() if n) + f"  ({not_approved} draft tests not run)"]
    if selected and selected[0] < selected[1]:  # --only / a limited selection: the verdict below covers these checks only
        out.append(f"Only {selected[0]} of {selected[1]} approved checks were selected: the result below says nothing about the others.")
    out.append(f"Result: {VERDICTS[exit_code(results)]}. {DISCLAIMER}")
    return "\n".join(out)


VERDICTS = {0: "no MUST check failed", 1: "FAILED", 2: "INCOMPLETE: some MUST checks could not run or were skipped",
            4: "INTERNAL ERROR: a check of the suite crashed (a bug of the suite, not a verdict on the runtime)"}
DISCLAIMER = "This run covers only the approved tests; it is not a full conformance claim."


# Everything XML 1.0 cannot hold (C0, U+FFFE/FFFF, lone surrogates) plus what can hijack a terminal or reorder text (C1 incl. CSI, bidi overrides).
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069\ud800-\udfff\ufffe\uffff]")
_SCRATCH_DIR = re.compile(re.escape(tempfile.gettempdir()) + r"/(?:tmp\w{6,}|oaax-conformance-(?:dump-)?\w{6,})")  # the scenario's own working directory (no --workdir) and our workspaces
MAX_TEXT = 20000  # a runtime can return megabytes of error text


def clean_text(text: str, private_paths: list[str] = (), secrets: list[str] = ()) -> str:
    """What is safe to show and to share: no control characters (they break XML and terminals), no directories, no secret values.
    Runtimes put the paths and the configuration they were given into their own messages."""
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, "***")
    for path in sorted({p.rstrip("/") for p in private_paths if p and p.rstrip("/")}, key=len, reverse=True):
        text = text.replace(path + "/", "").replace(path, "<dir>")
    text = _SCRATCH_DIR.sub("<tmp>", text)
    if len(text) > MAX_TEXT:
        text = f"{text[:MAX_TEXT]}... [{len(text) - MAX_TEXT} more characters]"
    return _CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}" if ord(m.group()) < 256 else f"\\u{ord(m.group()):04x}", text)


def scrub(value, private_paths: list[str] = (), secrets: list[str] = ()):
    """clean_text on every string of a JSON-like structure (keys included)."""
    if isinstance(value, str):
        return clean_text(value, private_paths, secrets)
    if isinstance(value, list):
        return [scrub(v, private_paths, secrets) for v in value]
    if isinstance(value, dict):
        return {clean_text(k, private_paths, secrets): scrub(v, private_paths, secrets) for k, v in value.items()}
    return value


def scrub_results(results: list[Result], private_paths: list[str], secrets: list[str]) -> list[Result]:
    """The results as they are shown on the console: the same cleaning as the reports."""
    return [replace(r, message=clean_text(r.message, private_paths, secrets), note=clean_text(r.note, private_paths, secrets), title=clean_text(r.title)) for r in results]


def mask_docker_args(args: list[str]) -> list[str]:
    """Environment values are the likeliest secrets in docker arguments: show the variable, not the value. Other flags (--label, --add-host...) are shown as given."""
    out = list(args)
    for i, head, pair in env_options(args):
        name, sep, _ = pair.partition("=")
        if sep:
            out[i] = head + name + sep + "***"
    return out


def secret_values(init_config: list[str], docker_args: list[str]) -> list[str]:
    """Every value that must never appear in a report: --init-config values and the environment values docker passes to the container,
    including `-e NAME` with no value (docker takes it from this process's environment)."""
    values = [pair.split("=", 1)[1] for pair in init_config if "=" in pair]
    for _, _, pair in env_options(docker_args):
        name, sep, value = pair.partition("=")
        values.append(value if sep else os.environ.get(name, ""))
    return [v for v in values if len(v) >= 3]  # a one-character "secret" would shred every message


def _document(suite, spec_version: str, subject: dict, options: dict, private_paths: list[str], secrets: list[str]) -> dict:
    results = suite.results
    levels = sorted({r.level for r in results}, key=["MUST", "SHOULD", "PROPOSED"].index)
    code = exit_code(results)
    doc = {
        "suite_version": __version__,
        "spec_version": spec_version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": {"exit_code": code, "text": VERDICTS[code], "note": DISCLAIMER},
        "subject": subject,
        "platform": platform.platform(),
        "options": options,
        "summary": {
            "total": len(results),
            "counts": {o: sum(r.outcome == o for r in results) for o in COLORS},
            "by_level": {lv: {o: sum(r.outcome == o and r.level == lv for r in results) for o in COLORS} for lv in levels},
            "draft_tests_not_run": suite.not_approved,
            "checks_selected": suite.selected,
            "checks_approved": suite.approved,
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
    if suite.leftovers:
        doc["leftovers"] = suite.leftovers
    # Only what was observed is scrubbed: the specification's own text (titles, pass_when, source) must not be rewritten because a secret happens to be a common word.
    hide = [[*private_paths, str(Path.home())], secrets]
    for r in doc["results"]:
        r["message"], r["note"] = clean_text(r["message"], *hide), clean_text(r["note"], *hide)
    for key in ("scenarios", "subject", "options", "leftovers"):
        if key in doc:
            doc[key] = scrub(doc[key], *hide)
    return doc


def build(suite, lib: str, spec_version: str, init_config: list[str], ignore_threads: list[str], private_paths: list[str] = ()) -> dict:
    """The complete report of a runtime run: the JSON file is this document, the HTML page is rendered from it."""
    static = suite.runs.get("static")
    identity = {}
    for key, step in (("runtime_name", "name"), ("runtime_version", "version")):
        try:
            raw = static.raw(step) if static else None
        except Exception:  # the static scenario may not have got that far; the report still has to be written
            raw = None
        identity[key] = raw.decode("utf-8", errors="replace") if raw else None
    detail = " ".join(filter(None, [identity["runtime_name"], identity["runtime_version"]])) or "unknown runtime"
    subject = {"kind": "runtime", "name": Path(lib).name, "sha256": hashlib.sha256(Path(lib).read_bytes()).hexdigest(), "detail": detail, **identity}
    # values are not stored: vendor keys can hold licences or paths
    options = {"init_config_keys": [pair.split("=", 1)[0] for pair in init_config], "ignore_threads": ignore_threads}
    return _document(suite, spec_version, subject, options, [str(Path(lib).parent), *private_paths], secret_values(init_config, []))


def build_toolchain(suite, image: str, info: dict, spec_version: str, docker_args: list[str], runtime: str | None, private_paths: list[str] = (),
                    init_config: list[str] = ()) -> dict:
    """The complete report of a toolchain run."""
    subject = {"kind": "toolchain", "name": image, "sha256": info["id"].removeprefix("sha256:"),
               "detail": f"{info.get('os') or '?'}/{info.get('architecture') or '?'}"}
    options = {"docker_args": mask_docker_args(docker_args), "runtime": Path(runtime).name if runtime else None}
    return _document(suite, spec_version, subject, options, list(private_paths), secret_values(init_config, docker_args))


def write_json(path: str, doc: dict) -> None:
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


def write_junit(path: str, doc: dict) -> None:
    results = [Result(**{k: v for k, v in r.items()}) for r in doc["results"]]
    # a MUST that could not run or was skipped is INCOMPLETE (exit code 2): an <error>, so that a CI tool reading only this file does not show green
    incomplete = lambda r: r.outcome in ("blocked", "skipped") and r.level == "MUST"  # noqa: E731
    count = lambda pred: str(sum(1 for r in results if pred(r)))  # noqa: E731
    suite = ET.Element("testsuite", name=f"oaax-conformance {doc['spec_version']}", tests=str(len(results)), failures=count(lambda r: r.outcome == "fail"),
                       errors=count(lambda r: r.outcome == "error" or incomplete(r)),
                       skipped=count(lambda r: r.outcome in ("blocked", "skipped") and not incomplete(r)))
    selected = doc.get("summary", {})
    if selected.get("checks_selected") is not None:  # a limited selection must not look like a full run to a CI that reads only this file
        props = ET.SubElement(suite, "properties")
        ET.SubElement(props, "property", name="checks_selected", value=f"{selected['checks_selected']} of {selected['checks_approved']}")
    for r in results:
        case = ET.SubElement(suite, "testcase", classname=r.level, name=f"{r.id} {r.title}")
        if r.outcome == "fail":
            ET.SubElement(case, "failure", message=r.message).text = f"spec: {r.source}"
        elif r.outcome == "error" or incomplete(r):
            ET.SubElement(case, "error", type=r.outcome.upper() if r.outcome == "error" else "INCOMPLETE", message=r.message).text = f"spec: {r.source}"
        elif r.outcome in ("blocked", "skipped"):
            ET.SubElement(case, "skipped", message=r.message)
        elif r.outcome == "warn":
            ET.SubElement(case, "system-out").text = f"WARNING [{r.level}] {r.message}"
        if r.note:
            ET.SubElement(case, "system-out").text = f"NOTE [{r.level}] {r.note}"
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)

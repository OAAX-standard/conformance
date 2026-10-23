"""A self-contained HTML page (no network, no dependencies) rendered from the report document of report.build().
Print it to PDF from the browser: the print stylesheet opens every section."""
import json
from html import escape

OUTCOMES = {"pass": "Pass", "warn": "Warning", "fail": "Fail", "error": "Error", "blocked": "Blocked", "skipped": "Skipped", "na": "n/a"}
LEVEL_HELP = {
    "MUST": "required by the spec text: a failure fails the run",
    "SHOULD": "recommended or derived from the text: a failure is a warning",
    "PROPOSED": "planned for the next spec version: always a warning",
}
KNOWN = {"call", "status", "status_name", "value", "value_is_utf8", "error", "error_is_utf8", "never_returned"}

CSS = """
:root{color-scheme:light dark;--bg:#f6f7f9;--card:#fff;--text:#1b1f24;--muted:#667085;--line:#e4e7ec;--code:#f2f4f7;
--pass:#1a7f4b;--warn:#b54708;--fail:#c4321c;--error:#c4321c;--blocked:#175cd3;--skipped:#667085;--na:#667085;
--pass-bg:#e6f4ec;--warn-bg:#fdf0dc;--fail-bg:#fbe7e4;--error-bg:#fbe7e4;--blocked-bg:#e5eefc;--skipped-bg:#eef0f3;--na-bg:#eef0f3}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d2127;--text:#e8eaed;--muted:#9aa4b2;--line:#2c323a;--code:#262b33;
--pass:#5fd39a;--warn:#f5b061;--fail:#ff8a78;--error:#ff8a78;--blocked:#7fb0ff;--skipped:#9aa4b2;--na:#9aa4b2;
--pass-bg:#16301f;--warn-bg:#3a2a12;--fail-bg:#3d1b16;--error-bg:#3d1b16;--blocked-bg:#172a49;--skipped-bg:#262b33;--na-bg:#262b33}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px}
.head{display:flex;gap:18px;justify-content:space-between;align-items:flex-start;flex-wrap:wrap}
.muted{color:var(--muted)}.small{font-size:13px}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
code{background:var(--code);padding:1px 5px;border-radius:4px;word-break:break-word}
.verdict{font-size:15px;font-weight:650;padding:8px 14px;border-radius:999px;white-space:nowrap}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,150px));gap:10px;margin-top:16px}
.tile{border:1px solid var(--line);border-radius:10px;padding:10px 12px}.tile b{display:block;font-size:24px;line-height:1.1}
dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:14px 0 0}dt{color:var(--muted)}dd{margin:0;word-break:break-all}
.pill{display:inline-block;justify-self:start;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:650;white-space:nowrap}
.legend{display:flex;gap:6px 18px;flex-wrap:wrap;margin:8px 0 0}
.bar{display:flex;gap:8px;align-items:center;margin:14px 0 8px;flex-wrap:wrap}
.bar button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--text);padding:5px 12px;border-radius:999px;cursor:pointer}
.bar button[aria-pressed=true]{background:var(--text);color:var(--bg);border-color:var(--text)}
.cols,.test>summary{display:grid;grid-template-columns:92px 1fr 96px 96px;gap:10px;align-items:center}
.cols{padding:6px 16px;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.test{background:var(--card);border:1px solid var(--line);border-radius:10px;margin-bottom:6px;overflow:hidden}
.test>summary{padding:10px 16px;cursor:pointer;list-style:none}.test>summary::-webkit-details-marker{display:none}
.test>summary:hover{background:var(--code)}.test[open]>summary{border-bottom:1px solid var(--line)}
.id{font-family:ui-monospace,monospace;font-size:13px}
.body{padding:14px 16px 16px;display:grid;gap:12px}.body h3{margin:0 0 3px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.body p{margin:0}.msg{padding:10px 12px;border-radius:8px}
table.calls{width:100%;border-collapse:collapse;font-size:13px}.calls th{text-align:left;color:var(--muted);font-weight:600;padding:4px 8px}
.calls td:first-child{white-space:nowrap}.calls td{padding:4px 8px;border-top:1px solid var(--line);vertical-align:top;word-break:break-word}
details.inner>summary{cursor:pointer;color:var(--muted);font-size:13px}
.hide-pass .test[data-outcome=pass]{display:none}
"""

SCRIPT = """
const root=document.getElementById('tests');
for(const b of document.querySelectorAll('.bar button'))b.onclick=()=>{
  root.classList.toggle('hide-pass',b.dataset.f==='problems');
  for(const o of document.querySelectorAll('.bar button'))o.setAttribute('aria-pressed',o===b)};
window.onbeforeprint=()=>document.querySelectorAll('details').forEach(d=>d.open=true);
"""


def _pill(outcome: str, text: str | None = None) -> str:
    return f'<span class="pill" style="color:var(--{outcome});background:var(--{outcome}-bg)">{escape(text or OUTCOMES[outcome])}</span>'


def _calls(observations: list[dict]) -> str:
    rows = []
    for o in observations:
        status = f'{o["status"]} <span class="muted">{o["status_name"]}</span>' if "status" in o else ""
        detail = []
        for key, label in (("value", "value"), ("error", "error")):
            if key == "value" and key in o and o[key] is None:
                detail.append("value: <code>NULL</code>")
            elif key in o:
                bad = "" if o.get(key + "_is_utf8", True) else ' <span class="muted">(not valid UTF-8)</span>'
                detail.append(f"{label}: <code>{escape(o[key])}</code>{bad}")
        if "never_returned" in o:
            detail.append(f'<b style="color:var(--fail)">never returned: {escape(o["never_returned"])}</b>')
        detail += [f"{escape(k)}: <code>{escape(json.dumps(v))}</code>" for k, v in o.items() if k not in KNOWN]
        rows.append(f'<tr><td class="mono">{escape(o["call"])}</td><td>{status}</td><td>{"<br>".join(detail)}</td></tr>')
    return '<table class="calls"><tr><th>Call</th><th>Status</th><th>Observed</th></tr>' + "".join(rows) + "</table>"


def _test(r: dict, scenarios: dict) -> str:
    if r["outcome"] == "pass":  # the appendix holds the full record; repeating it under every passing test would bury the failures
        links = ", ".join(f'<a href="#sc-{escape(n)}">{escape(n)}</a>' for n in r["scenarios"] if n in scenarios)
        evidence = f'<p class="small muted">Evidence: {links}</p>' if links else ""
    else:
        evidence = "".join(
            f'<details class="inner" open><summary>Calls of the “{escape(n)}” scenario ({len(scenarios[n]["observations"])})</summary>'
            f'{_calls(scenarios[n]["observations"])}</details>'
            for n in r["scenarios"]
            if n in scenarios
        )
    message = f'<div><h3>Observed</h3><p class="msg" style="background:var(--{r["outcome"]}-bg)">{escape(r["message"])}</p></div>' if r["message"] else ""
    return (
        f'<details class="test" data-outcome="{r["outcome"]}"><summary><span class="id">{escape(r["id"])}</span>'
        f'<span>{escape(r["title"].replace("`", ""))}</span><span class="muted small">{escape(r["level"])}</span>{_pill(r["outcome"])}</summary>'
        f'<div class="body"><div><h3>Expected</h3><p>{escape(r["pass_when"].replace("`", ""))}</p></div>'
        f'<div><h3>Source</h3><p class="small">{escape(r["source"].replace("`", ""))}</p></div>{message}{evidence}</div></details>'
    )


def render(doc: dict) -> str:
    code, subject, summary = doc["verdict"]["exit_code"], doc["subject"], doc["summary"]
    verdict_class = {0: "pass", 1: "fail", 2: "blocked"}[code]
    kind = subject["kind"]
    tiles = "".join(
        f'<div class="tile" style="border-color:var(--{o}-bg)"><b style="color:var(--{o})">{n}</b>{OUTCOMES[o]}</div>'
        for o, n in summary["counts"].items()
        if n
    )
    legend = "".join(f'<span class="small"><b>{lv}</b> <span class="muted">{escape(LEVEL_HELP[lv])}</span></span>' for lv in summary["by_level"])
    leftovers = (f'<p class="small" style="color:var(--warn)">Temporary directories that could not be removed: {escape(", ".join(doc["leftovers"]))}</p>' if doc.get("leftovers") else "")
    options = doc["options"]
    if kind == "toolchain":
        options_line = f"docker args: {escape(' '.join(options['docker_args']) or 'none')} · runtime: {escape(options['runtime'] or 'none')}"
    else:
        options_line = f"init keys: {escape(', '.join(options['init_config_keys']) or 'none')} · ignored threads: {escape(', '.join(options['ignore_threads']) or 'none')}"
    scenarios = "".join(
        f'<details class="card" id="sc-{escape(n)}" style="margin-bottom:8px"><summary><b>{escape(n)}</b> <span class="muted small">exit {s["returncode"]}'
        f'{" (" + s["signal"] + ")" if s["signal"] else ""}{", timed out" if s["timed_out"] else ""}, {s["duration_ms"]} ms, {len(s["observations"])} calls</span></summary>'
        f'{_calls(s["observations"])}</details>'
        for n, s in doc["scenarios"].items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OAAX {escape(doc["spec_version"])} {kind} conformance: {escape(subject["name"])}</title><style>{CSS}</style></head>
<body><main>
<div class="card"><div class="head"><div><h1>OAAX {escape(doc["spec_version"])} {kind} conformance</h1>
<div class="muted">{escape(subject["detail"])} · <span class="mono">{escape(subject["name"])}</span></div></div>
<span class="verdict" style="color:var(--{verdict_class});background:var(--{verdict_class}-bg)">{escape(doc["verdict"]["text"])}</span></div>
<div class="tiles">{tiles}</div>
<dl class="small"><dt>{"Image" if kind == "toolchain" else "Library"}</dt><dd class="mono">{escape(subject["name"])}</dd><dt>{"Image id" if kind == "toolchain" else "SHA-256"}</dt><dd class="mono">{escape(subject["sha256"])}</dd>
<dt>Platform</dt><dd>{escape(doc["platform"])}</dd><dt>Run</dt><dd>{escape(doc["generated_at"])} · suite {escape(doc["suite_version"])}</dd>
<dt>Options</dt><dd>{options_line}</dd></dl>
<p class="muted small" style="margin:14px 0 0">{escape(doc["verdict"]["note"])} {summary["draft_tests_not_run"]} draft test(s) were not run.</p>{leftovers}</div>
<h2>Tests</h2><div class="legend">{legend}</div>
<div class="bar"><button data-f="all" aria-pressed="true">All ({summary["total"]})</button>
<button data-f="problems" aria-pressed="false">Not passing ({summary["total"] - summary["counts"]["pass"]})</button></div>
<div class="cols"><span>ID</span><span>Check</span><span>Level</span><span>Status</span></div>
<div id="tests">{"".join(_test(r, doc["scenarios"]) for r in doc["results"])}</div>
<h2>Everything the harness observed</h2>{scenarios}
</main><script>{SCRIPT}</script></body></html>
"""

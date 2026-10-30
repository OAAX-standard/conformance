#!/usr/bin/env python3
"""Print the GitHub release notes for a version: its CHANGELOG.md section plus the sha256 of every asset.

    uv run --no-project python scripts/release_notes.py 0.1.0 dist/SHA256SUMS [--final]
CI runs it on every build (a dry run: it fails if CHANGELOG.md has no section for the version). On a tag it adds --final, which also
refuses a heading without a release date (YYYY-MM-DD): the date must be set before the tag.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def changelog_section(version: str, text: str, final: bool = False) -> str:
    """The body of `## 0.1.0 (...)` or `## [0.1.0] ...`, up to the next `## ` heading that is not inside a code fence."""
    body, heading, fenced = [], None, False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            if heading is not None:
                break
            if re.match(rf"## \[?{re.escape(version)}\]?(\s|$)", line):
                heading = line
            continue
        if heading is not None:
            body.append(line)
    if heading is None:
        raise SystemExit(f"CHANGELOG.md has no '## {version}' section: add one before releasing")
    if final and not re.search(r"\d{4}-\d{2}-\d{2}", heading):
        raise SystemExit(f"CHANGELOG.md heading '{heading}' has no release date (YYYY-MM-DD): set it before tagging")
    section = "\n".join(body).strip()
    if not section:
        raise SystemExit(f"CHANGELOG.md section for {version} is empty")
    return section


def install_line(sums: str) -> str:
    """The wheel names come from SHA256SUMS: they are platform wheels (manylinux, one per architecture), so a hard-coded name is wrong."""
    wheels = [line.split()[-1] for line in sums.splitlines() if line.strip().endswith(".whl")]
    if not wheels:
        raise SystemExit("SHA256SUMS lists no wheel: nothing to build the install line from")
    wheel = next((w for w in wheels if "x86_64" in w), wheels[0])
    return (
        "Install from PyPI: `uv tool install --python 3.12 'oaax-conformance[e2e]'`, or from the files of this release: "
        f"`uv tool install --python 3.12 './{wheel}[e2e]'` (the `aarch64` wheel on an arm machine). Verify the download against the checksums above.\n"
    )


def notes(version: str, sums: str, final: bool = False) -> str:
    section = changelog_section(version, (ROOT / "CHANGELOG.md").read_text(), final)
    return f"{section}\n\n## Checksums (sha256)\n\n```\n{sums.strip()}\n```\n\n{install_line(sums)}"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--final"]
    if len(args) != 2:
        raise SystemExit(__doc__)
    print(notes(args[0], Path(args[1]).read_text(), final="--final" in sys.argv))

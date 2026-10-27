"""The release notes come from CHANGELOG.md, and the release workflow keeps its guards (round 5, A15)."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("release_notes", ROOT / "scripts" / "release_notes.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

TEXT = "# Changelog\n\n## 1.2.3 (unreleased)\n\n- a change\n\n```\n## not a heading\n```\n\n## [1.2.2] - 2026-01-01\n\n- old\n\n## 1.2.1 - 2025-12-01\n\n"


def test_the_section_of_a_version_is_found_with_or_without_brackets_and_fences_are_respected():
    assert tool.changelog_section("1.2.3", TEXT).endswith("## not a heading\n```")
    assert tool.changelog_section("1.2.2", TEXT) == "- old"
    with pytest.raises(SystemExit, match="no '## 1.2'"):
        tool.changelog_section("1.2", TEXT)  # a prefix is not the version


def test_a_tag_needs_a_dated_heading_and_a_non_empty_section():
    with pytest.raises(SystemExit, match="no release date"):
        tool.changelog_section("1.2.3", TEXT, final=True)
    assert tool.changelog_section("1.2.2", TEXT, final=True) == "- old"
    with pytest.raises(SystemExit, match="empty"):
        tool.changelog_section("1.2.1", TEXT)


def test_the_current_version_has_a_section_in_the_changelog():
    from oaax_conformance import __version__

    assert tool.changelog_section(__version__, (ROOT / "CHANGELOG.md").read_text())


PINNED = re.compile(r"[\w.-]+/[\w./-]+@[0-9a-f]{40} +# v\d+(\.\d+)*")  # owner/repo@<full commit SHA> # v<version>


def unpinned_actions(workflow: str) -> list[str]:
    """Every `uses:` of a workflow that is not a local action (`./...`) and not a full commit SHA with a version comment. Dependabot bumps the SHA and the
    comment together, so the test checks that the action is pinned, not which version it is."""
    problems = []
    for line in workflow.splitlines():
        match = re.match(r"\s*-?\s*uses:\s*(\S.*?)\s*$", line)
        if match and not match[1].startswith("./") and not PINNED.fullmatch(match[1]):
            problems.append(match[1])
    return problems


@pytest.mark.parametrize("line", ["actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0", "astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7 # v10", "./.github/actions/x",
                                  "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02   # v7.0.2"])
def test_a_pinned_action_is_accepted_whatever_its_version(line):
    assert unpinned_actions(f"      - uses: {line}\n") == []


@pytest.mark.parametrize("line", ["actions/checkout@v4", "actions/checkout@main", "actions/checkout@11d5960", "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
                                  "actions/checkout@11d5960a326750d5838078e36cf38b85af67726 # v4", "actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # latest",
                                  "actions/checkout@11D5960A326750D5838078E36CF38B85AF677262 # v4"])
def test_an_action_that_is_not_a_full_sha_with_a_version_comment_is_refused(line):
    assert unpinned_actions(f"      - uses: {line}\n") == [line]


def test_every_action_of_every_workflow_is_pinned_to_a_commit():
    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.y*ml")):
        assert unpinned_actions(workflow.read_text()) == [], workflow.name


def test_the_release_workflow_keeps_its_guards():
    """The steps only run on a tag, so no pull request exercises them: a change that drops one must fail here."""
    text = (ROOT / ".github" / "workflows" / "build.yml").read_text()
    for needle in ("fetch-depth: 0", 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main', "release_notes.py", "--final", "--notes-file dist/release-notes.md",
                   "gh release list", "overwrite: true", "scripts/smoke.sh"):
        assert needle in text, needle
    assert "| head" not in text and "| tee" not in text  # GitHub's default shell has no pipefail: a pipe hides the exit status
    assert "no-such-test" not in text


def test_every_yaml_file_under_dot_github_parses():
    """GitHub drops an issue form it cannot parse without saying so, and a workflow with invalid YAML fails only when it is needed (the tag-only steps never
    run in a pull request)."""
    import yaml

    files = [p for ext in ("yml", "yaml") for p in (ROOT / ".github").rglob(f"*.{ext}")]
    assert len(files) >= 5
    for path in files:
        yaml.safe_load(path.read_text())

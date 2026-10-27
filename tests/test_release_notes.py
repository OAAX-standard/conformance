"""The release notes come from CHANGELOG.md, and the release workflow keeps its guards (round 5, A15)."""
import importlib.util
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


def test_the_release_workflow_keeps_its_guards():
    """The steps only run on a tag, so no pull request exercises them: a change that drops one must fail here."""
    text = (ROOT / ".github" / "workflows" / "build.yml").read_text()
    for needle in ("fetch-depth: 0", 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main', "release_notes.py", "--final", "--notes-file dist/release-notes.md",
                   "gh release list", "overwrite: true", "scripts/smoke.sh", "# v4.4.0", "# v4.6.2", "# v4.3.0", "# v10.2.0"):
        assert needle in text, needle
    assert "| head" not in text and "| tee" not in text  # GitHub's default shell has no pipefail: a pipe hides the exit status
    assert "no-such-test" not in text

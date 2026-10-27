"""The wheel carries the C harness: where it is looked up, and what the build hook makes of a machine with and without cmake."""
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from oaax_conformance import harness

ROOT = Path(__file__).resolve().parents[1]


def _fake(path: Path) -> str:
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path.resolve())


def test_the_harness_is_looked_up_in_the_documented_order(tmp_path, monkeypatch):
    explicit, env, packaged, checkout = (_fake(tmp_path / n) for n in ("explicit", "env", "packaged", "checkout"))
    monkeypatch.setattr(harness, "packaged_harness", lambda: Path(packaged))
    monkeypatch.setenv("OAAX_HARNESS", env)
    assert harness.find_harness(explicit) == explicit  # 1. --harness
    assert harness.find_harness() == env  # 2. OAAX_HARNESS
    monkeypatch.delenv("OAAX_HARNESS")
    assert harness.find_harness() == packaged  # 3. the one inside the wheel
    monkeypatch.setattr(harness, "packaged_harness", lambda: tmp_path / "absent")
    monkeypatch.setattr(harness, "CHECKOUT_HARNESS", Path(checkout))
    assert harness.find_harness() == checkout  # 4. ./build of a dev checkout


def test_an_explicit_harness_that_is_missing_is_refused_not_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "packaged_harness", lambda: Path(_fake(tmp_path / "packaged")))
    monkeypatch.setenv("OAAX_HARNESS", str(tmp_path / "typo"))
    with pytest.raises(FileNotFoundError, match="OAAX_HARNESS .*typo: no such file"):
        harness.find_harness()
    with pytest.raises(FileNotFoundError, match="--harness .*nope: no such file"):
        harness.find_harness(str(tmp_path / "nope"))


def test_no_harness_anywhere_is_one_clear_line_and_exit_3(build_dir, tmp_path, monkeypatch, capsys):
    from oaax_conformance import cli

    monkeypatch.setattr(harness, "packaged_harness", lambda: tmp_path / "absent")
    monkeypatch.setattr(harness, "CHECKOUT_HARNESS", tmp_path / "absent")
    monkeypatch.delenv("OAAX_HARNESS", raising=False)
    assert cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--quiet"]) == 3
    err = capsys.readouterr().err.strip()
    assert "\n" not in err and "oaax-harness" in err and "Traceback" not in err


def _build(tmp_path, path_dirs):
    out = tmp_path / "dist"
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv not found")
    env = {**os.environ, "PATH": os.pathsep.join(path_dirs), "UV_PYTHON": sys.executable}
    env.pop("OAAX_REQUIRE_HARNESS", None)
    subprocess.run([uv, "build", "--out-dir", str(out), str(ROOT)], check=True, capture_output=True, env=env)  # sdist first, then the wheel FROM the sdist
    return sorted(out.glob("*.whl")), sorted(out.glob("*.tar.gz"))


@pytest.mark.skipif(not sys.platform.startswith("linux") or not shutil.which("cmake"), reason="the harness needs Linux and cmake")
def test_the_wheel_built_from_the_sdist_contains_an_executable_harness(tmp_path):
    wheels, sdists = _build(tmp_path, [str(Path(shutil.which("uv")).parent), *os.environ["PATH"].split(os.pathsep)])
    assert len(sdists) == 1 and len(wheels) == 1
    assert wheels[0].name.endswith("-py3-none-linux_" + os.uname().machine + ".whl") or "manylinux" in wheels[0].name  # platform tag, never py3-none-any
    with zipfile.ZipFile(wheels[0]) as z:
        info = z.getinfo("oaax_conformance/bin/oaax-harness")
        assert stat.S_IMODE(info.external_attr >> 16) & 0o111, "the executable bit is lost"


@pytest.mark.skipif(not shutil.which("uv"), reason="uv not found")
def test_without_cmake_the_sdist_still_builds_a_pure_wheel(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").symlink_to(shutil.which("uv"))
    wheels, _ = _build(tmp_path, [str(bin_dir)])  # a PATH with uv only: no cmake, no compiler
    assert [w.name for w in wheels] == ["oaax_conformance-" + harness_version() + "-py3-none-any.whl"]
    assert not any("bin/oaax-harness" in n for n in zipfile.ZipFile(wheels[0]).namelist())


def harness_version() -> str:
    from oaax_conformance import __version__

    return __version__


def test_model_at_tiny_conv_is_the_packaged_model(build_dir, monkeypatch):
    """README quick start: an installed wheel has no `python/oaax_conformance/models/` to point at."""
    from oaax_conformance import cli, toolchain

    seen = []
    monkeypatch.setattr(cli, "_runtime", lambda args: seen.append(args.model) or 0)
    assert cli.main(["runtime", str(build_dir / "libmock_runtime.so"), "--model", "@tiny-conv"]) == 0
    assert seen == [toolchain.default_model()] and Path(seen[0]).is_file()


def test_a_harness_that_is_not_executable_is_one_clear_line_not_an_internal_error(tmp_path, monkeypatch):
    """F2: a packaged binary without the x bit, or --harness FILE that is not executable, ended in "internal error PermissionError" (exit 4)."""
    import pytest

    from oaax_conformance import harness

    plain = tmp_path / "oaax-harness"
    plain.write_text("#!/bin/sh\n")
    plain.chmod(0o644)
    with pytest.raises(FileNotFoundError, match="not executable"):
        harness.find_harness(str(plain))
    monkeypatch.delenv("OAAX_HARNESS", raising=False)
    monkeypatch.setattr(harness, "packaged_harness", lambda: plain)
    with pytest.raises(FileNotFoundError, match="not executable"):
        harness.find_harness(None)
    plain.chmod(0o755)
    assert harness.find_harness(str(plain)) == str(plain.resolve())


def test_the_build_hook_removes_its_temporary_directory():
    """F4: every uv sync / uv build outside the release container left an `oaax-harness-*` directory in /tmp."""
    text = (Path(__file__).resolve().parent.parent / "hatch_build.py").read_text()
    assert "def finalize" in text and "rmtree" in text and "_workdir" in text

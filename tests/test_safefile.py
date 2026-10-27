"""Files a toolchain leaves behind are untrusted: a pipe must not hang the suite, a link must not be followed, a huge file must not be read whole."""
import os
import signal

import pytest

from oaax_conformance import modelio, safefile


class Hang(Exception):
    pass


@pytest.fixture
def must_not_hang():
    """A read that blocks would stop the whole test run: turn it into a failure after 20 s."""

    def alarm(*_):
        raise Hang("the read blocked")

    old = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(20)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)


def test_a_regular_file_is_read(tmp_path):
    (tmp_path / "f.json").write_text('{"a": 1}')
    assert safefile.read_json(tmp_path / "f.json") == {"a": 1}


def test_a_named_pipe_is_refused_without_blocking(tmp_path, must_not_hang):
    os.mkfifo(tmp_path / "oaax-manifest.json")
    with pytest.raises(safefile.Unreadable, match="not a regular file"):
        safefile.read_limited(tmp_path / "oaax-manifest.json")


def test_a_link_is_never_followed(tmp_path, must_not_hang):
    secret = tmp_path / "secret"
    secret.write_text("{}")
    os.symlink(secret, tmp_path / "m.json")
    os.symlink("/dev/zero", tmp_path / "zero.json")
    for name in ("m.json", "zero.json"):
        with pytest.raises(safefile.Unreadable):
            safefile.read_limited(tmp_path / name)


def test_a_file_over_the_limit_is_not_read(tmp_path):
    with open(tmp_path / "big.json", "wb") as f:
        f.truncate(3 << 30)  # sparse: 3 GB of nothing
    with pytest.raises(safefile.TooLarge):
        safefile.read_limited(tmp_path / "big.json", 1 << 20)


@pytest.mark.parametrize("make", ["fifo", "link_to_zero", "sparse"])
def test_the_manifest_next_to_a_model_never_hangs_or_exhausts_memory(tmp_path, must_not_hang, make):
    """C8: modelio read the manifest with Path.read_text(): a pipe blocked every scenario, /dev/zero or a sparse 3 GB file read without bound."""
    model = tmp_path / "model.bin"
    model.write_bytes(b"x")
    manifest = tmp_path / "oaax-manifest.json"
    if make == "fifo":
        os.mkfifo(manifest)
    elif make == "link_to_zero":
        os.symlink("/dev/zero", manifest)
    else:
        with open(manifest, "wb") as f:
            f.truncate(3 << 30)
    with pytest.raises(modelio.NoModelIO, match="manifest next to it cannot be used"):
        modelio.discover(str(model), [])


def test_a_manifest_that_is_valid_json_but_no_object_is_not_this_models_manifest(tmp_path):
    """D1: modelio caught (ValueError, KeyError, TypeError) but not the AttributeError of `[1].get(...)`."""
    model = tmp_path / "model.bin"
    model.write_bytes(b"x")
    for text in ("[1]", '"x"', "7", "null", '{"model_file": "model.bin"}'):  # the last: an object without inputs
        (tmp_path / "oaax-manifest.json").write_text(text)
        with pytest.raises(modelio.NoModelIO, match="cannot tell what the model takes"):
            modelio.discover(str(model), [])


def _peak_memory_mb_of(code: str, *args: str):
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-c", "import resource, sys\n" + code + "\nprint('MAXRSS_MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)", *args],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return done.stdout, int(done.stdout.rsplit("MAXRSS_MB", 1)[1])


@pytest.fixture
def huge_model(tmp_path):
    path = tmp_path / "model.onnx"
    with open(path, "wb") as f:
        f.write(b"MOCKMODEL")
        f.truncate(3 << 30)  # sparse: 3 GiB that onnx.load would read whole
    return path


def test_a_toolchain_written_model_over_2_gib_is_never_read_into_memory(huge_model):
    """D3: onnx.load read the whole file (8.4 GB of peak memory for a sparse 8 GiB `model.onnx`): a file larger than RAM would kill the testing host."""
    out, peak = _peak_memory_mb_of("""
from oaax_conformance import modelio
try:
    modelio.discover(sys.argv[1], [])
except modelio.NoModelIO as e:
    print("NOMODELIO", e)
""", str(huge_model))
    assert "NOMODELIO" in out and "GB: the suite does not read a model that big into memory" in out and "--input" in out
    assert peak < 1000, f"peak memory {peak} MB"


def test_the_converted_models_io_is_not_read_from_a_huge_file_either(huge_model):
    out, peak = _peak_memory_mb_of("""
from oaax_conformance.checks.toolchain import _io_views
from oaax_conformance.registry import Blocked
try:
    _io_views(sys.argv[1], None)
except Blocked as e:
    print("BLOCKED", e)
""", str(huge_model))
    assert "BLOCKED" in out and "GB" in out and peak < 1000, (out, peak)


def test_json_nested_beyond_the_recursion_limit_is_a_value_error_not_a_crash(tmp_path):
    """F1: RecursionError is a RuntimeError, so every `except ValueError` around a toolchain's or a runtime's JSON let it through."""
    import pytest

    from oaax_conformance import safefile

    with pytest.raises(ValueError, match="nested too deeply"):
        safefile.parse_json("[" * 200000)
    path = tmp_path / "m.json"
    path.write_text("[" * 200000)
    with pytest.raises(ValueError):
        safefile.read_json(path)
    assert safefile.parse_json('{"a": [1, 2]}') == {"a": [1, 2]}

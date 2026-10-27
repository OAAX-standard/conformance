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

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def build_dir() -> Path:
    if not (ROOT / "build" / "oaax-harness").exists():
        subprocess.run([str(ROOT / "scripts" / "build.sh")], check=True, capture_output=True)
    (ROOT / "build" / "mock.model").write_bytes(b"MOCKMODEL: the only kind of model the mock runtime loads")
    return ROOT / "build"

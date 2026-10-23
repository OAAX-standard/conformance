import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def build_dir() -> Path:
    if not (ROOT / "build" / "oaax-harness").exists():
        subprocess.run([str(ROOT / "scripts" / "build.sh")], check=True, capture_output=True)
    (ROOT / "build" / "mock.model").write_bytes(b"MOCKMODEL: the only kind of model the mock runtime loads")
    tensor = lambda name, shape: {"name": name, "native": {"dtype": "FLOAT", "shape": shape, "layout": "X" * len(shape)}}  # noqa: E731
    (ROOT / "build" / "oaax-manifest.json").write_text(json.dumps({"model_file": "mock.model", "inputs": [tensor("x", [1, 4]), tensor("scale", [1])],
                                                                    "outputs": [tensor("y", [1, 4]), tensor("sum", [1])]}))
    return ROOT / "build"

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("OAAX_BUILD_DIR", ROOT / "build")).resolve()  # CI points it at a sanitizer build


@pytest.fixture(scope="session")
def build_dir() -> Path:
    if not (BUILD / "oaax-harness").exists():
        subprocess.run([str(ROOT / "scripts" / "build.sh")], check=True, capture_output=True)
    (BUILD / "mock.model").write_bytes(b"MOCKMODEL: the only kind of model the mock runtime loads")
    tensor = lambda name, shape: {"name": name, "native": {"dtype": "FLOAT", "shape": shape, "layout": "X" * len(shape)}}  # noqa: E731
    (BUILD / "oaax-manifest.json").write_text(json.dumps({"model_file": "mock.model", "inputs": [tensor("x", [1, 4]), tensor("scale", [1])],
                                                                    "outputs": [tensor("y", [1, 4]), tensor("sum", [1, 1])]}))
    return BUILD


IMAGES = {  # image reference -> Dockerfile target
    "oaax-mock-toolchain:1.0": "clean",
    "oaax-mockroot-toolchain:1.0": "root",
    "oaax-mocknoentry-toolchain:1.0": "noentry",
}


@pytest.fixture(scope="session")
def toolchain_images(tmp_path_factory):
    """Build the mock toolchain images once per test run (workers wait for each other); skip when Docker is not there."""
    import fcntl
    import os

    from oaax_conformance.toolchain import docker_available

    if not docker_available():
        pytest.skip("Docker is not available")
    lock = tmp_path_factory.getbasetemp().parent / "docker-build.lock"
    with open(lock, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        base = os.environ.get("OAAX_MOCK_BASE", "python:3.11-slim")
        for image, target in IMAGES.items():
            subprocess.run(["docker", "build", "-q", "--target", target, "-t", image, "--build-arg", f"BASE={base}", "-f", "mock-toolchain/Dockerfile", "."],
                           cwd=ROOT, check=True, capture_output=True)
        subprocess.run(["docker", "tag", "oaax-mock-toolchain:1.0", "mocktoolchain:latest"], check=True)  # a name that breaks the convention
    return list(IMAGES) + ["mocktoolchain:latest"]


@pytest.fixture(scope="session")
def mock_io_onnx(tmp_path_factory) -> str:
    """An ONNX model with the I/O of the mock runtime: y = x * scale, sum = ReduceSum(y)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "scale"], ["y"]), helper.make_node("ReduceSum", ["y", "axes"], ["sum"], keepdims=1)],
        "mock-io",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4]), helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4]), helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("models") / "mock-io.onnx"
    path.write_bytes(model.SerializeToString())
    return str(path)


def pytest_sessionfinish(session, exitstatus):
    """The generic image name of the naming test is ours and only used by the tests: remove it once, in the controller (not in each xdist worker)."""
    if not hasattr(session.config, "workerinput"):
        subprocess.run(["docker", "rmi", "mocktoolchain:latest"], capture_output=True)

"""Writes python/oaax_conformance/models/tiny-conv.onnx: the small valid model the toolchain tests convert by default.
Conv(3->4, 3x3) + Relu + GlobalAveragePool on a 1x3x32x32 float image; only operators every toolchain supports. Needs onnx and numpy."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

weights = np.random.default_rng(0).standard_normal((4, 3, 3, 3)).astype(np.float32) * 0.1
graph = helper.make_graph(
    [
        helper.make_node("Conv", ["input", "weights"], ["conv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("GlobalAveragePool", ["relu"], ["output"]),
    ],
    "tiny-conv",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])],
    [numpy_helper.from_array(weights, "weights")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
out = Path(__file__).resolve().parents[1] / "python" / "oaax_conformance" / "models" / "tiny-conv.onnx"
out.write_bytes(model.SerializeToString())
print(f"{out} ({out.stat().st_size} bytes)")

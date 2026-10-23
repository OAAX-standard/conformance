"""What a model takes and returns: the suite needs it to build valid (and invalid) requests.

Sources, in this order: an oaax-manifest.json next to the model (the `native` views), the ONNX file itself (needs the optional
`onnx` package), or --input flags. A dynamic dimension (-1, or a symbolic name in ONNX) is sent as 1."""
import json
from dataclasses import dataclass
from pathlib import Path

DTYPES = {"FLOAT", "UINT8", "INT8", "UINT16", "INT16", "INT32", "INT64", "BOOL", "FLOAT16", "DOUBLE", "UINT32", "UINT64", "BFLOAT16"}


class NoModelIO(Exception):
    """The inputs of the model cannot be determined."""


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple

    def arg(self) -> str:
        return f"{self.name}:{self.dtype}:{','.join(str(d) for d in self.shape)}"


@dataclass(frozen=True)
class ModelIO:
    inputs: tuple
    outputs: tuple


def parse_arg(text: str) -> Tensor:
    """name:DTYPE:d1,d2 (the name may contain ':', so read from the right)."""
    rest, _, dims = text.rpartition(":")
    name, _, dtype = rest.rpartition(":")
    if not name or dtype not in DTYPES:
        raise NoModelIO(f"--input {text!r}: expected NAME:DTYPE:D1,D2 with DTYPE one of {sorted(DTYPES)}")
    return Tensor(name, dtype, tuple(max(int(d), 1) for d in dims.split(",") if d))


def _from_manifest(path: Path, model: Path) -> ModelIO | None:
    try:
        doc = json.loads(path.read_text())
        if doc.get("model_file") != model.name:
            return None
        view = lambda t: Tensor(t["name"], t["native"]["dtype"], tuple(max(d, 1) for d in t["native"]["shape"]))  # noqa: E731
        return ModelIO(tuple(view(t) for t in doc["inputs"]), tuple(view(t) for t in doc["outputs"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _from_onnx(model: Path) -> ModelIO:
    try:
        import onnx
    except ImportError:
        raise NoModelIO("cannot read the ONNX model: pip install oaax-conformance[onnx], or use --input")
    graph = onnx.load(str(model), load_external_data=False).graph
    initializers = {i.name for i in graph.initializer}

    def tensor(value) -> Tensor:
        tt = value.type.tensor_type
        dims = tuple(d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1 for d in tt.shape.dim)
        return Tensor(value.name, onnx.TensorProto.DataType.Name(tt.elem_type), dims)

    return ModelIO(tuple(tensor(i) for i in graph.input if i.name not in initializers), tuple(tensor(o) for o in graph.output))


def discover(model: str, manual_inputs: list[str]) -> ModelIO:
    if manual_inputs:
        return ModelIO(tuple(parse_arg(a) for a in manual_inputs), ())
    path = Path(model)
    for manifest in sorted(path.parent.glob("oaax-manifest.json")):
        io = _from_manifest(manifest, path)
        if io:
            return io
    if path.suffix == ".onnx":
        return _from_onnx(path)
    raise NoModelIO("cannot tell what the model takes: put its oaax-manifest.json next to it, or pass --input NAME:DTYPE:DIMS for every input")

"""What a model takes and returns: the suite needs it to build valid (and invalid) requests.

Sources: --input flags (explicit, they win), else an oaax-manifest.json next to the model (the `native` views), else the ONNX file itself
(needs the optional `onnx` package). A dynamic dimension (-1 in a manifest, a symbolic name in ONNX, `?` in --input) is None: requests send 1,
and no request is built with a "wrong" size for it."""
import json
from dataclasses import dataclass
from pathlib import Path

from .safefile import model_too_big, read_json

DTYPES = {"FLOAT", "UINT8", "INT8", "UINT16", "INT16", "INT32", "INT64", "BOOL", "FLOAT16", "DOUBLE", "UINT32", "UINT64", "BFLOAT16"}


class NoModelIO(Exception):
    """The inputs of the model cannot be determined."""


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple  # ints; None for a dynamic dimension

    def arg(self) -> str:
        return f"{self.name}:{self.dtype}:{','.join('?' if d is None else str(d) for d in self.shape)}"

    def concrete_shape(self) -> tuple:
        return tuple(1 if d is None else d for d in self.shape)


@dataclass(frozen=True)
class ModelIO:
    inputs: tuple
    outputs: tuple


DTYPE_BYTES = {"FLOAT": 4, "UINT8": 1, "INT8": 1, "UINT16": 2, "INT16": 2, "INT32": 4, "INT64": 8, "BOOL": 1, "FLOAT16": 2, "DOUBLE": 8, "UINT32": 4,
               "UINT64": 8, "BFLOAT16": 2}
MAX_REQUEST_BYTES = 1 << 30  # the harness builds every request in memory: a bigger one is a typo, not a model


def check_inputs(inputs) -> None:
    """User errors that must not look like a verdict on the runtime: an input named twice, a request too big to build."""
    names = [t.name for t in inputs]
    for name in sorted({n for n in names if names.count(n) > 1}):
        raise NoModelIO(f"the input {name!r} is listed twice")
    total = 0
    for t in inputs:
        elements = 1
        for d in t.concrete_shape():
            elements *= d
        total += elements * DTYPE_BYTES.get(t.dtype, 0)
    if total > MAX_REQUEST_BYTES:
        raise NoModelIO(f"one request would take {total / (1 << 30):.1f} GiB of memory (limit {MAX_REQUEST_BYTES >> 30} GiB): check the dimensions")


def parse_arg(text: str) -> Tensor:
    """name:DTYPE:d1,d2 (the name may contain ':', so read from the right)."""
    rest, _, dims = text.rpartition(":")
    name, _, dtype = rest.rpartition(":")
    if not name or dtype not in DTYPES:
        raise NoModelIO(f"--input {text!r}: expected NAME:DTYPE:D1,D2 with DTYPE one of {sorted(DTYPES)}")
    try:
        shape = tuple(None if d == "?" else int(d) for d in dims.split(",") if d)
    except ValueError:
        raise NoModelIO(f"--input {text!r}: a dimension must be a whole number, or ? when the model accepts any size")
    if any(d is not None and d < 1 for d in shape):
        raise NoModelIO(f"--input {text!r}: dimensions must be at least 1 (use ? for a dynamic one)")
    return Tensor(name, dtype, shape)


def _from_manifest(path: Path, model: Path) -> ModelIO | None:
    try:
        doc = read_json(path)  # OSError (a pipe, a link, too big) is the caller's to report; a bad document just is not this model's
        if doc.get("model_file") != model.name:
            return None
        view = lambda t: Tensor(t["name"], t["native"]["dtype"], tuple(None if d < 1 else d for d in t["native"]["shape"]))  # noqa: E731
        return ModelIO(tuple(view(t) for t in doc["inputs"]), tuple(view(t) for t in doc["outputs"]))
    except (ValueError, KeyError, TypeError, AttributeError):  # not an object (`[1]`, `"x"`), a missing key: a manifest that is not this model's
        return None


def _from_onnx(model: Path) -> ModelIO:
    try:
        import onnx
    except ImportError:
        raise NoModelIO("cannot read the ONNX model: install the onnx extra (README, Install), or use --input")
    if too_big := model_too_big(model):  # a toolchain wrote this file: onnx.load would read all of it into memory
        raise NoModelIO(too_big)
    try:
        graph = onnx.load(str(model), load_external_data=False).graph
    except Exception as e:  # noqa: BLE001 - a missing file, a protobuf error: all mean "cannot read it"
        raise NoModelIO(f"cannot read {model.name} as an ONNX model ({type(e).__name__}): pass --input NAME:DTYPE:DIMS for every input")
    initializers = {i.name for i in graph.initializer}

    def tensor(value) -> Tensor:
        tt = value.type.tensor_type
        dims = tuple(d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else None for d in tt.shape.dim)
        return Tensor(value.name, onnx.TensorProto.DataType.Name(tt.elem_type), dims)

    return ModelIO(tuple(tensor(i) for i in graph.input if i.name not in initializers), tuple(tensor(o) for o in graph.output))


def _buildable(io: ModelIO) -> ModelIO:
    """The harness builds a request only for the data types it knows; anything else must be blocked here, not crash the scenario."""
    for t in io.inputs:
        if t.dtype not in DTYPES:
            raise NoModelIO(f"input {t.name!r} has the data type {t.dtype}, which the suite cannot build a request for (supported: {sorted(DTYPES)})")
    return io


def discover(model: str, manual_inputs: list[str]) -> ModelIO:
    io = _discover(model, manual_inputs)
    check_inputs(io.inputs)
    return io


def _discover(model: str, manual_inputs: list[str]) -> ModelIO:
    if manual_inputs:
        return ModelIO(tuple(parse_arg(a) for a in manual_inputs), ())
    path = Path(model)
    unusable = []
    for manifest in sorted(path.parent.glob("oaax-manifest.json")):
        try:
            io = _from_manifest(manifest, path)
        except OSError as e:  # a pipe, a link or an absurdly big file where the manifest should be: never read it, say so
            unusable.append(str(e))
            continue
        if io:
            return _buildable(io)
    if path.suffix == ".onnx":
        return _buildable(_from_onnx(path))
    why = f" (the manifest next to it cannot be used: {'; '.join(unusable)})" if unusable else ""
    raise NoModelIO("cannot tell what the model takes: put its oaax-manifest.json next to it, or pass --input NAME:DTYPE:DIMS for every input" + why)

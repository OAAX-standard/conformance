"""The input data the C harness sends (harness/src/scenarios_model.c: fill), reproduced in Python for the end-to-end comparison.
Keep the two in step: tests/test_suite.py checks that the harness and this function produce the same bytes."""
import numpy as np

NUMPY_DTYPES = {"FLOAT": np.float32, "DOUBLE": np.float64, "FLOAT16": np.float16, "BOOL": np.bool_, "UINT8": np.uint8, "INT8": np.int8,
                "UINT16": np.uint16, "INT16": np.int16, "INT32": np.int32, "INT64": np.int64, "UINT32": np.uint32, "UINT64": np.uint64}


def deterministic_input(dtype: str, shape: tuple) -> np.ndarray:
    if dtype not in NUMPY_DTYPES:
        raise NotImplementedError(f"no reference input for {dtype}")
    count = int(np.prod(shape)) if shape else 1
    state = 12345
    values = np.empty(count, dtype=np.float64)
    small = np.empty(count, dtype=np.int64)
    for i in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        values[i] = np.float32((state >> 8) / 16777216.0)
        small[i] = (state >> 16) % 7
    if dtype in ("FLOAT", "DOUBLE"):
        return values.astype(NUMPY_DTYPES[dtype]).reshape(shape)
    if dtype == "FLOAT16":
        return np.full(shape, 0.5, dtype=np.float16)
    if dtype == "BOOL":
        return np.zeros(shape, dtype=np.bool_)
    return small.astype(NUMPY_DTYPES[dtype]).reshape(shape)

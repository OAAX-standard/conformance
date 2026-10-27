"""Shared by the self-tests: build a runtime Context with small load sizes (the mock needs 5 ms per request)."""
from oaax_conformance.suite import Context, run_suite


def context(build_dir, variant="mock_runtime", init_config=("vendor_key=1",), model=True, **overrides):
    values = dict(settle_ms=500, timeout=60, cycles=12, flood=100, requests=60, producers=3, per_producer=10, model_cycles=10, lost_after_ms=400,
                  retrieve_timeout_ms=1500, warmup_ms=3000)
    values.update(overrides)
    return Context(str(build_dir / f"lib{variant}.so"), str(build_dir / "oaax-harness"), list(init_config), str(build_dir / "mock.model") if model else None, **values)


def run(ctx, only=None, skip=None):
    return {r.id: r for r in run_suite(ctx, only, skip=skip).results}


def model_variant(tmp_path, inputs, outputs=None):
    """A copy of the mock model with a manifest of your own: inputs are (name, dtype, shape) with None for a dynamic dimension."""
    import json
    import shutil

    from conftest import BUILD

    tensor = lambda t: {"name": t[0], "native": {"dtype": t[1], "shape": [-1 if d is None else d for d in t[2]], "layout": "X" * len(t[2])}}  # noqa: E731
    shutil.copy(BUILD / "mock.model", tmp_path / "mock.model")
    outputs = outputs or [("y", "FLOAT", (1, 4)), ("sum", "FLOAT", (1, 1))]
    (tmp_path / "oaax-manifest.json").write_text(json.dumps({"model_file": "mock.model", "inputs": [tensor(t) for t in inputs], "outputs": [tensor(t) for t in outputs]}))
    return str(tmp_path / "mock.model")

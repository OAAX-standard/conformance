"""RT-M: tests that need a model (--model). Specs: docs/tests/runtime-model.md"""
from ..registry import Blocked, Fail, check

SUCCESS = 0


def _needs_model(ctx) -> None:
    if not ctx.model:
        raise Blocked("needs a model: pass --model FILE (a model file this runtime can load)")


@check("RT-M-01")
def second_load_models(ctx):
    _needs_model(ctx)
    status = ctx.run("load-twice").step("load_second")["status"]
    if status == SUCCESS:
        raise Fail("a second load_models call returned SUCCESS; it should return an error and leave the loaded models alone")

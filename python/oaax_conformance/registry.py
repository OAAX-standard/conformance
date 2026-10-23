"""Check registry and the two ways a check can end without passing."""
from typing import Callable

MUST, SHOULD, PROPOSED = "MUST", "SHOULD", "PROPOSED"


class Fail(Exception):
    """The runtime does not do what the check requires."""


class Blocked(Exception):
    """The check could not run (library did not load, init failed, ...); not a verdict on the runtime."""


CHECKS: dict[str, Callable] = {}


def check(test_id: str):
    """Register the implementation of a test whose metadata (level, source) lives in docs/tests."""

    def register(fn: Callable) -> Callable:
        assert test_id not in CHECKS, f"{test_id} registered twice"
        CHECKS[test_id] = fn
        return fn

    return register

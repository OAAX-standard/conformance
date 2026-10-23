"""Runs the approved checks against a runtime library."""
import traceback
from dataclasses import dataclass, field

from . import checks  # noqa: F401  (registers the checks)
from .harness import Run, run_scenario
from .registry import CHECKS, MUST, Blocked, Fail
from .report import Result
from .specdocs import Spec, load


@dataclass
class Context:
    """What a check may use: the library under test and the scenarios (each run once, cached)."""

    lib: str
    harness: str
    init_config: list[str] = field(default_factory=list)
    ignore_threads: list[str] = field(default_factory=list)
    timeout: int = 60
    cycles: int = 50
    settle_ms: int = 3000
    _runs: dict = field(default_factory=dict)

    def run(self, scenario: str) -> Run:
        if scenario not in self._runs:
            extra = ["--cycles", str(self.cycles), "--settle-ms", str(self.settle_ms)]
            self._runs[scenario] = run_scenario(self.harness, scenario, self.lib, self.init_config, self.timeout, extra)
        return self._runs[scenario]


def run_check(spec: Spec, ctx: Context) -> Result:
    result = Result(spec.id, spec.title, spec.level, spec.source, "pass")
    fn = CHECKS.get(spec.id)
    if fn is None:
        result.outcome, result.message = "skipped", "approved but not implemented"
        return result
    try:
        fn(ctx)
    except Fail as e:
        result.outcome, result.message = ("fail" if spec.level == MUST else "warn"), str(e)
    except Blocked as e:
        result.outcome, result.message = "blocked", str(e)
    except Exception:  # a bug in the check, not a verdict on the runtime
        result.outcome, result.message = "error", traceback.format_exc(limit=3).strip().splitlines()[-1]
    return result


def run_suite(ctx: Context, only: list[str] | None = None) -> tuple[list[Result], int]:
    specs = load()
    approved = [s for s in specs.values() if s.status == "approved" and (not only or s.id in only)]
    return [run_check(s, ctx) for s in approved], len(specs) - len(approved)


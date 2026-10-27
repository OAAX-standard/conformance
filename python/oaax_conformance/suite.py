"""Runs the approved checks against a runtime library."""
import traceback
from dataclasses import dataclass, field

from . import checks  # noqa: F401  (registers the checks)
import sys
import time

from .harness import Run, run_scenario
from .modelio import ModelIO, NoModelIO, discover
from .registry import CHECKS, MUST, Blocked, Fail, NotApplicable, Warn
from .report import Result
from .specdocs import Spec, load


@dataclass
class Context:
    """What a check may use: the library under test and the scenarios (each run once, cached)."""

    lib: str
    harness: str
    init_config: list[str] = field(default_factory=list)
    model: str | None = None  # --model: a model file this runtime can load; enables the model tier
    ignore_threads: list[str] = field(default_factory=list)
    timeout: int = 300  # seconds per scenario; loops stop at 60% of it, a call still running at the limit is blocked, not failed
    cycles: int = 50
    settle_ms: int = 3000
    manual_inputs: list = field(default_factory=list)  # --input NAME:DTYPE:DIMS: when the model cannot be read
    flood: int = 1000  # sizes of the load scenarios; --hardware-safe shrinks them
    requests: int = 500
    producers: int = 4
    per_producer: int = 25
    model_cycles: int = 20
    safe: bool = False  # --hardware-safe: fewer cycles by design (the notes and hints of the cycle checks say so)
    dump_dir: str | None = None  # the harness writes the outputs of the roundtrip scenario here (E2E tests)
    retrieve_timeout_ms: int = 10000  # how long a round trip waits for its output (scaled up from the first answered request)
    warmup_ms: int = 120000  # how long the very first request of a scenario may take
    lost_after_ms: int = 3000  # a request not answered after this long counts as lost
    pause: float = 0.0  # seconds between scenarios, for hardware that needs to settle
    workdir: str | None = None  # the harness runs here (default: a fresh temporary directory), for runtimes that need relative files
    progress: bool = False  # print each scenario to stderr as it starts
    _runs: dict = field(default_factory=dict)
    _io: ModelIO | None = None
    used: list = field(default_factory=list)  # scenarios read by the check running now

    def hardware_safe(self) -> None:
        """Gentle on a shared or fragile device: few cycles, small floods, a pause between scenarios."""
        self.cycles, self.model_cycles, self.flood, self.requests = 3, 3, 50, 50
        self.safe = True
        self.producers, self.per_producer, self.pause = 2, 10, 3.0

    def model_io(self) -> ModelIO:
        if not self.model:
            raise Blocked("needs a model: pass --model FILE (a model file this runtime can load)")
        if self._io is None:
            try:
                self._io = discover(self.model, self.manual_inputs)
            except NoModelIO as e:
                raise Blocked(str(e))
        return self._io

    def run(self, scenario: str) -> Run:
        if scenario not in self.used:
            self.used.append(scenario)
        if scenario not in self._runs:
            if self._runs and self.pause:
                time.sleep(self.pause)
            if self.progress:
                print(f"  running {scenario} ...", file=sys.stderr, flush=True)
            self._runs[scenario] = run_scenario(self.harness, scenario, self.lib, self.init_config, self.timeout, self._args(), self.workdir)
            warmup = next((l for l in self._runs[scenario].lines if l.get("step") == "warmup"), None)
            if warmup and warmup.get("status") != 0:  # the first request was not answered: do not wait that long again in every scenario
                self.warmup_ms = min(self.warmup_ms, 10000)
        return self._runs[scenario]

    def _args(self) -> list[str]:
        args = ["--cycles", str(self.cycles), "--settle-ms", str(self.settle_ms), "--flood", str(self.flood), "--requests", str(self.requests),
                "--producers", str(self.producers), "--per-producer", str(self.per_producer), "--model-cycles", str(self.model_cycles),
                "--timeout-ms", str(self.retrieve_timeout_ms), "--lost-after-ms", str(self.lost_after_ms), "--warmup-ms", str(self.warmup_ms)]
        if self.dump_dir:
            args += ["--dump", self.dump_dir]
        if self.model:
            args += ["-m", self.model]
            try:
                for tensor in self.model_io().inputs:
                    args += ["-i", tensor.arg()]
            except Blocked:
                pass  # scenarios that send requests are blocked by their checks; the others do not need the inputs
        return args


def run_check(spec: Spec, ctx, skip: dict | None = None) -> Result:
    result = Result(spec.id, spec.title, spec.level, spec.source, "pass", pass_when=spec.pass_when)
    ctx.used = []
    ctx.notes = []  # a check that passes on less than the specified work says so (Result.note)
    fn = CHECKS.get(spec.id)
    if skip and spec.id in skip:
        result.outcome, result.message = "skipped", skip[spec.id]
        return result
    if fn is None:
        result.outcome, result.message = "error", "approved in the specification but not implemented (a bug of the suite)"
        return result
    try:
        fn(ctx)
    except Fail as e:
        result.outcome, result.message = ("fail" if spec.level == MUST else "warn"), str(e)
    except Warn as e:
        result.outcome, result.message = "warn", str(e)
    except Blocked as e:
        result.outcome, result.message = "blocked", str(e)
    except NotApplicable as e:
        result.outcome, result.message = "na", str(e)
    except Exception:  # a bug in the check, not a verdict on the runtime
        result.outcome, result.message = "error", traceback.format_exc(limit=3).strip().splitlines()[-1]
    result.scenarios = list(ctx.used)
    result.note = "; ".join(ctx.notes)
    return result


@dataclass
class SuiteRun:
    results: list[Result]
    not_approved: int
    runs: dict  # scenario / conversion name -> what was observed (anything with observations())
    leftovers: list = field(default_factory=list)  # temporary directories that could not be removed
    selected: int = 0  # approved checks selected to run (--only), of `approved`
    approved: int = 0


def run_suite(ctx, only: list[str] | None = None, prefixes: tuple = ("RT-",), skip: dict | None = None) -> SuiteRun:
    """Run the approved checks whose id starts with one of `prefixes` (RT- for a runtime, TC- and E2E- for a toolchain)."""
    specs = {i: s for i, s in load().items() if i.startswith(prefixes)}
    approved = [s for s in specs.values() if s.status == "approved"]
    selected = [s for s in approved if not only or s.id in only]
    results = [run_check(s, ctx, skip) for s in selected]
    runs = {k: v for k, v in ctx._runs.items() if not isinstance(v, (list, Context))}
    for key, value in ctx._runs.items():
        if isinstance(value, list):  # conversions started together
            runs.update({f"{key}-{i}": run for i, run in enumerate(value)})
    return SuiteRun(results, len(specs) - len(approved), runs, selected=len(selected), approved=len(approved))

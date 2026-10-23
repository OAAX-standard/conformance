"""Runs the C harness (one scenario = one process) and reads back what it observed."""
import json
import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .registry import Blocked, Fail

EXIT_DLOPEN, EXIT_SYMBOLS, EXIT_INIT = 3, 4, 5


def find_harness(explicit: str | None = None) -> str:
    repo_build = Path(__file__).resolve().parents[2] / "build" / "oaax-harness"
    candidates = [explicit, os.environ.get("OAAX_HARNESS"), str(repo_build), shutil.which("oaax-harness")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError("oaax-harness not found: build it with scripts/build.sh, or pass --harness / set OAAX_HARNESS")


@dataclass
class Run:
    scenario: str
    returncode: int
    timed_out: bool
    lines: list[dict] = field(default_factory=list)

    @property
    def signal_name(self) -> str | None:
        return signal.Signals(-self.returncode).name if self.returncode < 0 else None

    @property
    def crashed_in(self) -> str | None:
        """The call that never returned: its begin line has no matching step line."""
        if self.returncode == 0 and not self.timed_out:
            return None
        pending = None
        for line in self.lines:
            if "begin" in line:
                pending = line["begin"]
            elif line.get("step") == pending:
                pending = None
        return pending

    def step(self, name: str) -> dict:
        for line in self.lines:
            if line.get("step") == name:
                return line
        if self.crashed_in == name:
            how = "did not return in time" if self.timed_out else f"crashed ({self.signal_name or 'exit ' + str(self.returncode)})"
            raise Fail(f"the call {how}: {name}")
        raise Blocked(self._why_missing(name))

    def raw(self, name: str, key: str = "str") -> bytes | None:
        value = self.step(name).get(key)
        return None if value is None else bytes.fromhex(value)

    def _why_missing(self, name: str) -> str:
        if self.returncode == EXIT_DLOPEN:
            return "the library does not load: " + (self.raw("dlopen") or b"?").decode(errors="replace")
        if self.returncode == EXIT_SYMBOLS:
            return "exports are missing (RT-S-01)"
        if self.returncode == EXIT_INIT:
            init = next((l for l in self.lines if l.get("step") == "init"), {})
            err = bytes.fromhex(init["err"]).decode(errors="replace") if init.get("err") else "no error text"
            return f"init failed with status {init.get('status')} ({err}); pass --init-config for this runtime"
        if self.crashed_in:
            return f"the scenario died in {self.crashed_in}"
        return f"the scenario ended (exit {self.returncode}) before {name}"


def run_scenario(harness: str, scenario: str, lib: str, config: list[str], timeout: int, extra: list[str] = ()) -> Run:
    with tempfile.TemporaryDirectory() as workdir:  # runtimes drop log files into their working directory
        out = Path(workdir) / "observations.jsonl"
        cmd = [harness, scenario, lib, "--out", str(out), *extra]
        for pair in config:
            cmd += ["-c", pair]
        timed_out = False
        try:
            returncode = subprocess.run(cmd, cwd=workdir, timeout=timeout, capture_output=True).returncode
        except subprocess.TimeoutExpired:
            returncode, timed_out = -signal.SIGKILL, True
        text = out.read_text() if out.exists() else ""

    return Run(scenario, returncode, timed_out, [json.loads(l) for l in text.splitlines() if l.strip()])

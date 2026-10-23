"""Runs the C harness (one scenario = one process) and reads back what it observed."""
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .registry import Blocked, Fail

EXIT_DLOPEN, EXIT_SYMBOLS, EXIT_INIT = 3, 4, 5
# RuntimeStatus of the 2.0 header (tests/test_suite.py checks this table against spec/2.0/interface.h)
STATUS_NAMES = ["SUCCESS", "ERROR", "NOT_INITIALIZED", "ALREADY_INITIALIZED", "MODEL_NOT_LOADED", "INVALID_ARGUMENT", "INVALID_MODEL",
                "FILE_NOT_FOUND", "OUT_OF_MEMORY", "INVALID_TENSOR", "TENSOR_SHAPE_MISMATCH", "TENSOR_TYPE_MISMATCH", "NO_OUTPUT_AVAILABLE",
                "INFERENCE_ERROR", "NOT_IMPLEMENTED", "TIMEOUT", "DEVICE_ERROR", "UNKNOWN_ERROR", "INVALID_MODEL_ID"]


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
    duration_ms: int = 0

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

    def observations(self) -> list[dict]:
        """Every call the scenario made, decoded for reports; a call that never returned comes last, marked."""
        out = []
        for line in self.lines:
            if "step" not in line:
                continue
            obs = {"call": line["step"]}
            status = line.get("status")
            if status is not None:
                obs["status"] = status
                obs["status_name"] = STATUS_NAMES[status] if 0 <= status < len(STATUS_NAMES) else "?"
            for key, label in (("str", "value"), ("err", "error")):
                if line.get(key) is not None:
                    raw = bytes.fromhex(line[key])
                    obs[label] = raw.decode("utf-8", errors="replace")
                    if label == "value" or raw:
                        obs[label + "_is_utf8"] = _is_utf8(raw)
            if line.get("getter") and line.get("str") is None:
                obs["value"] = None  # the runtime returned NULL
            obs.update({k: v for k, v in line.items() if k not in ("step", "status", "str", "err", "getter")})
            out.append(obs)
        if self.crashed_in:
            out.append({"call": self.crashed_in, "never_returned": "timed out" if self.timed_out else f"crashed ({self.signal_name or self.returncode})"})
        return out

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


def _is_utf8(raw: bytes) -> bool:
    try:
        raw.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def run_scenario(harness: str, scenario: str, lib: str, config: list[str], timeout: int, extra: list[str] = ()) -> Run:
    with tempfile.TemporaryDirectory() as workdir:  # runtimes drop log files into their working directory
        out = Path(workdir) / "observations.jsonl"
        cmd = [harness, scenario, lib, "--out", str(out), *extra]
        for pair in config:
            cmd += ["-c", pair]
        timed_out, started = False, time.monotonic()
        try:
            returncode = subprocess.run(cmd, cwd=workdir, timeout=timeout, capture_output=True).returncode
        except subprocess.TimeoutExpired:
            returncode, timed_out = -signal.SIGKILL, True
        text = out.read_text() if out.exists() else ""

    lines = [json.loads(l) for l in text.splitlines() if l.strip()]
    return Run(scenario, returncode, timed_out, lines, round((time.monotonic() - started) * 1000))

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

EXIT_USAGE = 2
# RuntimeStatus of the 2.0 header (tests/test_suite.py checks this table against spec/2.0/interface.h)
STATUS_NAMES = ["SUCCESS", "ERROR", "NOT_INITIALIZED", "ALREADY_INITIALIZED", "MODEL_NOT_LOADED", "INVALID_ARGUMENT", "INVALID_MODEL",
                "FILE_NOT_FOUND", "OUT_OF_MEMORY", "INVALID_TENSOR", "TENSOR_SHAPE_MISMATCH", "TENSOR_TYPE_MISMATCH", "NO_OUTPUT_AVAILABLE",
                "INFERENCE_ERROR", "NOT_IMPLEMENTED", "TIMEOUT", "DEVICE_ERROR", "UNKNOWN_ERROR", "INVALID_MODEL_ID"]
TAIL = 2000  # bytes of the runtime's own stdout / stderr kept for the report
GRACE_SECONDS = 5  # SIGTERM first: let the runtime release the device before it is killed


def status_name(status) -> str:
    """'NOT_INITIALIZED (2)'; a status the header does not define is shown as is."""
    if isinstance(status, int) and 0 <= status < len(STATUS_NAMES):
        return f"{STATUS_NAMES[status]} ({status})"
    return f"unknown status {status}"


def find_harness(explicit: str | None = None) -> str:
    if explicit:  # an explicit choice must be honoured or refused, never silently replaced
        if not Path(explicit).is_file():
            raise FileNotFoundError(f"--harness {explicit}: no such file")
        return str(Path(explicit).resolve())
    repo_build = Path(__file__).resolve().parents[2] / "build" / "oaax-harness"
    for candidate in (os.environ.get("OAAX_HARNESS"), str(repo_build), shutil.which("oaax-harness")):
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError("oaax-harness not found: build it with scripts/build.sh, or pass --harness / set OAAX_HARNESS")


def _is_utf8(raw: bytes) -> bool:
    try:
        raw.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


# the lifecycle scenario first runs init/cleanup once, throw-away: a crash there is a crash of the call a check is about
THROWAWAY_CALLS = {"init_plain": "init", "cleanup_plain": "cleanup"}


@dataclass
class Run:
    scenario: str
    returncode: int
    timed_out: bool
    lines: list[dict] = field(default_factory=list)
    duration_ms: int = 0
    timeout: int = 0  # seconds the scenario was allowed
    stdout_tail: str = ""  # what the runtime itself printed
    stderr_tail: str = ""

    @property
    def signal_name(self) -> str | None:
        if self.returncode >= 0:
            return None
        try:
            return signal.Signals(-self.returncode).name
        except ValueError:  # real-time signals (SIGRTMIN+n) are not in the enum
            return f"signal {-self.returncode}"

    @property
    def end(self) -> str | None:
        """Why the harness ended, from its own last line: ok, dlopen, symbols, init, model. None: the process ended some other way."""
        for line in reversed(self.lines):
            if "end" in line:
                return line["end"]
        return None

    @property
    def crashed_in(self) -> str | None:
        """The call that never returned: its begin line has no matching step line. Judged by the end marker, not the exit code: a
        runtime that calls exit() inside a call leaves no marker either."""
        if self.end is not None:
            return None
        pending = None
        for line in self.lines:
            if "begin" in line:
                pending = line["begin"]
            elif line.get("step") == pending:
                pending = None
        return pending

    def how_it_ended(self) -> str:
        if self.timed_out:
            return f"did not finish within {self.timeout} s"
        return f"crashed ({self.signal_name})" if self.returncode < 0 else f"ended the process (exit code {self.returncode})"

    def step(self, name: str) -> dict:
        for line in self.lines:
            if line.get("step") == name:
                return line
        if THROWAWAY_CALLS.get(self.crashed_in, self.crashed_in) == name:
            if self.timed_out:  # the spec sets no time limit for a call: a slow device is not a failing one
                raise Blocked(f"the call {name} {self.how_it_ended()}: a slow device or a hang? raise --timeout if the device is just slow")
            raise Fail(f"the call {name} {self.how_it_ended()}")
        raise Blocked(self._why_missing(name))

    def raw(self, name: str, key: str = "str") -> bytes | None:
        value = self.step(name).get(key)
        return None if value is None else bytes.fromhex(value)

    def thread_names(self, name: str) -> list[str]:
        return [bytes.fromhex(h).decode("utf-8", errors="replace") for h in self.step(name)["threads"]]

    def _err(self, step_names: tuple) -> str:
        for line in self.lines:
            if line.get("step") in step_names and line.get("status") and line.get("err"):
                return f" ({bytes.fromhex(line['err']).decode(errors='replace')})"
        return ""

    def _why_missing(self, name: str) -> str:
        end = self.end
        if end == "dlopen":
            text = bytes.fromhex(next((l["str"] for l in self.lines if l.get("step") == "dlopen" and l.get("str")), "")).decode(errors="replace")
            return f"the library does not load: {text or '?'}; its dependencies must be found (set LD_LIBRARY_PATH)"
        if end == "symbols":
            return "exports are missing (RT-S-01)"
        if end == "init":
            init = next((l for l in self.lines if l.get("step") in ("init", "init_plain") and l.get("status")), {})
            return f"init failed with status {init.get('status')}{self._err(('init', 'init_plain'))}; pass the vendor's keys with --init-config"
        if end == "model":
            load = next((l for l in self.lines if str(l.get("step", "")).startswith("load") and l.get("status")), {})
            return f"the runtime could not load the model given with --model (status {load.get('status')}{self._err(('load', 'load_first', 'load_one'))})"
        if self.crashed_in:
            return f"the scenario {self.how_it_ended()} in {self.crashed_in}"
        if self.timed_out:
            return f"the scenario {self.how_it_ended()} before {name}: a slow device? raise --timeout"
        return f"the scenario {self.how_it_ended()} before {name}"

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
            obs.update({k: self._plain(k, v) for k, v in line.items() if k not in ("step", "status", "str", "err", "getter")})
            out.append(obs)
        if self.crashed_in:
            out.append({"call": self.crashed_in, "never_returned": self.how_it_ended()})
        for key, text in (("stdout_tail", self.stdout_tail), ("stderr_tail", self.stderr_tail)):
            if text.strip():
                out.append({"call": f"(the runtime's own {key.split('_')[0]})", "value": text})
        return out

    @staticmethod
    def _plain(key: str, value):
        """Thread names travel as hex (they are bytes); everything else is already plain."""
        if key == "threads":
            return [bytes.fromhex(h).decode("utf-8", errors="replace") for h in value]
        return value


def _tail(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - TAIL))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _stop(proc: subprocess.Popen) -> None:
    """SIGTERM the whole process group (the runtime may have forked helpers), SIGKILL it after a grace period."""
    for sig, wait in ((signal.SIGTERM, GRACE_SECONDS), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # helpers that outlived the harness
    except ProcessLookupError:
        pass


def run_scenario(harness: str, scenario: str, lib: str, config: list[str], timeout: int, extra: list[str] = (), workdir: str | None = None) -> Run:
    own_dir = tempfile.TemporaryDirectory() if workdir is None else None  # runtimes drop log files into their working directory
    cwd = Path(workdir) if workdir else Path(own_dir.name)
    scratch = tempfile.TemporaryDirectory()
    try:
        scratch_dir = Path(scratch.name)
        out, stdout, stderr = scratch_dir / "observations.jsonl", scratch_dir / "stdout", scratch_dir / "stderr"
        cmd = [harness, scenario, lib, "--out", str(out), "--budget-ms", str(int(timeout * 600)), *extra]
        for pair in config:
            cmd += ["-c", pair]
        timed_out, started = False, time.monotonic()
        # output goes to files, not pipes: a runtime that forks a helper keeps a pipe open for as long as the helper lives
        with open(stdout, "wb") as o, open(stderr, "wb") as e:
            proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=o, stderr=e, start_new_session=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop(proc)
            except BaseException:
                _stop(proc)
                raise
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # leftovers of a runtime that daemonised
                except ProcessLookupError:
                    pass
        text = out.read_text(errors="replace") if out.exists() else ""
        lines = []
        for line in text.splitlines():
            try:
                lines.append(json.loads(line))
            except ValueError:
                pass  # a half-written last line of a killed process
        return Run(scenario, proc.returncode, timed_out, lines, round((time.monotonic() - started) * 1000), timeout, _tail(stdout), _tail(stderr))
    finally:
        scratch.cleanup()
        if own_dir:
            own_dir.cleanup()

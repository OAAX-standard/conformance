"""Runs the C harness (one scenario = one process) and reads back what it observed."""
import contextlib
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .registry import Blocked, Fail

WINDOWS = os.name == "nt"
HARNESS_NAME = "oaax-harness.exe" if WINDOWS else "oaax-harness"
LIB_HINT = "RuntimeLibrary.dll" if WINDOWS else "libRuntimeLibrary.so"  # how the runtime file is usually called on this OS
# Windows reports a crash as an exception code (the exit code of the process), not as a signal
NT_EXCEPTIONS = {0xC0000005: "ACCESS_VIOLATION", 0xC00000FD: "STACK_OVERFLOW", 0xC0000094: "INT_DIVIDE_BY_ZERO", 0xC000001D: "ILLEGAL_INSTRUCTION",
                 0xC0000409: "STACK_BUFFER_OVERRUN (fast fail, e.g. abort())", 0xC0000374: "HEAP_CORRUPTION", 0xC0000142: "DLL_INIT_FAILED", 0xC0000135: "DLL_NOT_FOUND"}
SIGHUP = getattr(signal, "SIGHUP", None)  # no SIGHUP on Windows

EXIT_USAGE = 2
# RuntimeStatus of the 2.0 header (tests/test_suite.py checks this table against spec/2.0/interface.h)
STATUS_NAMES = ["SUCCESS", "ERROR", "NOT_INITIALIZED", "ALREADY_INITIALIZED", "MODEL_NOT_LOADED", "INVALID_ARGUMENT", "INVALID_MODEL",
                "FILE_NOT_FOUND", "OUT_OF_MEMORY", "INVALID_TENSOR", "TENSOR_SHAPE_MISMATCH", "TENSOR_TYPE_MISMATCH", "NO_OUTPUT_AVAILABLE",
                "INFERENCE_ERROR", "NOT_IMPLEMENTED", "TIMEOUT", "DEVICE_ERROR", "UNKNOWN_ERROR", "INVALID_MODEL_ID"]
TAIL = 2000  # bytes of the runtime's own stdout / stderr kept for the report
CAPTURE = 20000  # bytes captured: secrets are removed from these first and the tail is cut afterwards, so a cut can never leave half of one


def hide_secrets(text: str, secrets) -> str:
    """Replace every secret value by ***, longest first. Call it BEFORE cutting a text to its tail or head."""
    for secret in sorted({s for s in secrets if s and len(s) >= 3}, key=len, reverse=True):
        text = text.replace(secret, "***")
    return text
GRACE_SECONDS = 5  # SIGTERM first: let the runtime release the device before it is killed


def status_name(status) -> str:
    """'NOT_INITIALIZED (2)'; a status the header does not define is shown as is."""
    if isinstance(status, int) and 0 <= status < len(STATUS_NAMES):
        return f"{STATUS_NAMES[status]} ({status})"
    return f"unknown status {status}"


CHECKOUT_HARNESS = Path(__file__).resolve().parents[2] / "build" / HARNESS_NAME  # `scripts/build.sh` in a dev checkout


def packaged_harness() -> Path:
    """Where the wheel keeps the harness (the build hook puts it there; absent in a pure wheel or a checkout)."""
    return Path(__file__).resolve().parent / "bin" / HARNESS_NAME


def find_harness(explicit: str | None = None) -> str:
    """--harness, OAAX_HARNESS, the harness inside the wheel, then ./build of a dev checkout. An explicit choice is honoured or refused, never replaced."""
    for source, path in (("--harness", explicit), ("OAAX_HARNESS", os.environ.get("OAAX_HARNESS"))):
        if path:
            if not Path(path).is_file():
                raise FileNotFoundError(f"{source} {path}: no such file")
            if not WINDOWS and not os.access(path, os.X_OK):
                raise FileNotFoundError(f"{source} {path}: not executable (chmod +x it)")
            return str(Path(path).resolve())
    for candidate in (packaged_harness(), CHECKOUT_HARNESS):
        if candidate.is_file():
            if not WINDOWS and not os.access(candidate, os.X_OK):
                raise FileNotFoundError(f"{candidate}: the harness is not executable (a broken install: reinstall, or chmod +x it)")
            return str(candidate)
    raise FileNotFoundError("oaax-harness (the C harness of the runtime tier) not found: this install has none (the Linux and Windows wheels contain it; a macOS or source install "
                            "does not). Use a wheel, or build it (scripts/build.sh, scripts/build.ps1 on Windows) and pass --harness PATH or set OAAX_HARNESS")


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
    lib: str = ""  # the library under test: a dlopen error that starts with its own name is about that file, not about a missing dependency

    @property
    def signal_name(self) -> str | None:
        if self.returncode >= 0:
            return None
        if WINDOWS:  # run_scenario stores an NTSTATUS exit code (0xC0000005) as a negative number
            code = self.returncode + 2**32
            return NT_EXCEPTIONS.get(code, f"exception 0x{code:08X}")
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
            return f"stalled for {self.timeout} s (or took over {self.timeout * TOTAL_FACTOR} s in all)"
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

    def dlopen_error(self) -> str:
        """What dlerror() said when the harness could not load the library (empty when it did)."""
        return bytes.fromhex(next((l["str"] for l in self.lines if l.get("step") == "dlopen" and l.get("str")), "")).decode(errors="replace")

    def _why_missing(self, name: str) -> str:
        end = self.end
        if end == "dlopen":
            text = self.dlopen_error()
            own = bool(self.lib) and text.startswith(f"{self.lib}:")  # "<the library>: cannot open shared object file: Permission denied": not a dependency
            missing = "cannot open shared object file" in text or "Windows error 126" in text or "Windows error 1114" in text
            where = "put them next to the library or on PATH" if WINDOWS else "set LD_LIBRARY_PATH"
            hint = f"; its dependencies must be found ({where})" if missing and not own else ""  # advice only for a missing dependency
            return f"the library does not load: {text or '?'}{hint}"
        if end == "symbols":
            return "exports are missing (RT-S-01)"
        if end == "init":
            init = next((l for l in self.lines if l.get("step") in ("init", "init_plain") and l.get("status")), {})
            return f"init failed with status {init.get('status')}{self._err(('init', 'init_plain'))}; pass the vendor's keys with --init-config"
        if end == "model":
            load = next((l for l in self.lines if str(l.get("step", "")).startswith("load") and l.get("status")), {})
            return f"the runtime could not load the model given with --model (status {load.get('status')}{self._err(('load', 'load_first', 'load_one'))})"
        if self.timed_out:
            where = f" in the call {self.crashed_in}" if self.crashed_in else ""
            return f"the scenario {self.how_it_ended()}{where} before {name}: a slow device? raise --timeout"
        if self.crashed_in:
            return f"the scenario {self.how_it_ended()} in {self.crashed_in}"
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


class TailCapture:
    """The last `limit` bytes of what a child process prints. A pipe read by a thread, never a file: a runtime (or a toolchain) that prints without
    end would fill the host's disk, and the suite's temp directories with it. A helper that outlives the child holds the pipe open: nobody waits for EOF."""

    def __init__(self, limit: int = TAIL):
        self.limit, self._tail = limit, b""
        self._read, self.write_fd = os.pipe()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while True:
            try:
                chunk = os.read(self._read, 1 << 16)
            except OSError:
                return
            if not chunk:
                return
            self._tail = (self._tail + chunk)[-self.limit:]

    def child_started(self) -> None:
        os.close(self.write_fd)  # the child has its own copy: EOF arrives when it (and its helpers) are gone

    def abandon(self) -> None:
        """Before an error leaves: the child never started, so nobody else holds the write end."""
        for fd in (self.write_fd,):
            try:
                os.close(fd)
            except OSError:
                pass

    def value(self) -> bytes:
        self._thread.join(timeout=2)
        if not self._thread.is_alive():
            os.close(self._read)
        return self._tail


@contextlib.contextmanager
def uninterruptible():
    """Cleanup that must finish: a second Ctrl-C (uv forwards one) or a SIGTERM must not abort it and leave a harness or a container running."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    saved = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM, SIGHUP) if s is not None}
    try:
        yield
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)


def _stop(proc: subprocess.Popen) -> None:
    """SIGTERM the whole process group (the runtime may have forked helpers), SIGKILL it after a grace period."""
    with uninterruptible():
        _stop_group(proc)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Hard kill of the process and the helpers it started."""
    if WINDOWS:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _stop_group(proc: subprocess.Popen) -> None:
    if not WINDOWS:  # a polite SIGTERM first: lets the runtime release the device. Windows has no equivalent for a console process: it is killed.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=GRACE_SECONDS)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass
    _kill_tree(proc)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


TOTAL_FACTOR = 6  # a scenario may take this many timeouts in all, as long as no single call stalls


def _wait_while_progressing(proc, observations: Path, timeout: int) -> bool:
    """True when the process ended. The timeout is the longest a single call may take, not the length of the scenario: the harness writes a line
    before every call and after it, so a slow device (a 76 s init, four times) is not cut off while each call is still finishing."""
    started = last_progress = time.monotonic()
    size = 0
    while True:
        try:
            proc.wait(timeout=0.5)
            return True
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        grown = observations.stat().st_size if observations.exists() else 0
        if grown != size:
            size, last_progress = grown, now
        if now - last_progress > timeout or now - started > TOTAL_FACTOR * timeout:
            return False


def run_scenario(harness: str, scenario: str, lib: str, config: list[str], timeout: int, extra: list[str] = (), workdir: str | None = None) -> Run:
    own_dir = tempfile.TemporaryDirectory() if workdir is None else None  # runtimes drop log files into their working directory
    cwd = Path(workdir) if workdir else Path(own_dir.name)
    scratch = tempfile.TemporaryDirectory()
    try:
        scratch_dir = Path(scratch.name)
        out = scratch_dir / "observations.jsonl"
        cmd = [harness, scenario, lib, "--out", str(out), "--budget-ms", str(int(timeout * 600)), *extra]
        for pair in config:
            cmd += ["-c", pair]
        timed_out, started = False, time.monotonic()
        stdout, stderr = TailCapture(CAPTURE), TailCapture(CAPTURE)  # bounded: see TailCapture
        try:
            # own session / process group: a Ctrl-C of the user's terminal must not reach the harness before the suite has stopped it properly
            detach = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {"start_new_session": True}
            proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stdout.write_fd, stderr=stderr.write_fd, **detach)
        except BaseException:
            stdout.abandon(), stderr.abandon()
            raise
        stdout.child_started(), stderr.child_started()
        try:
            timed_out = not _wait_while_progressing(proc, out, timeout)
            if timed_out:
                _stop(proc)
        except BaseException:
            _stop(proc)
            raise
        if not timed_out and not WINDOWS:
            _kill_tree(proc)  # leftovers of a runtime that daemonised
        text = out.read_text(errors="replace") if out.exists() else ""
        lines = []
        for line in text.splitlines():
            try:
                lines.append(json.loads(line))
            except ValueError:
                pass  # a half-written last line of a killed process
        secrets = [pair.split("=", 1)[1] for pair in config if "=" in pair]
        tail = lambda capture: hide_secrets(capture.value().decode("utf-8", errors="replace"), secrets)[-TAIL:]  # noqa: E731 - scrub first, cut afterwards
        returncode = proc.returncode - 2**32 if WINDOWS and proc.returncode >= 0xC0000000 else proc.returncode  # an NTSTATUS crash counts like a signal: negative
        return Run(scenario, returncode, timed_out, lines, round((time.monotonic() - started) * 1000), timeout, tail(stdout), tail(stderr), lib)
    finally:
        scratch.cleanup()
        if own_dir:
            own_dir.cleanup()

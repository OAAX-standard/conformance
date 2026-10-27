"""Runs a toolchain image on temporary directories and records what it did.

Every conversion gets its own workspace: `in/` (the input, writable by any uid) and `out/` (the output mount, writable by any uid).
Directories owned by another uid are never created by the suite itself: a test that runs the container as an arbitrary uid writes into
the existing `out/`, so every file the container creates sits in a directory the invoking user owns and can delete."""
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .harness import TailCapture, uninterruptible
from .registry import Blocked

DOCKER_ERROR = 125  # `docker run` exit code for its own failures (not the container's)

ARBITRARY_UID = "12345:12345"
LOG_NAMES = ("conversion-log.json", "logs.json")
MANIFEST = "oaax-manifest.json"
TAIL = 3000  # bytes of console output kept per stream


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def inspect(image: str) -> dict:
    """The parts of `docker inspect` the checks use."""
    out = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    if out.returncode:
        raise FileNotFoundError(f"image {image!r} not found: {out.stderr.strip()}")
    info = json.loads(out.stdout)[0]
    config = info.get("Config") or {}
    return {"id": info["Id"], "user": config.get("User") or "", "entrypoint": config.get("Entrypoint") or [], "cmd": config.get("Cmd") or [],
            "architecture": info.get("Architecture"), "os": info.get("Os")}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:  # chunked: a compiled model can be several GB
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tree(root: Path) -> dict[str, dict]:
    """Every file and directory under root: relative path -> size, mode, sha256 (files)."""
    entries = {}
    for path in sorted(root.rglob("*")):
        st = path.lstat()
        entry = {"size": st.st_size, "mode": oct(st.st_mode & 0o7777), "dir": stat.S_ISDIR(st.st_mode), "uid": st.st_uid}
        if stat.S_ISLNK(st.st_mode):  # never followed: it may point at a host file the toolchain has no business with
            entry["link"] = os.readlink(path)
        elif stat.S_ISREG(st.st_mode):
            try:
                entry["sha256"] = _digest(path)
            except OSError:
                entry["sha256"] = None  # not readable by us: TC-22 reports it
        entries[str(path.relative_to(root))] = entry
    return entries


@dataclass
class Workspace:
    root: Path
    in_dir: Path
    out_dir: Path

    @classmethod
    def create(cls) -> "Workspace":
        root = Path(tempfile.mkdtemp(prefix="oaax-conformance-"))
        in_dir, out_dir = root / "in", root / "out"
        for d in (in_dir, out_dir):
            d.mkdir()
            d.chmod(0o777)  # any uid the container runs as may write here
        return cls(root, in_dir, out_dir)

    def stage(self, filename: str, content: bytes) -> Path:
        path = self.in_dir / filename
        path.write_bytes(content)
        path.chmod(0o644)
        return path


KEEP_BYTES = 1 << 20


def _drop_big_files(directory: Path) -> None:
    """Delete the big files (compiled models are GB) of a finished conversion so disk use does not grow with the number of conversions.
    A file another uid created in a directory of its own cannot be removed here: close() opens those up and removes them at the end."""
    try:
        for path in directory.rglob("*"):
            try:
                if path.name not in (*LOG_NAMES, MANIFEST) and path.is_file() and not path.is_symlink() and path.stat().st_size > KEEP_BYTES:
                    path.unlink()
            except OSError:
                pass
    except OSError:
        pass


@dataclass
class Conversion:
    """One `docker run` of the toolchain and what it left behind."""

    name: str
    command: list[str]  # with host paths replaced by <in> / <out>
    returncode: int
    timed_out: bool
    duration_ms: int
    stdout: str
    stderr: str
    workspace: Workspace
    input_before: dict
    input_after: dict
    output: dict  # tree of out/ after the run
    out_path: str  # the output argument, relative to the mount
    container_changes: list = field(default_factory=list)  # paths added or changed inside the container (docker diff), if it was asked for

    signal_name = None

    @property
    def console(self) -> str:
        return (self.stdout + self.stderr).strip()

    @property
    def result_dir(self) -> Path:
        return self.workspace.out_dir / Path(self.out_path).relative_to("/output")

    def files(self) -> dict:
        """Files of the result directory, relative to it."""
        prefix = str(Path(self.out_path).relative_to("/output"))
        prefix = "" if prefix == "." else prefix + "/"
        return {k[len(prefix):]: v for k, v in self.output.items() if k.startswith(prefix) and not v["dir"] and "link" not in v}

    def log_files(self) -> list[str]:
        return [n for n in self.files() if n in LOG_NAMES]

    def model_files(self) -> list[str]:
        return [n for n, v in self.files().items() if n not in LOG_NAMES and n != MANIFEST and v["size"] > 0]

    def input_changes(self) -> list[str]:
        return sorted(k for k in set(self.input_before) | set(self.input_after) if self.input_before.get(k) != self.input_after.get(k))

    def observations(self) -> list[dict]:
        listing = [{"file": n, "size": v["size"], "mode": v["mode"]} for n, v in self.files().items()]
        obs = {"call": " ".join(self.command), "exit_code": self.returncode, "duration_ms": self.duration_ms, "output_files": listing}
        if self.timed_out:
            obs["never_returned"] = "timed out"
        if self.input_changes():
            obs["input_directory_changed"] = self.input_changes()
        for key, text in (("stdout", self.stdout), ("stderr", self.stderr)):
            if text.strip():
                obs[key + "_tail"] = text[-TAIL:]
        return [obs]


def _redact(text: str, workspace: Workspace) -> str:
    return text.replace(str(workspace.root), "<tmp>")


def make_hostile_zip(model: bytes) -> bytes:
    """A zip with a valid model and members that climb out of an extraction directory (zip-slip)."""
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("model.onnx", model)
        for depth in range(1, 5):
            z.writestr("../" * depth + f"escape-{depth}.txt", "outside the output directory\n")
    return buf.getvalue()


@dataclass
class ToolchainContext:
    """What a toolchain check may use: the image under test and its conversions (each run once, cached)."""

    image: str
    onnx_path: str
    docker_args: list[str] = field(default_factory=list)
    timeout: int = 900
    runtime_lib: str | None = None
    harness: str | None = None
    init_config: list[str] = field(default_factory=list)
    runtime_timeout: int = 300
    warmup_ms: int = 120000  # for the runtime of the E2E tests
    workspaces: list = field(default_factory=list)
    leftovers: list = field(default_factory=list)
    used: list = field(default_factory=list)
    _runs: dict = field(default_factory=dict)
    docker_failure: str = ""  # set when `docker run` itself failed
    timeouts: dict = field(default_factory=dict)  # conversion name -> why it blocks: it did not finish in time
    live: set = field(default_factory=set)  # names of the containers started and not yet finished
    _inspect: dict | None = None
    _dump: str | None = None
    progress: bool = False  # print each conversion to stderr as it starts

    def runtime_context(self, model_path: str):
        """A runtime Context for the converted model (E2E tests); None if no --runtime was given."""
        from .suite import Context

        if not self.runtime_lib:
            return None
        key = f"runtime:{model_path}"
        if key not in self._runs:
            self._dump = tempfile.mkdtemp(prefix="oaax-conformance-dump-")
            self._runs[key] = Context(self.runtime_lib, self.harness, list(self.init_config), model_path, timeout=self.runtime_timeout, dump_dir=self._dump, warmup_ms=self.warmup_ms)
        return self._runs[key]

    def model_bytes(self) -> bytes:
        return Path(self.onnx_path).read_bytes()

    def image_info(self) -> dict:
        if self._inspect is None:
            self._inspect = inspect(self.image)
        return self._inspect

    def _docker(self, ws: Workspace, name: str, argv: list[str], *, user: str | None, network: bool, read_only_rootfs: bool, ro_input: bool,
                ro_output: bool) -> list[str]:
        cmd = ["docker", "run", "--rm", "--name", f"oaax-conformance-{uuid.uuid4().hex[:12]}"]
        cmd += ["--user", user or f"{os.getuid()}:{os.getgid()}"]
        if not network:
            cmd += ["--network", "none"]
        if read_only_rootfs:
            cmd += ["--read-only", "--tmpfs", "/tmp"]
        cmd += ["-v", f"{ws.in_dir}:/input{':ro' if ro_input else ''}", "-v", f"{ws.out_dir}:/output{':ro' if ro_output else ''}"]
        return cmd + self.docker_args + [self.image] + argv

    def convert(self, name: str, *, filename: str = "model.onnx", content: bytes | None = None, out_path: str = "/output/result",
                user: str | None = None, network: bool = True, read_only_rootfs: bool = False, ro_input: bool = False, ro_output: bool = False,
                argv: list[str] | None = None, workspace: Workspace | None = None, keep_container: bool = False) -> "Conversion":
        """Run the toolchain once. `argv` replaces the two normal arguments (wrong argument counts); `content` is the input file."""
        if self.docker_failure:  # docker itself cannot start containers: no run can say anything about the toolchain
            raise Blocked(self.docker_failure)
        if name in self.timeouts:
            raise Blocked(self.timeouts[name])
        if name in self._runs:
            return self._runs[name]
        if self.progress:
            import sys

            print(f"  converting: {name} ...", file=sys.stderr, flush=True)
        ws = workspace or Workspace.create()
        if ws not in self.workspaces:
            self.workspaces.append(ws)
        if content is not None or argv is None:
            ws.stage(filename, self.model_bytes() if content is None else content)
        args = argv if argv is not None else [f"/input/{filename}", out_path]
        cmd = self._docker(ws, name, args, user=user, network=network, read_only_rootfs=read_only_rootfs, ro_input=ro_input, ro_output=ro_output)
        before = tree(ws.in_dir)
        started, timed_out = time.monotonic(), False
        container = cmd[cmd.index("--name") + 1]
        if keep_container:
            cmd.remove("--rm")  # so that `docker diff` can still list what the toolchain changed inside the container
        self.live.add(container)
        out_capture, err_capture = TailCapture(20000), TailCapture(20000)  # bounded: an image that prints without end must not fill the disk
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out_capture.write_fd, stderr=err_capture.write_fd)
        except BaseException:
            out_capture.abandon(), err_capture.abandon()
            raise
        out_capture.child_started(), err_capture.child_started()
        try:
            try:
                code = proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                timed_out, code = True, -signal.SIGKILL
                proc.kill()
                proc.wait()
        except BaseException:  # Ctrl-C, or anything else: the container must not outlive the suite
            self.remove_live()
            raise
        if timed_out:
            self.remove_live(container)  # only this one: a parallel conversion may still be running
        out, err = out_capture.value(), err_capture.value()
        self.live.discard(container)
        changes = []
        if keep_container:
            diff = subprocess.run(["docker", "diff", container], capture_output=True, text=True).stdout
            changes = [line[2:] for line in diff.splitlines() if line[:2] in ("A ", "C ")]
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        shown = [c.replace(str(ws.in_dir), "<in>").replace(str(ws.out_dir), "<out>") for c in cmd]
        run = Conversion(name, shown, code, timed_out, round((time.monotonic() - started) * 1000), _redact(out.decode(errors="replace"), ws),
                         _redact(err.decode(errors="replace"), ws), ws, before, tree(ws.in_dir), tree(ws.out_dir), _result_path(args), changes)
        # Later checks read the snapshot above and small files (logs, manifest); only the first valid conversion's model is loaded again (E2E).
        _drop_big_files(ws.in_dir)
        if name != "valid" and workspace is None:
            _drop_big_files(ws.out_dir)
        if timed_out:  # the spec sets no time limit: what a stalled run did or did not leave behind proves nothing, for any check
            self.timeouts[name] = f"a conversion did not finish within {self.timeout} s (the spec sets no limit): raise --timeout if the model is just big"
            raise Blocked(self.timeouts[name])
        self._runs[name] = run
        if code == DOCKER_ERROR and not out and b"docker" in err.lower():  # docker's own failure (bad --docker-arg, daemon, GPU runtime): a rejection by the toolchain looks different
            first = (run.stderr.strip().splitlines() or ["no message"])[0][:200]
            self.docker_failure = f"docker could not start the container (exit {DOCKER_ERROR}: {first}); check the Docker setup and --docker-arg"
            raise Blocked(self.docker_failure)
        return run

    def convert_together(self, name: str, count: int = 2) -> list["Conversion"]:
        """Start several conversions at the same time, each in its own workspace."""
        if name in self._runs:
            return self._runs[name]
        results: list = [None] * count
        import threading

        def worker(i: int) -> None:
            try:
                results[i] = self.convert(f"{name}-{i}")
            except Exception as e:  # a thread's exception would be lost (Blocked: docker could not start)
                results[i] = e

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(count)]
        [t.start() for t in threads]
        try:
            [t.join() for t in threads]
        except BaseException:  # Ctrl-C reaches this thread only: the workers' own cleanup never runs, and a container may ignore SIGINT
            self.remove_live()
            raise
        for r in results:
            if isinstance(r, Exception):
                raise r
        self._runs[name] = results
        return results

    def remove_live(self, only: str | None = None) -> None:
        """`docker rm -f` the containers this context started and that have not finished (all, on an interrupt): nothing may outlive the suite."""
        with uninterruptible():
            for name in [only] if only else list(self.live):
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
                self.live.discard(name)

    def close(self) -> None:
        """Remove the temporary directories. A toolchain that creates subdirectories as another uid leaves some we cannot delete: open them up
        from a throwaway container of the same image (limited to that workspace), and report anything that still cannot be removed."""
        for ws in self.workspaces:
            try:
                shutil.rmtree(ws.root)
                continue
            except OSError:
                pass
            self._make_removable(ws)
            try:
                shutil.rmtree(ws.root)
            except OSError:
                self.leftovers.append(str(ws.root))
        self.workspaces.clear()
        if self._dump:
            shutil.rmtree(self._dump, ignore_errors=True)

    def _make_removable(self, ws: Workspace) -> None:
        cmd = ["docker", "run", "--rm", "--user", "0", "--network", "none", "-v", f"{ws.root}:/w", "--entrypoint", "chmod", self.image, "-R", "a+rwX", "/w"]
        subprocess.run(cmd, capture_output=True, timeout=120)  # best effort: an image without chmod simply leaves the directory


def _result_path(args: list[str]) -> str:
    """Where the results are: the last argument if it is a path under /output, else the whole output mount (wrong argument counts)."""
    last = args[-1] if args else ""
    return last if last == "/output" or last.startswith("/output/") else "/output"


def default_model() -> str:
    return str(Path(__file__).parent / "models" / "tiny-conv.onnx")


_NAME = re.compile(r"^oaax-[a-z0-9][a-z0-9._-]*-toolchain(-[a-z0-9][a-z0-9._-]*)?:[A-Za-z0-9_][A-Za-z0-9._-]*$")


def follows_naming(image: str) -> bool:
    """oaax-<vendor>-toolchain[-<version>]:<tag>, with the registry/namespace prefix of the reference ignored."""
    return bool(_NAME.match(image.rsplit("/", 1)[-1]))

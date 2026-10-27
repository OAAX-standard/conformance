"""Build hook: compile the C harness and ship it inside the wheel (oaax_conformance/bin/oaax-harness).

- OAAX_WHEEL_PLAT (e.g. manylinux_2_28_x86_64) is the platform tag of the wheel. The release builds in a manylinux container, where that tag is honest.
  Without it the tag is this machine's (linux_x86_64): fine to install here, not to publish.
- No cmake / compiler / Linux: a pure wheel without the harness (the CLI says so when the runtime tier needs it), unless OAAX_REQUIRE_HARNESS=1 (the release).
"""
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ROOT = Path(__file__).resolve().parent


class HarnessHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        required = os.environ.get("OAAX_REQUIRE_HARNESS") == "1"
        try:
            binary = self._compile()
        except (OSError, subprocess.CalledProcessError) as e:
            if required:
                raise
            print(f"WARNING: the C harness was not built ({e}): the wheel lacks the runtime tier", file=sys.stderr)
            return
        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-" + (os.environ.get("OAAX_WHEEL_PLAT") or sysconfig.get_platform().replace("-", "_").replace(".", "_"))
        build_data["force_include"][str(binary)] = "oaax_conformance/bin/oaax-harness"

    def _compile(self) -> Path:
        if not sys.platform.startswith("linux") or not shutil.which("cmake"):  # the harness uses dlfcn and /proc
            raise OSError("needs Linux and cmake")
        out = Path(tempfile.mkdtemp(prefix="oaax-harness-"))  # kept: hatch reads the file after this hook returns
        # -ffile-prefix-map: the same bytes whatever directory the build ran in
        flags = f"-ffile-prefix-map={ROOT}=."
        run = lambda *cmd: subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: E731
        try:
            run("cmake", "-S", str(ROOT), "-B", str(out), "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_C_FLAGS={flags}")
            run("cmake", "--build", str(out), "--target", "oaax-harness", "-j")
        except subprocess.CalledProcessError as e:
            raise OSError(f"cmake failed: {(e.stderr or e.stdout or '').strip()[-300:]}") from e
        binary = out / "oaax-harness"
        subprocess.run(["strip", str(binary)], check=False)  # smaller; harmless when strip is missing
        return binary

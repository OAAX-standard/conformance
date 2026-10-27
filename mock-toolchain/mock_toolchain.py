"""A tiny OAAX toolchain for testing the conformance suite itself. It is correct unless the environment variable OAAX_MOCK_FAULT
names a fault: every toolchain check must fail on its fault and pass on the clean mock.

"Converting" validates the ONNX model and writes model.bin (the 9 bytes MOCKMODEL followed by the ONNX bytes, which is what the mock
runtime loads), logs.json and oaax-manifest.json. Usage: mock_toolchain.py <input.onnx> <output-directory>"""
import json
import os
import shutil
import socket
import sys
import time
import zipfile

sys.path.insert(0, "/app")
from oaax_conformance import manifest  # noqa: E402


def fault(name: str) -> bool:
    return name in os.environ.get("OAAX_MOCK_FAULT", "").split(",")


def say(text: str) -> None:
    if not fault("silent_failure"):
        print(text, flush=True)


def write_log(out: str, ok: bool, message: str) -> None:
    if os.path.isdir(out) and not (fault("no_log") or (not ok and fault("no_log_on_failure"))):
        if fault("fifo_log"):  # a named pipe where the log should be: whoever opens it blocks forever
            os.mkfifo(os.path.join(out, "logs.json"))
            return
        with open(os.path.join(out, "logs.json"), "w") as f:
            f.write("not json" if fault("bad_json_log") else json.dumps([{"Message": message, "Data": {"ok": ok}}]))
            if fault("huge_log"):  # sparse: no disk is used
                f.truncate(2 << 20)


def fail(out: str, message: str, code: int = 1) -> int:
    say(f"error: {message}")
    try:
        os.makedirs(out, exist_ok=True)
        if fault("leave_model_on_failure"):
            with open(os.path.join(out, "model.bin"), "wb") as f:
                f.write(b"MOCKMODEL")
        write_log(out, False, message)
    except OSError:
        pass
    return 0 if fault("exit0_on_error") else code


def validate(data: bytes) -> str | None:
    if fault("accept_garbage"):
        return None
    try:
        import onnx

        model = onnx.load_model_from_string(data)
    except Exception as e:  # noqa: BLE001 - any parse failure is an invalid model
        return f"not an ONNX model: {type(e).__name__}"
    if not fault("accept_empty") and not model.graph.node:
        return "the model has no nodes"
    return None


def extract_archive(path: str, out: str) -> int:
    if fault("zipslip_tmp"):  # extracts under /tmp inside the container: the members climb out into /tmp, where no mount can see them
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                target = os.path.join("/tmp/work/extract", member)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as f:
                    f.write(z.read(member))
        return fail(out, "archives are not supported")
    if not fault("zip_slip"):
        return fail(out, "archives are not supported")
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():  # naive: joins the member path to the output directory
            target = os.path.join(out, member)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(z.read(member))
    return 0


def describe(data: bytes, out: str) -> dict | None:
    """The I/O of the model, or None when there is no model to describe (fault accept_garbage)."""
    path = os.path.join(out, "original.onnx")
    with open(path, "wb") as f:
        f.write(data)
    try:
        return manifest.derive_io(path)
    except Exception:  # noqa: BLE001
        return None
    finally:
        os.remove(path)


def tamper(doc: dict, out: str) -> dict:
    if fault("manifest_names_link"):  # model_file is a link to the model: validate() followed it, the suite's listing does not
        os.symlink("model.bin", os.path.join(out, "model.lnk"))
        doc["model_file"] = "model.lnk"
    if fault("manifest_invalid"):
        doc["inputs"][0]["native"]["dtype"] = "STRING"
    if fault("manifest_wrong_names"):
        doc["inputs"][0]["name"] = doc["inputs"][0]["name"] + "_renamed"
    return doc


def main(argv: list[str]) -> int:
    if len(argv) != 2 and not (fault("bad_args_ok") and len(argv) == 3):
        print("usage: mock_toolchain.py <input> <output-directory>")
        return 2
    path, out = argv[0], argv[-1]
    if fault("echo_token"):  # prints the TOKEN variable so that the last 3000 characters begin in the middle of it (padding after it)
        # 4000 + 14 + 2984 + newline + the 10 characters of "converted\n" printed at the end = 7009: the last 3000 start 5 characters before the end of a 14-character TOKEN
        print("A" * 4000 + os.environ.get("TOKEN", "") + "B" * 2984, flush=True)
    if fault("fail_always"):
        return fail(out, "failing on purpose")
    if fault("hang"):
        time.sleep(600)
    if fault("needs_root") and os.getuid() != 0:
        return fail(out, "must run as root")
    if fault("needs_network"):
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=2).close()
        except OSError:
            return fail(out, "no network")
    if fault("rw_root"):
        open("/app/cache.txt", "w").close()
    if fault("ascii_only") and not path.isascii():
        return fail(out, "non-ASCII path")
    if fault("lock") and os.path.isdir("/lock"):
        try:
            os.mkdir("/lock/busy")
        except FileExistsError:
            return fail(out, "another conversion is running")
        time.sleep(2)
        os.rmdir("/lock/busy")
    if os.path.isdir(path):
        return 0 if fault("accept_dir") else fail(out, "the input is a directory", 2)
    if not os.path.exists(path):
        return 0 if fault("accept_missing") else fail(out, f"input not found: {path}", 2)
    if path.endswith(".zip"):
        return extract_archive(path, out)
    with open(path, "rb") as f:
        data = f.read()
    problem = validate(data)
    if problem:
        return fail(out, problem)
    if fault("not_idempotent") and os.path.isdir(out) and os.listdir(out):
        return fail(out, "the output directory is not empty")
    if fault("wipe_second") and os.path.isdir(out) and os.listdir(out):  # a "clean the output directory first" toolchain that then names its file differently
        shutil.rmtree(out)
        model_name_override = "other.bin"
    else:
        model_name_override = None
    if fault("no_mkdir") and not os.path.isdir(out):
        return fail(out, "the output directory does not exist")
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:
        return 0 if fault("unwritable_ok") else fail(out, f"cannot create the output directory: {e}")
    if fault("writes_input_dir"):
        open(path + ".cache", "w").close()
    model_name = model_name_override or ("model.onnx" if fault("huge_model") else "model.hef" if fault("vendor_format") else "model.bin")  # a vendor format, and (see below) no manifest to describe it
    if not fault("no_model"):
        with open(os.path.join(out, model_name), "wb") as f:
            f.write(b"JUNK" if fault("corrupt_model") else b"MOCKMODEL" + data)
            if fault("huge_model"):  # sparse: no disk is used, but onnx.load would read 3 GiB
                f.truncate(3 << 30)
    if fault("two_models"):  # a second file the runtime also loads, sorting before model.bin
        with open(os.path.join(out, "a-copy.bin"), "wb") as f:
            f.write(b"MOCKMODEL" + data)
    if fault("second_file"):  # vendor metadata next to the model (2.0 toolchains have no manifest to say which file is the model)
        with open(os.path.join(out, "a-labels.json"), "w") as f:
            f.write('{"labels": ["face"]}')
    if fault("big_output"):  # a compiled model is big: a sparse 8 MiB file stands for it
        with open(os.path.join(out, "weights.bin"), "wb") as f:
            f.truncate(8 << 20)
    if fault("unreadable_file"):  # a mode-000 file next to the model, owned by the uid that runs the suite
        secret = os.path.join(out, "secret.dat")
        open(secret, "w").write("x")
        os.chmod(secret, 0)
    if fault("deep_tree"):  # ~1700 nested directories: a host path of ~3.5 kB (below PATH_MAX) and far above any recursion limit
        here = os.getcwd()
        os.chdir(out)
        for _ in range(1700):
            os.mkdir("d")
            os.chdir("d")
        os.chdir(here)
    if fault("subdirs"):  # subdirectories created as the uid the container runs as
        os.makedirs(os.path.join(out, "sub", "deep"), exist_ok=True)
        open(os.path.join(out, "sub", "deep", "file"), "w").close()
    write_log(out, True, "converted")
    io = None if fault("no_manifest") or fault("vendor_format") or fault("huge_model") else describe(data, out)
    if io:
        doc = {"schema": "oaax-manifest/1", "oaax_version": "2.1", "model_file": "model.bin", "precision": "FP32", "equivalent": True, **io}
        if fault("raw_heads"):  # the runtime returns something else than the ONNX outputs: declared honestly
            doc["equivalent"] = False
            doc["outputs"] = [{"name": "head0", "native": {"dtype": "FLOAT", "shape": [1, 8], "layout": "XX"}}]
        target = os.path.join(out, "oaax-manifest.json")
        payload = next((v for f, v in (("manifest_list", [1]), ("manifest_string", "x"), ("manifest_number", 7)) if fault(f)), None)  # valid JSON that is no object
        if fault("manifest_no_inputs"):
            del doc["inputs"]
        if fault("fifo_manifest"):  # a named pipe where the manifest should be: whoever opens it blocks forever
            os.mkfifo(target)
        elif fault("link_manifest"):  # a link to something endless
            os.symlink("/dev/zero", target)
        else:
            with open(target, "w") as f:
                json.dump(payload if payload is not None else tamper(doc, out), f)
                if fault("huge_manifest"):  # sparse: no disk is used
                    f.truncate(40 << 20)
    if fault("no_perm"):
        for name in os.listdir(out):
            os.chmod(os.path.join(out, name), 0o600)
    print("converted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

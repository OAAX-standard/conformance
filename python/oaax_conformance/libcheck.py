"""Is this file a shared library this machine can load? One line when it is not, before 60 BLOCKED rows say the same thing."""
import platform
import struct
import sys
from pathlib import Path

MACHINES = {3: "x86 (32-bit)", 8: "MIPS", 40: "ARM (32-bit)", 62: "x86_64", 183: "aarch64", 243: "RISC-V"}
HOST = {"x86_64": 62, "amd64": 62, "aarch64": 183, "arm64": 183, "riscv64": 243}


def library_problem(path: str) -> str | None:
    """Why `path` cannot be a runtime library for this machine (read from its ELF header), or None."""
    if not sys.platform.startswith("linux"):
        return None  # the harness loads .so files with dlopen: Linux is where the checks run
    p = Path(path)
    if p.is_dir():
        return "is a directory"
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            head = f.read(64)
    except PermissionError:
        return "cannot be read (permission denied): fix its permissions"
    except OSError as e:
        return f"cannot be read ({e.strerror})"
    if size == 0:
        return "is empty"
    if head[:4] != b"\x7fELF":
        return "is not an ELF file: pass the libRuntimeLibrary.so built for this machine"
    if len(head) < 52 or head[4] not in (1, 2) or head[5] not in (1, 2):
        return "has an invalid ELF header"
    endian = "<" if head[5] == 1 else ">"
    kind = struct.unpack(endian + "H", head[16:18])[0]
    if kind != 3:  # ET_DYN: a shared library (a position-independent executable is ET_DYN too: dlopen tells those apart)
        return {2: "is an executable, not a shared library", 1: "is an object file, not a shared library", 4: "is a core dump"}.get(kind, f"is not a shared library (ELF type {kind})")
    machine = struct.unpack(endian + "H", head[18:20])[0]
    host_name, host = platform.machine(), HOST.get(platform.machine().lower())
    if host and machine != host:
        return f"is built for {MACHINES.get(machine, f'machine {machine}')}, this machine is {host_name}: pass the library built for this machine"
    if host and head[4] == 1:
        return f"is a 32-bit library, this machine ({host_name}) needs a 64-bit one"
    if head[4] == 2 and len(head) >= 64:
        phoff, shoff = struct.unpack(endian + "QQ", head[32:40] + head[40:48])
        phentsize, phnum, shentsize, shnum = struct.unpack(endian + "HHHH", head[54:62])
        ends = phoff + phentsize * phnum, shoff + shentsize * shnum if shoff else 0
        if max(ends) > size:
            return f"is truncated ({size} bytes; its headers say it is at least {max(ends)} bytes): download it again"
    return None

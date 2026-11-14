"""Is this file a shared library this machine can load? One line when it is not, before 60 BLOCKED rows say the same thing."""
import platform
import struct
import sys
from pathlib import Path

MACHINES = {3: "x86 (32-bit)", 8: "MIPS", 40: "ARM (32-bit)", 62: "x86_64", 183: "aarch64", 243: "RISC-V"}
HOST = {"x86_64": 62, "amd64": 62, "aarch64": 183, "arm64": 183, "riscv64": 243}


PE_MACHINES = {0x14C: "x86 (32-bit)", 0x1C4: "ARM (32-bit)", 0x8664: "x86_64", 0xAA64: "aarch64"}
PE_HOST = {"x86_64": 0x8664, "amd64": 0x8664, "aarch64": 0xAA64, "arm64": 0xAA64}


def library_problem(path: str) -> str | None:
    """Why `path` cannot be a runtime library for this machine (read from its ELF / PE header), or None."""
    if sys.platform == "win32":
        return _pe_problem(path)
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
    if head[:2] == b"MZ":
        return "is a Windows DLL (a PE file): run the suite on Windows, or pass the .so built for this machine"
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


def _read_pe(path: str):
    """(machine, characteristics, optional-header magic, e_lfanew, header bytes) of a PE file, or a problem text."""
    p = Path(path)
    if p.is_dir():
        return "is a directory"
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            head = f.read(65536)
    except PermissionError:
        return "cannot be read (permission denied): fix its permissions"
    except OSError as e:
        return f"cannot be read ({e.strerror})"
    if size == 0:
        return "is empty"
    if head[:4] == b"\x7fELF":
        return "is a Linux library (an ELF file): run the suite on Linux, or pass the .dll built for Windows"
    if head[:2] != b"MZ" or len(head) < 0x40:
        return "is not a Windows DLL (no PE header): pass the RuntimeLibrary.dll built for this machine"
    pe = struct.unpack("<I", head[0x3C:0x40])[0]
    if pe + 26 > len(head) or head[pe:pe + 4] != b"PE\0\0":
        return "has an invalid PE header"
    machine, _sections, = struct.unpack("<HH", head[pe + 4:pe + 8])
    characteristics, magic = struct.unpack("<H", head[pe + 22:pe + 24])[0], struct.unpack("<H", head[pe + 24:pe + 26])[0]
    return machine, characteristics, magic, pe, head


def _pe_problem(path: str) -> str | None:
    header = _read_pe(path)
    if isinstance(header, str):
        return header
    machine, characteristics, magic, _pe, _head = header
    if not characteristics & 0x2000:  # IMAGE_FILE_DLL
        return "is an executable, not a DLL"
    host_name, host = platform.machine(), PE_HOST.get(platform.machine().lower())
    if host and machine != host:
        return f"is built for {PE_MACHINES.get(machine, f'machine 0x{machine:x}')}, this machine is {host_name}: pass the library built for this machine"
    if magic != 0x20B and host:
        return f"is a 32-bit DLL, this machine ({host_name}) needs a 64-bit one"
    return None


def pe_exports(path: str) -> list[str] | str:
    """Names a DLL exports (its export directory), or a text saying why they cannot be listed."""
    header = _read_pe(path)
    if isinstance(header, str):
        return header
    _machine, _chars, magic, pe, head = header
    nsections = struct.unpack("<H", head[pe + 6:pe + 8])[0]
    opt_size = struct.unpack("<H", head[pe + 20:pe + 22])[0]
    data = Path(path).read_bytes()
    opt = pe + 24
    dir_at = opt + (112 if magic == 0x20B else 96)  # the data directories follow the fixed fields
    export_rva, export_size = struct.unpack("<II", data[dir_at:dir_at + 8])
    if not export_rva:
        return []
    sections = [struct.unpack("<IIII", data[opt + opt_size + 40 * i + 8:opt + opt_size + 40 * i + 24]) for i in range(nsections)]  # vsize, rva, rawsize, rawptr

    def offset(rva: int) -> int:
        for vsize, base, raw_size, raw_ptr in sections:
            if base <= rva < base + max(vsize, raw_size):
                return raw_ptr + rva - base
        raise ValueError("a pointer outside every section")

    try:
        at = offset(export_rva)
        number_of_functions = struct.unpack("<I", data[at + 20:at + 24])[0]
        number_of_names, names_rva = struct.unpack("<II", data[at + 24:at + 32])[0], struct.unpack("<I", data[at + 32:at + 36])[0]
        names = []
        for i in range(number_of_names):
            name_at = offset(struct.unpack("<I", data[offset(names_rva) + 4 * i:offset(names_rva) + 4 * i + 4])[0])
            names.append(data[name_at:data.index(b"\0", name_at)].decode("ascii", errors="replace"))
        names += [f"#ordinal-only export {i + 1}" for i in range(max(0, number_of_functions - number_of_names))]  # NONAME exports have no name but still count
        return names
    except (ValueError, struct.error):
        return "its export table is damaged"

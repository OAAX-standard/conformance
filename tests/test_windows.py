"""The Windows-specific logic that can be tested anywhere: PE header checks and crash names (crafted bytes, no Windows needed)."""
import struct
import sys

import pytest
from conftest import WINDOWS, lib_name

from oaax_conformance import harness, libcheck
from oaax_conformance.harness import Run


def pe(machine=0x8664, characteristics=0x2022, magic=0x20B) -> bytes:
    """The first bytes of a PE file: DOS stub pointer at 0x3C, then 'PE\\0\\0', the COFF header and the start of the optional header."""
    dos = b"MZ" + bytes(0x3A) + struct.pack("<I", 0x40)
    coff = struct.pack("<HHIIIHH", machine, 0, 0, 0, 0, 240, characteristics)
    return dos + b"PE\0\0" + coff + struct.pack("<H", magic) + bytes(300)


@pytest.fixture
def x64_host(monkeypatch):
    monkeypatch.setattr(libcheck.platform, "machine", lambda: "AMD64")


@pytest.mark.parametrize("data,cause", [
    (b"", "is empty"),
    (b"just text, no header at all, but long enough to read\n" * 3, "no PE header"),
    (b"\x7fELF" + bytes(100), "Linux library"),
    (pe(characteristics=0x0022), "executable, not a DLL"),
    (pe(machine=0xAA64), "is built for aarch64"),
    (pe(machine=0x14C, magic=0x10B), "is built for x86 (32-bit)"),
])
def test_a_file_that_cannot_be_a_windows_runtime_is_named_in_one_line(tmp_path, x64_host, data, cause):
    path = tmp_path / "RuntimeLibrary.dll"
    path.write_bytes(data)
    assert cause in libcheck._pe_problem(str(path))


def test_a_64_bit_dll_for_this_machine_is_accepted(tmp_path, x64_host):
    path = tmp_path / "RuntimeLibrary.dll"
    path.write_bytes(pe())
    assert libcheck._pe_problem(str(path)) is None


def test_a_dll_given_on_linux_is_named_not_confused_with_an_elf(tmp_path):
    path = tmp_path / "RuntimeLibrary.dll"
    path.write_bytes(pe())
    if sys.platform.startswith("linux"):
        assert "Windows DLL" in libcheck.library_problem(str(path))


@pytest.mark.skipif(not WINDOWS, reason="the mock DLLs are only built on Windows")
def test_the_export_table_of_a_real_dll_is_read(build_dir):
    names = libcheck.pe_exports(str(build_dir / lib_name("mock_runtime")))
    assert sorted(names) == sorted("runtime_" + n for n in ["init", "load_models", "enqueue_input", "retrieve_output", "cleanup", "get_error", "get_version", "get_name", "get_info"])
    assert "mock_extra_symbol" in libcheck.pe_exports(str(build_dir / lib_name("mock_extra_export")))


def test_a_windows_crash_is_named_by_its_exception_code(monkeypatch):
    monkeypatch.setattr(harness, "WINDOWS", True)
    assert Run("x", 0xC0000005 - 2**32, False).signal_name == "ACCESS_VIOLATION"
    assert Run("x", 0xC0000409 - 2**32, False).signal_name.startswith("STACK_BUFFER_OVERRUN")
    assert Run("x", 0xDEADBEEF - 2**32, False).signal_name == "exception 0xDEADBEEF"
    assert Run("x", 3, False).signal_name is None  # an ordinary exit code, e.g. abort()


def test_a_missing_dependency_gets_the_windows_advice(monkeypatch):
    monkeypatch.setattr(harness, "WINDOWS", True)
    dependency = "The specified module could not be found. (Windows error 126)"
    run = Run("static", 3, False, [{"begin": "dlopen"}, {"step": "dlopen", "status": None, "str": dependency.encode().hex()}, {"end": "dlopen"}], lib="C:\\x\\RuntimeLibrary.dll")
    assert "put them next to the library or on PATH" in run._why_missing("symbols")
    own = Run("static", 3, False, [{"step": "dlopen", "status": None, "str": b"C:\\x\\RuntimeLibrary.dll: %1 is not a valid Win32 application. (Windows error 193)".hex()}, {"end": "dlopen"}],
              lib="C:\\x\\RuntimeLibrary.dll")
    assert "PATH" not in own._why_missing("symbols")


@pytest.mark.skipif(not WINDOWS, reason="the mock DLLs are only built on Windows")
def test_exports_without_a_name_still_count_as_exports(build_dir, tmp_path):
    """A .def file entry `f @5 NONAME` has no name in the export table: only NumberOfFunctions > NumberOfNames shows it (code-review finding)."""
    data = bytearray((build_dir / lib_name("mock_runtime")).read_bytes())
    pe_at = struct.unpack("<I", data[0x3C:0x40])[0]
    opt, sections, opt_size = pe_at + 24, struct.unpack("<H", data[pe_at + 6:pe_at + 8])[0], struct.unpack("<H", data[pe_at + 20:pe_at + 22])[0]
    rva = struct.unpack("<I", data[opt + 112:opt + 116])[0]
    for i in range(sections):
        vsize, base, raw_size, raw_ptr = struct.unpack("<IIII", data[opt + opt_size + 40 * i + 8:opt + opt_size + 40 * i + 24])
        if base <= rva < base + max(vsize, raw_size):
            at = raw_ptr + rva - base
    data[at + 20:at + 24] = struct.pack("<I", 11)  # 11 functions, 9 names
    patched = tmp_path / "RuntimeLibrary.dll"
    patched.write_bytes(data)
    assert len(libcheck.pe_exports(str(patched))) == 11

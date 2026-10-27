"""Reading a file a toolchain (an image we did not write) left behind.

Such a file can be a named pipe (reading blocks forever), a link to a host file or to /dev/zero, or a sparse multi-GB file (reading exhausts memory).
Every read of a file under a toolchain-controlled directory goes through here: no link is followed, nothing blocks, only regular files, only up to a limit.
"""
import json
import os
import stat
from pathlib import Path

MAX_JSON_BYTES = 10 << 20


class Unreadable(OSError):
    """Not something to read: a pipe, a link, a device."""


class TooLarge(OSError):
    """Bigger than the limit: whatever it is, it is not a log or a manifest."""


def read_limited(path, limit: int = MAX_JSON_BYTES) -> bytes:
    name = Path(path).name
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    except OSError as e:
        raise Unreadable(f"{name} cannot be opened ({e.strerror}); it may be a link") from e
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise Unreadable(f"{name} is not a regular file")
        if info.st_size > limit:
            raise TooLarge(f"{name} is {info.st_size / 1e6:.0f} MB: the suite reads at most {limit >> 20} MB of a file like this")
        chunks, total = [], 0
        while chunk := os.read(fd, min(1 << 20, limit + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:  # it grew while we read
                raise TooLarge(f"{name} is bigger than {limit >> 20} MB")
        return b"".join(chunks)
    finally:
        os.close(fd)


def read_json(path, limit: int = MAX_JSON_BYTES):
    return json.loads(read_limited(path, limit).decode("utf-8"))

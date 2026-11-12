"""Pull single members out of a large remote zip over HTTP range requests.

RLDD is 120 GB of zips. Reading the central directory remotely and fetching one
video at a time means it can be processed on a machine with far less free disk,
and only the videos actually needed cross the network.
"""
from __future__ import annotations

import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

USER_AGENT = "drowsyguard-dataset-prep"
CHUNK = 1 << 20


class NotFileData(OSError):
    """The host answered with something other than the requested bytes (often a rate-limit page)."""


def _open(url: str, start: int, end: int, timeout: int = 60):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


class HttpRangeFile:
    """Just enough of a seekable file for ``zipfile`` to read the directory."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with _open(url, 0, 0) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        with _open(self.url, self.pos, self.pos + n - 1) as r:
            data = r.read()
        self.pos += len(data)
        return data


def list_members(url: str) -> list:
    return list(zipfile.ZipFile(HttpRangeFile(url)).infolist())


def fetch_member(url: str, info: zipfile.ZipInfo, dest: Path, retries: int = 8) -> None:
    """Stream one member to ``dest``, resuming after drops and verifying its CRC."""
    if info.compress_type not in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
        raise ValueError(f"unsupported compression in {info.filename}")

    header = HttpRangeFile(url)
    header.seek(info.header_offset)
    raw = header.read(30)
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    end = start + info.compress_size - 1

    inflate = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    crc = 0
    done = 0
    failures = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        while start + done <= end:
            try:
                with _open(url, start + done, end, timeout=120) as r:
                    kind = r.headers.get("Content-Type", "")
                    if r.status != 206 or "html" in kind or "text" in kind:
                        raise NotFileData(f"got HTTP {r.status} {kind} instead of a byte range")
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        done += len(chunk)
                        data = inflate.decompress(chunk) if inflate else chunk
                        crc = zlib.crc32(data, crc)
                        out.write(data)
                        failures = 0
            except zlib.error as err:
                # Bad bytes from the host: restart this member from scratch after a pause.
                failures += 1
                if failures > retries:
                    raise RuntimeError(f"giving up on {info.filename}: {err}") from err
                out.seek(0)
                out.truncate()
                done, crc = 0, 0
                inflate = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
                time.sleep(min(300, 15 * 2**failures))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
                failures += 1
                if failures > retries:
                    raise RuntimeError(f"giving up on {info.filename}: {err}") from err
                time.sleep(min(60, 2**failures))
        if inflate:
            tail = inflate.flush()
            crc = zlib.crc32(tail, crc)
            out.write(tail)
    if crc & 0xFFFFFFFF != info.CRC:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"CRC mismatch for {info.filename}")

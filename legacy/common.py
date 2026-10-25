"""Shared helpers for the chunked reproduction of Landauer et al. (2022).

clue.json is streamed straight out of clue.zip, so the 14.5 GB file never
has to be unpacked on disk.
"""
import io
import zipfile
import zlib


def open_events(path, member='clue.json'):
    """Yield raw JSON lines (str) from clue.zip, or from a plain JSON-lines file."""
    if path.endswith('.zip'):
        z = zipfile.ZipFile(path)
        with z.open(member) as raw:
            for line in io.TextIOWrapper(raw, encoding='utf-8'):
                yield line
    else:
        with open(path, encoding='utf-8') as f:
            for line in f:
                yield line


def chunk_of(uid, n_chunks):
    """Stable (process-independent) assignment of a user to a chunk."""
    return zlib.crc32(uid.encode('utf-8')) % n_chunks

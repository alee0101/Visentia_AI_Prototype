"""Fetch only the TREX-DME OCT B-scans from the OLIVES Zenodo archive.

Why this exists: OLIVES.zip is 34 GB, Zenodo throttles single connections to
~0.35 MB/s, and the archive is a deflated zip that contains nested zips, so
individual files can't be range-read. Instead we:
  1. download the deflated bytes of the inner TREX_DME.zip member with many
     parallel HTTP range requests (one chunk file each, bounded look-ahead window),
  2. inflate those bytes as a stream, in order,
  3. parse the inner zip as a stream and save each B-scan as a downsized PNG,
  4. delete every chunk once it has been consumed.
Peak disk usage is roughly WINDOW * CHUNK bytes plus the PNG output.

Usage: python scripts/fetch_olives_trex.py --out data/raw/olives_trex [--max-chunks N]
"""
from __future__ import annotations

import argparse
import io
import os
import struct
import subprocess
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from stream_unzip import stream_unzip

URL = "https://zenodo.org/api/records/7105232/files/OLIVES.zip/content"
MEMBER = "OLIVES/TREX_DME.zip"
CHUNK = 128 * 1024 * 1024
WINDOW = 48  # max chunks on disk ahead of the consumer (~6 GB)
WORKERS = 24


def http_range(start: int, end: int, dest: Path | None = None, tries: int = 8) -> bytes | None:
    for attempt in range(tries):
        tmp = None if dest is None else dest.with_suffix(".part")
        cmd = ["curl", "-sfL", "--max-time", "1800", "-r", f"{start}-{end}", URL]
        if tmp is not None:
            cmd += ["-o", str(tmp)]
        res = subprocess.run(cmd, capture_output=True)
        want = end - start + 1
        if res.returncode == 0:
            if tmp is None and len(res.stdout) == want:
                return res.stdout
            if tmp is not None and tmp.stat().st_size == want:
                tmp.rename(dest)
                return None
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range {start}-{end} failed after {tries} tries")


def member_data_span() -> tuple[int, int]:
    """Locate the deflated bytes of MEMBER inside the outer zip (via its central directory)."""
    from remotezip import RemoteZip

    with RemoteZip(URL) as z:
        info = z.getinfo(MEMBER)
    assert info.compress_type == zlib.DEFLATED, info.compress_type
    hdr = http_range(info.header_offset, info.header_offset + 29)
    sig, *_rest = struct.unpack("<IHHHHHIIIHH", hdr)
    assert sig == 0x04034B50, hex(sig)
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    return start, info.compress_size


class ChunkFeeder:
    """Downloads chunks in parallel and yields them in order, deleting as it goes."""

    def __init__(self, start: int, size: int, tmpdir: Path, max_chunks: int | None):
        self.start, self.size, self.tmp = start, size, tmpdir
        self.n = (size + CHUNK - 1) // CHUNK
        if max_chunks:
            self.n = min(self.n, max_chunks)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.consumed = 0
        self.cv = threading.Condition()

    def _path(self, i: int) -> Path:
        return self.tmp / f"chunk_{i:05d}.bin"

    def _fetch(self, i: int) -> None:
        with self.cv:
            self.cv.wait_for(lambda: i < self.consumed + WINDOW)
        p = self._path(i)
        if not p.exists():
            a = self.start + i * CHUNK
            b = min(self.start + self.size, a + CHUNK) - 1
            http_range(a, b, dest=p)

    def __iter__(self):
        pool = ThreadPoolExecutor(WORKERS)
        futs = [pool.submit(self._fetch, i) for i in range(self.n)]
        t0 = time.time()
        for i, f in enumerate(futs):
            f.result()
            p = self._path(i)
            with open(p, "rb") as fh:
                while blk := fh.read(4 * 1024 * 1024):
                    yield blk
            p.unlink()
            with self.cv:
                self.consumed = i + 1
                self.cv.notify_all()
            mb = (i + 1) * CHUNK / 1e6
            print(f"[fetch] chunk {i + 1}/{self.n}  {mb / (time.time() - t0):.1f} MB/s avg", flush=True)
        pool.shutdown()


def inflate(blocks):
    d = zlib.decompressobj(-15)
    for b in blocks:
        out = d.decompress(b)
        if out:
            yield out
    tail = d.flush()
    if tail:
        yield tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/olives_trex")
    ap.add_argument("--max-width", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=None, help="for smoke tests")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    start, size = member_data_span()
    print(f"[fetch] {MEMBER}: {size / 1e9:.2f} GB deflated, starting at byte {start}", flush=True)
    feeder = ChunkFeeder(start, size, out.parent / "_chunks", args.max_chunks)

    saved = skipped = 0
    manifest = open(out / "manifest.tsv", "a")
    try:
        for name, _sz, chunks in stream_unzip(inflate(feeder)):
            name = name.decode("utf-8", "replace")
            data = b"".join(chunks)
            if not name.lower().endswith((".tif", ".tiff", ".png", ".jpg")):
                skipped += 1
                continue
            rel = Path(name).with_suffix(".png")
            dest = out / rel
            if dest.exists():
                saved += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            im = Image.open(io.BytesIO(data))
            orig = im.size
            im = im.convert("L")
            if im.width > args.max_width:
                h = round(im.height * args.max_width / im.width)
                im = im.resize((args.max_width, h), Image.BILINEAR)
            im.save(dest, optimize=True)
            manifest.write(f"{name}\t{orig[0]}\t{orig[1]}\t{im.width}\t{im.height}\n")
            saved += 1
            if saved % 2000 == 0:
                manifest.flush()
                print(f"[extract] {saved} images saved (last: {name})", flush=True)
    except Exception as e:  # smoke tests end mid-stream on purpose
        if args.max_chunks is None:
            raise
        print(f"[extract] stopped at chunk limit ({type(e).__name__})")
    manifest.close()
    print(f"[done] saved={saved} skipped_non_images={skipped}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    main()

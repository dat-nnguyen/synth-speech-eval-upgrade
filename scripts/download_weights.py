#!/usr/bin/env python3
"""Download the public improved SQ-AST checkpoints linked by its authors."""

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src/models/weights"
SOURCE = "https://tubcloud.tu-berlin.de/s/K9owXP3Fnj4pnJg"
DAV = "https://tubcloud.tu-berlin.de/public.php/webdav/"
# This is the public share identifier, not an account password.
AUTH = "Basic " + base64.b64encode(b"K9owXP3Fnj4pnJg:").decode()
SIZES = {dim: 344817004 for dim in ("mos", "noi", "dis", "col")}
SIZES["loud"] = 344817209


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


CHUNK_SIZE = 16 * 1024 * 1024


def download_chunk(dim, start, end, path):
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return dim, expected
    partial = path.with_suffix(".part")
    for attempt in range(1, 4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise RuntimeError(f"Oversized chunk: {partial}")
        if offset == expected:
            break
        position = start + offset
        request = urllib.request.Request(DAV + f"{dim}.pth", headers={
            "Authorization": AUTH, "Range": f"bytes={position}-{end}",
        })
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                wanted = f"bytes {position}-{end}/{SIZES[dim]}"
                if response.status != 206 or response.headers.get("Content-Range") != wanted:
                    raise RuntimeError(f"{dim}: server returned an invalid chunk range")
                with partial.open("ab" if offset else "wb") as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
            if partial.stat().st_size != expected:
                raise RuntimeError(f"Incomplete chunk: {partial}")
            break
        except (OSError, RuntimeError) as error:
            print(f"{dim}: chunk {start}, attempt {attempt}/3: {error}", flush=True)
            if attempt == 3:
                raise
    partial.replace(path)
    return dim, expected


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    plans = {}
    progress = {}
    for dim, expected in SIZES.items():
        target = DEST / f"{dim}.pth"
        if target.exists():
            if target.stat().st_size != expected:
                raise RuntimeError(f"Unexpected size; existing file preserved: {target}")
            continue
        folder = DEST / ".download_chunks" / dim
        folder.mkdir(parents=True, exist_ok=True)
        prefix = folder / "prefix.bin"
        old_partial = target.with_suffix(".pth.part")
        if old_partial.exists() and not prefix.exists():
            old_partial.replace(prefix)
        offset = prefix.stat().st_size if prefix.exists() else 0
        if offset > expected:
            raise RuntimeError(f"Oversized prefix: {prefix}")
        chunks = [(start, min(start + CHUNK_SIZE, expected) - 1,
                   folder / f"{start:012d}.bin")
                  for start in range(offset, expected, CHUNK_SIZE)]
        plans[dim] = (prefix, chunks)
        progress[dim] = offset
        print(f"{dim}: reusing {offset} downloaded bytes", flush=True)
    # Interleave files so the bandwidth is shared across all five checkpoints.
    jobs = []
    count = max((len(chunks) for _, chunks in plans.values()), default=0)
    for index in range(count):
        for dim, (_, chunks) in plans.items():
            if index < len(chunks):
                jobs.append((dim, *chunks[index]))
    reported = {dim: value * 10 // SIZES[dim] for dim, value in progress.items()}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_chunk, *job) for job in jobs]
        for future in as_completed(futures):
            dim, size = future.result()
            progress[dim] += size
            tenth = progress[dim] * 10 // SIZES[dim]
            if tenth > reported[dim]:
                print(f"{dim}: {min(100, tenth * 10)}%", flush=True)
                reported[dim] = tenth
    for dim, (prefix, chunks) in plans.items():
        assembled = DEST / f"{dim}.pth.assembling"
        with assembled.open("wb") as output:
            parts = ([prefix] if prefix.exists() else []) + [path for _, _, path in chunks]
            for part in parts:
                with part.open("rb") as stream:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
        if assembled.stat().st_size != SIZES[dim]:
            raise RuntimeError(f"Wrong assembled size: {assembled}")
        with assembled.open("rb") as stream:
            magic = stream.read(4)
        if not (magic.startswith(b"PK") or magic.startswith(b"\x80")):
            raise RuntimeError(f"Unexpected checkpoint header: {dim}: {magic!r}")
        assembled.replace(DEST / f"{dim}.pth")
        shutil.rmtree(prefix.parent)
    files = []
    for dim, expected in SIZES.items():
        target = DEST / f"{dim}.pth"
        checksum = sha256(target)
        print(f"{dim}: verified {expected} bytes, SHA256={checksum}", flush=True)
        files.append({"file": target.name, "bytes": expected, "sha256": checksum})
    manifest = {"source": SOURCE, "variant": "improved", "files": files}
    (DEST / "weights_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("All five checkpoints downloaded; weights_manifest.json saved.", flush=True)


if __name__ == "__main__":
    main()

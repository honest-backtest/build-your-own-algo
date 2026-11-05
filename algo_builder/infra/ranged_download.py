"""
Book 2, Chapter 3 — The data layer ("large downloads that survive throttling").
Used by the instrument-master cache and any large daily file.

Download in small HTTP Range pieces, retrying each piece in place, keeping a
.part file that the NEXT run resumes. Three hard-won rules:

  1. Resume only from a CLEAN part (a JSON text file never contains NUL bytes —
     NULs mean two writers punched holes in it) whose expected size still matches.
  2. If the lock heartbeat fails, another process owns the download: stop writing.
  3. Only a parse/decode error means "corrupt". A failed rename is not corruption —
     return the good data and let the next run publish the complete .part.

The URL is supplied by the caller (from an environment variable); none is hard-coded.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import requests

log = logging.getLogger("ranged_download")

RANGE_CHUNK_BYTES = 1 << 20
CHUNK_RETRIES = 8
TIMEOUT = (10, 60)
MIN_VALID_ROWS = 1000


class LostLock(Exception):
    pass


def _probe_total(s: requests.Session, url: str) -> Optional[int]:
    r = s.get(url, headers={"Range": "bytes=0-0"}, timeout=TIMEOUT)
    cr = r.headers.get("Content-Range", "")             # 'bytes 0-0/<total>'
    return int(cr.rsplit("/", 1)[-1]) if r.status_code == 206 and "/" in cr else None


def _part_is_clean(path: str) -> bool:
    with open(path, "rb") as f:
        return b"\x00" not in f.read()


def download_ranged(url: str, part: str, touch_lock: Callable[[], bool],
                    publish: Callable[[str], bool], budget_s: int = 900) -> Optional[list]:
    s = requests.Session()
    total = _probe_total(s, url)
    if not total:
        return None                                    # no Range support: caller streams instead
    total_file = part + ".total"

    resume = 0
    if os.path.exists(part):
        prev = int(Path(total_file).read_text()) if os.path.exists(total_file) else -1
        size = os.path.getsize(part)
        if prev == total and size <= total and _part_is_clean(part):
            resume = size if size == total else (size // RANGE_CHUNK_BYTES) * RANGE_CHUNK_BYTES
        else:
            for p in (part, total_file):
                if os.path.exists(p):
                    os.remove(p)
    Path(total_file).write_text(str(total))

    got, deadline = resume, time.time() + budget_s
    try:
        with open(part, "r+b" if resume else "wb") as f:
            f.seek(resume); f.truncate(resume)
            while got < total:
                if time.time() > deadline:
                    raise IOError("download budget exceeded")
                end = min(got + RANGE_CHUNK_BYTES, total) - 1
                buf = None
                for attempt in range(CHUNK_RETRIES):
                    if not touch_lock():
                        raise LostLock()
                    try:
                        r = s.get(url, headers={"Range": f"bytes={got}-{end}"}, timeout=TIMEOUT)
                        r.raise_for_status()
                        if len(r.content) == end - got + 1:
                            buf = r.content
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(min(2 ** attempt, 8))
                if buf is None:
                    raise IOError(f"chunk at {got} failed")
                f.write(buf)
                got += len(buf)
    except LostLock:
        log.warning("another process took over the download; leaving .part to it")
        return None
    except IOError as e:
        log.warning("stopped at %.1f/%.1f MB (%s); .part kept for resume", got / 1e6, total / 1e6, e)
        return None

    try:                                               # only THIS block may discard the .part
        with open(part, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) < MIN_VALID_ROWS:
            raise ValueError("payload too small")
    except (ValueError, UnicodeDecodeError) as e:
        os.remove(part); os.remove(total_file)
        log.warning("download corrupt (%s); next run restarts clean", e)
        return None

    if publish(part):
        os.remove(total_file)
    return data                                        # valid even if publish was blocked

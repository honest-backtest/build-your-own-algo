"""
Book 2, Chapter 3 — The data layer ("instrument master: one download, many readers").

Problem: at the market open a dozen scripts start at once, each wanting today's
~35 MB instrument master. If all of them download it, the endpoint throttles and
they corrupt each other's partial files. Solution:

  * one cross-process lock (O_EXCL) that records an OWNERSHIP TOKEN;
  * the holder heartbeats the lock; anyone else only waits for a live holder;
  * a failure marker gives a cooldown so nobody hammers a throttled endpoint;
  * the result is published atomically (with retries for Windows sharing errors);
  * a process holding yesterday's copy upgrades from disk, never re-downloads.

The download itself is in infra/ranged_download.py; wire it in with
  load_instrument_master(lambda: download_ranged(os.environ["INSTRUMENT_MASTER_URL"],
                         CACHE_FILE + ".part", touch_lock, publish))
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("instrument_master")

CACHE_DIR = os.environ.get("ALGO_CACHE_DIR", "./cache")
CACHE_FILE = os.path.join(CACHE_DIR, "instrument_master.json")
META_FILE = CACHE_FILE + ".meta"        # date of last good download
LOCK_FILE = CACHE_FILE + ".lock"
FAIL_FILE = CACHE_FILE + ".failed"      # epoch of last failed download

SHARE_WAIT_SECONDS = 300      # with an older cache on disk, wait this long for the holder
LOCK_STALE_SECONDS = 300      # lock untouched this long = crashed holder
FAIL_COOLDOWN_SECONDS = 600   # after a failure, nobody retries for this long
MAX_DOWNLOAD_SECONDS = 900
PUBLISH_RETRIES = 15

_LOCK_TOKEN = f"{os.getpid()}-{os.urandom(6).hex()}"


def _today() -> str:
    return date.today().isoformat()


def cache_is_fresh() -> bool:
    try:
        return Path(META_FILE).read_text().strip() == _today() and os.path.exists(CACHE_FILE)
    except OSError:
        return False


def read_cache() -> Optional[list]:
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ── lock with ownership token ────────────────────────────────────────────────
def _lock_owner() -> Optional[str]:
    try:
        return Path(LOCK_FILE).read_text(encoding="utf-8").split(" ", 1)[0].strip()
    except OSError:
        return None


def owns_lock() -> bool:
    return _lock_owner() == _LOCK_TOKEN


def acquire_lock() -> bool:
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        if os.path.exists(LOCK_FILE) and time.time() - os.path.getmtime(LOCK_FILE) > LOCK_STALE_SECONDS:
            os.remove(LOCK_FILE)                       # crashed holder
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{_LOCK_TOKEN} {time.time()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except OSError as e:
        # never download without a lock we can PROVE we own
        log.warning("lock error (%s); not downloading", e)
        return False


def release_lock() -> None:
    if owns_lock():                                    # never delete someone else's lock
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass


def touch_lock() -> bool:
    """Heartbeat; False means another process took over -> stop writing."""
    if not owns_lock():
        return False
    try:
        os.utime(LOCK_FILE, None)
    except OSError:
        pass
    return True


def lock_is_live() -> bool:
    try:
        return time.time() - os.path.getmtime(LOCK_FILE) < LOCK_STALE_SECONDS
    except OSError:
        return False


# ── failure cooldown ─────────────────────────────────────────────────────────
def recent_failure() -> bool:
    try:
        return time.time() - float(Path(FAIL_FILE).read_text()) < FAIL_COOLDOWN_SECONDS
    except (OSError, ValueError):
        return False


def mark_failure() -> None:
    with open(FAIL_FILE, "w") as f:
        f.write(str(time.time()))


def clear_failure() -> None:
    try:
        os.remove(FAIL_FILE)
    except OSError:
        pass


# ── atomic publish ───────────────────────────────────────────────────────────
def publish(src: str) -> bool:
    """os.replace fails on Windows while a reader has the file open: that is a
    transient sharing conflict, NOT corruption — retry, then give up gracefully."""
    for i in range(PUBLISH_RETRIES):
        try:
            os.replace(src, CACHE_FILE)
            with open(META_FILE, "w") as f:
                f.write(_today())
            return True
        except PermissionError:
            time.sleep(1)
    return False


# ── the coordination loop ────────────────────────────────────────────────────
def refresh_under_lock(download: Callable[[], Optional[list]],
                       ignore_cooldown: bool = False) -> Optional[list]:
    """ONE process downloads; all others wait for it and read what it published.
    Returns None -> caller falls back to the older cache."""
    have_cache = os.path.exists(CACHE_FILE)
    deadline = time.time() + (SHARE_WAIT_SECONDS if have_cache else MAX_DOWNLOAD_SECONDS)
    while True:
        if cache_is_fresh():
            return read_cache()
        if have_cache and not ignore_cooldown and recent_failure():
            return None
        if acquire_lock():
            try:
                if cache_is_fresh():                   # published while we acquired
                    return read_cache()
                data = download()                      # must call touch_lock() as it goes
                if data is None:
                    mark_failure()                     # others back off for a while
                else:
                    clear_failure()
                return data
            finally:
                release_lock()
        if time.time() > deadline:
            return None
        time.sleep(2)                                  # a live holder exists: wait, never race


_MEM, _MEM_DATE = None, None


def load_instrument_master(download: Callable[[], Optional[list]]) -> list:
    global _MEM, _MEM_DATE
    if _MEM is not None:
        if _MEM_DATE == _today():
            return _MEM
        if cache_is_fresh() and (data := read_cache()):   # upgrade from disk, no download
            _MEM, _MEM_DATE = data, _today()
            return data
        if lock_is_live() or recent_failure():
            return _MEM
    if cache_is_fresh() and (data := read_cache()):
        _MEM, _MEM_DATE = data, _today()
        return data
    data = refresh_under_lock(download) or read_cache()   # older copy is still usable
    if not data:
        raise RuntimeError("instrument master unavailable and no cache exists")
    _MEM, _MEM_DATE = data, _today()
    return data

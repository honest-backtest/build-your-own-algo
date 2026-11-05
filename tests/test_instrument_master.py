import os

from algo_builder.infra import instrument_master_lock as im


def _isolate(monkeypatch, tmp_path):
    d = str(tmp_path)
    monkeypatch.setattr(im, "CACHE_DIR", d)
    for name, suffix in (("CACHE_FILE", ""), ("META_FILE", ".meta"), ("LOCK_FILE", ".lock"),
                         ("FAIL_FILE", ".failed")):
        monkeypatch.setattr(im, name, os.path.join(d, "instrument_master.json" + suffix))


def test_lock_is_exclusive_and_only_the_owner_releases_it(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert im.acquire_lock() is True
    monkeypatch.setattr(im, "_LOCK_TOKEN", "someone-else")
    assert im.acquire_lock() is False          # held by the first process
    im.release_lock()                           # not ours: must not delete it
    assert os.path.exists(im.LOCK_FILE)


def test_heartbeat_fails_after_a_takeover(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert im.acquire_lock() and im.touch_lock()
    with open(im.LOCK_FILE, "w") as f:          # another process replaced the lock
        f.write("other-token 0")
    assert im.touch_lock() is False


def test_one_download_is_shared_and_failure_sets_a_cooldown(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    calls = []

    def good():
        calls.append(1)
        tmp = im.CACHE_FILE + ".part"
        with open(tmp, "w") as f:
            f.write('[{"symbol": "X"}]')
        im.publish(tmp)
        return [{"symbol": "X"}]

    assert im.refresh_under_lock(good) == [{"symbol": "X"}]
    assert im.refresh_under_lock(good) == [{"symbol": "X"}] and len(calls) == 1   # read, not refetched

    os.remove(im.META_FILE)                    # stale again; this time the download fails
    assert im.refresh_under_lock(lambda: None) is None and im.recent_failure()
    assert im.refresh_under_lock(good) is None and len(calls) == 1               # cooldown honoured

import pytest


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Every test gets its own cache, state and control folders."""
    for name in ("ALGO_CACHE_DIR", "ALGO_STATE_DIR", "ALGO_CONTROL_DIR", "ALGO_DATA_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    yield

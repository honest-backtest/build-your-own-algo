"""Front matter: configuration from the environment only."""
import pytest

from algo_builder.common import config_env as ce


def test_env_file_never_overrides_and_summary_never_shows_values(monkeypatch, tmp_path):
    for n in ce.SECRET_NAMES:
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("BROKER_CLIENT_ID", "from-shell")
    f = tmp_path / ".env"
    f.write_text("# comment\nBROKER_API_KEY='abc123'\nBROKER_CLIENT_ID=from-file\nALERT_WEBHOOK_URL=\n")
    assert ce.load_env_file(f) == ["BROKER_API_KEY"]
    assert ce.require_env("BROKER_API_KEY") == "abc123"
    s = ce.config_summary()
    assert s["BROKER_API_KEY"] == "set" and s["ALERT_WEBHOOK_URL"] == "NOT SET"
    assert "abc123" not in str(s) and "from-shell" not in str(s)
    monkeypatch.delenv("BROKER_TOTP_SEED", raising=False)
    with pytest.raises(RuntimeError):
        ce.require_env("BROKER_TOTP_SEED")
    monkeypatch.delenv("BROKER_API_KEY")

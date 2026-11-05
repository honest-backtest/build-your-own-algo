"""
Book 2, front matter ("Who this book is for and what you need").

Secrets and machine-specific paths are read from environment variables ONLY.
Nothing in this repository contains a real key, token, account id or chat id.
Set them in your shell, or in a local .env file that git ignores.
"""
from __future__ import annotations

import os
from pathlib import Path

# Broker credentials: read only by YOUR concrete BrokerAdapter, never by strategy code.
BROKER_API_KEY_ENV = "BROKER_API_KEY"          # placeholder names
BROKER_CLIENT_ID_ENV = "BROKER_CLIENT_ID"
BROKER_TOTP_SEED_ENV = "BROKER_TOTP_SEED"
INSTRUMENT_MASTER_URL_ENV = "INSTRUMENT_MASTER_URL"   # your broker documents it
ALERT_WEBHOOK_URL_ENV = "ALERT_WEBHOOK_URL"           # optional (Chapter 28)
SECRET_NAMES = (BROKER_API_KEY_ENV, BROKER_CLIENT_ID_ENV, BROKER_TOTP_SEED_ENV,
                INSTRUMENT_MASTER_URL_ENV, ALERT_WEBHOOK_URL_ENV)

# Folders: relative defaults, resolved against the folder a job starts in.
DATA_DIR = Path(os.environ.get("ALGO_DATA_DIR", "./data"))
CACHE_DIR = Path(os.environ.get("ALGO_CACHE_DIR", "./cache"))
STATE_DIR = Path(os.environ.get("ALGO_STATE_DIR", "./state"))
CONTROL_DIR = Path(os.environ.get("ALGO_CONTROL_DIR", "./control"))


def require_env(name: str) -> str:
    """Read a mandatory setting; fail loudly instead of falling back to a literal."""
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"environment variable {name} is not set")
    return val


def load_env_file(path: str | Path = ".env") -> list[str]:
    """Load NAME=value lines from a local file into the environment, for jobs a
    scheduler starts without your shell's settings. Values already set win.
    Returns the names loaded, never the values."""
    p, loaded = Path(path), []
    if not p.exists():
        return loaded
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (s.strip() for s in line.split("=", 1))
        if value and name not in os.environ:
            os.environ[name] = value.strip("'\"")
            loaded.append(name)
    return loaded


def config_summary() -> dict[str, str]:
    """What a job may safely log at start-up: which secrets are set, and the
    resolved folders. Values of secrets are never included."""
    out = {n: ("set" if os.environ.get(n, "").strip() else "NOT SET") for n in SECRET_NAMES}
    for label, d in (("data", DATA_DIR), ("cache", CACHE_DIR), ("state", STATE_DIR), ("control", CONTROL_DIR)):
        out[f"{label}_dir"] = str(d.resolve())
    return out

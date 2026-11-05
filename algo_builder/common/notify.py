"""
Book 2, Chapter 28 — Scheduling, logging and alerts ("fail loudly").

One place that sends alerts. The transport is pluggable: pass any callable that
delivers a message (a messaging app, e-mail, a webhook) and returns True on
success. An alert that cannot be delivered is written to the log at ERROR, with
its text, and never raises: a failed alert must not break a trading job.
Whether a live system may START without a working transport is decided once, at
start-up, by `transport_ready`.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import requests

log = logging.getLogger("notify")

Transport = Callable[[str, bool], bool]          # (text, quiet) -> delivered?


def webhook_transport(url_env: str = "ALERT_WEBHOOK_URL") -> Optional[Transport]:
    """A generic JSON-over-HTTPS transport. The URL (which may embed a token) is read
    from the environment and never written in code. Returns None if it is not set."""
    url = os.environ.get(url_env, "").strip()
    if not url:
        return None

    def send(text: str, quiet: bool) -> bool:
        r = requests.post(url, json={"text": text[:4000], "quiet": quiet}, timeout=10)
        return r.ok

    return send


def send_alert(text: str, quiet: bool = False, transport: Optional[Transport] = None,
               tries: int = 2) -> bool:
    """True if delivered. Otherwise the text goes to the log at ERROR, so that an
    alert nobody received is at least in the file someone will read."""
    transport = transport or webhook_transport()
    if transport is None:
        log.error("ALERT (no transport configured): %s", text)
        return False
    for _ in range(tries):
        try:
            if transport(text, quiet):
                return True
        except Exception as e:                   # alerts must never break trading
            log.warning("alert attempt failed: %s", e)
    log.error("ALERT (not delivered): %s", text)
    return False


def transport_ready(transport: Optional[Transport] = None) -> bool:
    """Start-up check: is there a transport, and does a quiet test message arrive?
    A live system that cannot alert should not open new positions."""
    transport = transport or webhook_transport()
    if transport is None:
        return False
    try:
        return bool(transport("start-up check", True))
    except Exception:
        return False

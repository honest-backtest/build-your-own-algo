"""
Fail if anything that looks like a credential, personal identifier, machine path
or URL appears in the repository. Run in CI on every push, and before every commit.

    python tools/scan_secrets.py              # the whole repository
    python tools/scan_secrets.py FILE...      # named files, e.g. manuscript chapters
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
TEXT = {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".cfg", ".ini", ".json", ".example", ""}
NAMED = {".env.example", ".gitignore", ".pre-commit-config.yaml"}

PATTERNS = {
    "URL": re.compile(r"https?://"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "assigned secret": re.compile(r"(?i)(api[_-]?key|secret|token|password|totp)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
    # 32+ characters mixing letters and digits, as real keys and tokens do
    "long token": re.compile(r"\b(?=[\w\-]*\d)(?=[\w\-]*[A-Za-z])[\w\-]{32,}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "phone": re.compile(r"(?<![\d.])(?:\+91[\s-]?)?[6-9]\d{9}(?![\d.])"),
    "machine path": re.compile(r"(?i)\b[a-z]:\\\\|/home/\w|/Users/\w"),
}
# Lines allowed to match, for a stated reason (keep this list short).
ALLOW = {
    ("URL", ".pre-commit-config.yaml"): "the public gitleaks hook repository, not a data or broker endpoint",
}


# The repository's own public address may appear (README, pyproject); nothing else.
OWN_REPO = "https://github.com/honest-backtest/build-your-own-algo"

SNAKE_CASE = re.compile(r"[a-z]+(_[a-z0-9]+){3,}")      # long identifiers, e.g. test names


def _real_hit(label: str, pat: re.Pattern, line: str) -> bool:
    if label == "URL":
        return bool(pat.search(line.replace(OWN_REPO, "")))
    if label != "long token":
        return bool(pat.search(line))
    return any(not SNAKE_CASE.fullmatch(m.group(0)) for m in pat.finditer(line))


def scan() -> list[str]:
    hits = []
    for path in REPO.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts) or not path.is_file():
            continue
        if (path.suffix not in TEXT and path.name not in NAMED) or path.resolve() == Path(__file__).resolve():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for label, pat in PATTERNS.items():
                if _real_hit(label, pat, line) and (label, path.name) not in ALLOW:
                    hits.append(f"{path.relative_to(REPO)}:{lineno}: {label}: {line.strip()[:80]}")
    return hits


def scan_files(paths: list[str]) -> list[str]:
    """Scan named files outside the repository, such as manuscript chapters."""
    hits = []
    for name in paths:
        for lineno, line in enumerate(Path(name).read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for label, pat in PATTERNS.items():
                if _real_hit(label, pat, line):
                    hits.append(f"{name}:{lineno}: {label}: {line.strip()[:80]}")
    return hits


if __name__ == "__main__":
    found = scan_files(sys.argv[1:]) if len(sys.argv) > 1 else scan()
    for h in found:
        print(h)
    print(f"{len(found)} possible secrets or identifiers found")
    sys.exit(1 if found else 0)

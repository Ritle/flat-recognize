"""Service build identity, resolved once when the demo starts."""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import subprocess

SERVICE_VERSION = "0.1.0"
MOSCOW = timezone(timedelta(hours=3))


def parse_updated_at(value):
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return date if date.tzinfo is not None else None
    except (ValueError, AttributeError):
        return None


def load_service_info(root: Path, environ=None):
    env = os.environ if environ is None else environ
    revision, committed_at = None, None
    # Do not accidentally identify a parent repository in an exported source tree.
    if (root / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%h%n%cI"], cwd=root,
                capture_output=True, text=True, encoding="utf-8", timeout=2, check=True,
            )
            lines = result.stdout.strip().splitlines()
            if len(lines) == 2 and re.fullmatch(r"[0-9a-f]{7,40}", lines[0]):
                revision = lines[0]
                committed_at = lines[1]
        except (OSError, subprocess.SubprocessError):
            pass
    version = env.get("FLAT_RECO_SERVICE_VERSION", "").strip() or SERVICE_VERSION
    if revision and not env.get("FLAT_RECO_SERVICE_VERSION", "").strip():
        version += f"+{revision}"
    # An invalid explicit date stays unknown instead of falling back to a
    # different build's timestamp or pretending the process start was an update.
    date = parse_updated_at(env.get("FLAT_RECO_UPDATED_AT", committed_at))
    return {"version": version, "revision": revision,
            "updated_at": date.isoformat() if date else None}


def format_updated_at(value):
    date = parse_updated_at(value)
    return date.astimezone(MOSCOW).strftime("%d.%m.%Y %H:%M МСК") if date else "не указана"

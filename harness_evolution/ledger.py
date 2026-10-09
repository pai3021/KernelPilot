"""Small JSONL ledger and monotonic H0/H1 version marker."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def read_version(path: Path) -> str:
    if not path.is_file():
        return "H0"
    return str(json.loads(path.read_text())["version"])


def write_version(path: Path, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": version}, indent=2) + "\n")


def next_version(version: str) -> str:
    if not version.startswith("H") or not version[1:].isdigit():
        raise ValueError(f"invalid harness version: {version}")
    return f"H{int(version[1:]) + 1}"


def append_ledger(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

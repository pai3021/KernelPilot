"""Persistent stop and audit records for a managed remote benchmark scope."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .budget import BUDGET_FILE_ENV


class RemoteResourceBlocked(RuntimeError):
    """The controller must inspect this scope before any further dispatch."""


def block_path(ledger: Path) -> Path:
    return ledger.with_suffix(ledger.suffix + ".remote-block.json")


def read_block(ledger: Path) -> dict[str, Any] | None:
    path = block_path(ledger)
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected object")
        return value
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        # An incomplete/corrupt stop record must never silently reopen dispatch.
        return {"reason": "Unreadable remote stop record", "path": str(path)}


def ensure_not_blocked() -> None:
    raw = os.environ.get(BUDGET_FILE_ENV)
    if raw and (record := read_block(Path(raw))) is not None:
        raise RemoteResourceBlocked(
            f"Remote evaluation is blocked: {record.get('reason', 'infrastructure fault')}. "
            "Stop this branch; the controller must inspect the stop record."
        )


def record_event(event: str, metadata: Mapping[str, Any]) -> None:
    raw = os.environ.get(BUDGET_FILE_ENV)
    if not raw:
        return
    path = Path(raw).with_suffix(Path(raw).suffix + ".remote-events.jsonl")
    line = json.dumps({**metadata, "at": datetime.now(timezone.utc).isoformat(),
                       "event": event}, allow_nan=False) + "\n"
    # Each append is one small write; no old event is rewritten or refunded.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def block_remote(reason: str, metadata: Mapping[str, Any]) -> None:
    raw = os.environ.get(BUDGET_FILE_ENV)
    if not raw:
        return
    path = block_path(Path(raw))
    value = {**metadata, "at": datetime.now(timezone.utc).isoformat(), "reason": reason}
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError:
        pass  # Keep the first cause, including after a worker/process restart.
    record_event("REMOTE_BLOCKED", value)

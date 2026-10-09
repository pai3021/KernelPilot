"""Immutable-on-write candidate witness captured immediately before evaluation."""
from __future__ import annotations
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def capture_evaluated_candidate(workspace: Path) -> dict[str, str]:
    """Copy the exact kernel about to be packed and bind it to a SHA-256."""
    source = workspace / "solution" / "kernel.py"
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    captured_at = datetime.now(timezone.utc).isoformat()
    directory = workspace / ".ako" / "evaluated-candidates"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / f"{digest}.kernel.py"
    if not snapshot.exists():
        shutil.copy2(source, snapshot)
    record = {"candidate_sha256": digest, "artifact_path": str(snapshot.resolve()), "evaluation_timestamp": captured_at}
    (workspace / ".ako" / "evaluated-candidate.json").write_text(json.dumps(record, indent=2) + "\n")
    return record

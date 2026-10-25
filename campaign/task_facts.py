"""Build read-only, source-bound context without executing benchmark code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def build_task_facts(
    *,
    task_id: str,
    task_source: Path,
    parent_source: Path,
    environment: Mapping[str, Any],
    evaluation_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Carry complete task/parent sources and explicit caller-verified policy.

    Shape and dtype may depend on arbitrary Python expressions. Preserve the
    source instead of executing get_inputs() or guessing its runtime values.
    Hardware and numerical settings are supplied by the execution owner.
    """
    if not task_id.strip():
        raise ValueError("task_id must not be empty")

    def witness(path: Path) -> dict[str, str]:
        payload = Path(path).read_bytes()
        source = payload.decode("utf-8")
        if not source.strip():
            raise ValueError(f"Empty planning source: {path}")
        return {"source": source, "sha256": hashlib.sha256(payload).hexdigest()}

    facts = {
        "task_id": task_id,
        "task": witness(task_source),
        "parent_candidate": witness(parent_source),
        "environment": dict(environment),
        "evaluation_policy": dict(evaluation_policy),
    }
    # Fail before any paid agent call if caller metadata cannot be persisted.
    return json.loads(json.dumps(facts, allow_nan=False))

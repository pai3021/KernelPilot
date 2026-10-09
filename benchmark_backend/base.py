"""Small, benchmark-agnostic execution-backend contract.

The value crossing the existing harness seam remains the adapter's normalized
``{definition: {uuid: result}}`` mapping.  ``BenchmarkExecutionResult`` is an
envelope for direct backend clients; it intentionally does not replace that
per-workload schema.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INFRA_FAILURE = "INFRA_FAILURE"
    CANDIDATE_FAILURE = "CANDIDATE_FAILURE"
    CORRECTNESS_FAILURE = "CORRECTNESS_FAILURE"
    TIMEOUT = "TIMEOUT"


@dataclass(slots=True)
class BenchmarkExecutionResult:
    """A direct-client summary plus the adapter-normalized workload results."""

    status: ExecutionStatus
    correct: bool | None
    reference_latency_ms: float | None = None
    candidate_latency_ms: float | None = None
    speedup_factor: float | None = None
    error_log: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    results: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        results: dict[str, dict[str, dict[str, Any]]],
        *,
        metadata: Mapping[str, Any] | None = None,
        error_log: str = "",
    ) -> "BenchmarkExecutionResult":
        items = [item for traces in results.values() for item in traces.values()]
        if not items:
            return cls(ExecutionStatus.INFRA_FAILURE, None, error_log="Evaluator returned no workloads", metadata=dict(metadata or {}), results=results)
        if all(item.get("status") == "PASSED" for item in items):
            status, correct = ExecutionStatus.SUCCESS, True
        elif any(item.get("status") == "TIMEOUT" for item in items):
            status, correct = ExecutionStatus.TIMEOUT, None
        elif any(item.get("status") == "INCORRECT_NUMERICAL" for item in items):
            status, correct = ExecutionStatus.CORRECTNESS_FAILURE, False
        else:
            status, correct = ExecutionStatus.CANDIDATE_FAILURE, False
        first = items[0]
        return cls(
            status=status,
            correct=correct,
            reference_latency_ms=first.get("reference_latency_ms"),
            candidate_latency_ms=first.get("latency_ms"),
            speedup_factor=first.get("speedup_factor"),
            error_log=error_log or str(first.get("error_log", "")),
            metadata=dict(metadata or {}),
            results=results,
        )


class BenchmarkBackend(ABC):
    """Evaluator API; implementations must never expose arbitrary shell access."""

    @abstractmethod
    def evaluate_blob(
        self,
        candidate_blob: str,
        task_ids: Sequence[str],
        config: Mapping[str, Any],
        *,
        capture_logs: bool = False,
    ) -> BenchmarkExecutionResult:
        """Evaluate a packed candidate against pre-resolved workload IDs."""

    def evaluate(
        self,
        task: str | Sequence[str],
        candidate_path: str | Path,
        config: Mapping[str, Any],
    ) -> BenchmarkExecutionResult:
        """Evaluate an already-packed candidate JSON file (direct-client API)."""
        task_ids = [task] if isinstance(task, str) else list(task)
        return self.evaluate_blob(Path(candidate_path).read_text(), task_ids, config)


def candidate_definition(candidate_blob: str) -> str:
    """Read only the definition required to render normalized transport failures."""
    try:
        return str(json.loads(candidate_blob).get("definition") or "remote-evaluation")
    except (TypeError, json.JSONDecodeError):
        return "remote-evaluation"


def failure_results(
    candidate_blob: str,
    task_ids: Sequence[str],
    *,
    status: ExecutionStatus,
    error_log: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Render transport failures in the existing per-workload result shape."""
    definition = candidate_definition(candidate_blob)
    item_status = status.value
    return {
        definition: {
            task_id: {
                "status": item_status,
                "solution": "remote-candidate",
                "axes": {},
                "max_abs_error": "NaN",
                "max_rel_error": "NaN",
                "error_log": error_log,
                "metadata": dict(metadata or {}),
            }
            for task_id in task_ids
        }
    }

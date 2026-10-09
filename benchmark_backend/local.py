"""The existing in-process KernelBench evaluator, behind the backend contract."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import BenchmarkBackend, BenchmarkExecutionResult, ExecutionStatus, failure_results


class LocalBenchmarkBackend(BenchmarkBackend):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def evaluate_blob(
        self, candidate_blob: str, task_ids: Sequence[str], config: Mapping[str, Any], *, capture_logs: bool = False,
    ) -> BenchmarkExecutionResult:
        from scripts import benchmark_adapter as adapter

        try:
            results = adapter.run(
                candidate_blob, list(task_ids), dict(config), dataset_path=self.dataset_path,
                capture_logs=capture_logs,
            )
            return BenchmarkExecutionResult.from_results(
                results, metadata={"backend": "local", "dataset_path": self.dataset_path},
            )
        except Exception as exc:
            message = f"Local evaluator failure: {type(exc).__name__}: {exc}"
            results = failure_results(candidate_blob, task_ids, status=ExecutionStatus.INFRA_FAILURE, error_log=message,
                                      metadata={"backend": "local"})
            return BenchmarkExecutionResult(ExecutionStatus.INFRA_FAILURE, None, error_log=message,
                                            metadata={"backend": "local"}, results=results)

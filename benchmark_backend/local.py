"""The existing in-process KernelBench evaluator, behind the backend contract."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import BenchmarkBackend, BenchmarkExecutionResult, ExecutionStatus, failure_results


class LocalBenchmarkBackend(BenchmarkBackend):
    def __init__(self, dataset_path: str):
        # 创建后端时只保存数据集位置，尚未执行评测。
        self.dataset_path = dataset_path

    def evaluate_blob(
        self, candidate_blob: str, task_ids: Sequence[str], config: Mapping[str, Any], *, capture_logs: bool = False,
    ) -> BenchmarkExecutionResult:
        # 具体的任务加载、正确性检查和计时交给共享 adapter；导入位于 try 外。
        from scripts import benchmark_adapter as adapter

        try:
            results = adapter.run(
                candidate_blob, list(task_ids), dict(config), dataset_path=self.dataset_path,
                capture_logs=capture_logs,
            )
            # adapter 正常返回也可能包含候选失败；统一结果对象根据报告归类。
            return BenchmarkExecutionResult.from_results(
                results, metadata={"backend": "local", "dataset_path": self.dataset_path},
            )
        except Exception as exc:
            # 调用向外抛出异常时，返回基础设施失败，并保留标准 workload 结果结构。
            message = f"Local evaluator failure: {type(exc).__name__}: {exc}"
            results = failure_results(candidate_blob, task_ids, status=ExecutionStatus.INFRA_FAILURE, error_log=message,
                                      metadata={"backend": "local"})
            return BenchmarkExecutionResult(ExecutionStatus.INFRA_FAILURE, None, error_log=message,
                                            metadata={"backend": "local"}, results=results)

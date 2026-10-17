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

# 评测执行的结果类别
class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INFRA_FAILURE = "INFRA_FAILURE"  # 评测基础设施或执行环境报错，例如传输、远程环境或评测器异常
    CANDIDATE_FAILURE = "CANDIDATE_FAILURE"  # 候选代码编译或运行失败
    CORRECTNESS_FAILURE = "CORRECTNESS_FAILURE"  # 正确性没通过
    TIMEOUT = "TIMEOUT"     # 评测执行超时

# 一次后端评测的概况和详细结果放在同一个对象里，供调用方读取
@dataclass(slots=True)
class BenchmarkExecutionResult:
    """A direct-client summary plus the adapter-normalized workload results."""

    status: ExecutionStatus  # 后端评测状态，与 Agent 进程的 completed / failed 分别记录
    correct: bool | None  # 正确性概况；未能确定时为 None，失败分类也可能记为 False
    reference_latency_ms: float | None = None  # 参考实现耗时，单位毫秒
    candidate_latency_ms: float | None = None  # 候选实现耗时，单位毫秒
    speedup_factor: float | None = None  # 加速比
    error_log: str = ""  # 错误日志
    metadata: dict[str, Any] = field(default_factory=dict)  # 后端、远程目录、GPU 等补充信息
    results: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)  # {definition: {workload_id: 详细结果}}
    # 从详细结果生成这个对象
    @classmethod
    def from_results(
        cls,
        results: dict[str, dict[str, dict[str, Any]]],
        *,
        metadata: Mapping[str, Any] | None = None,
        error_log: str = "",
    ) -> "BenchmarkExecutionResult":
        # 总体状态参考全部 workload；耗时和加速比只提取第一项，不计算多任务平均成绩。
        items = [item for traces in results.values() for item in traces.values()]
        if not items:
            return cls(ExecutionStatus.INFRA_FAILURE, None, error_log="Evaluator returned no workloads", metadata=dict(metadata or {}), results=results)
        if all(item.get("status") == "PASSED" for item in items):
            status, correct = ExecutionStatus.SUCCESS, True
        # 混合失败时，超时的判断优先于数值错误，其余失败归为候选失败。
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
        # 子类实现具体执行；输入为打包后的 JSON 文本、workload ID 和本次评测参数。

    def evaluate(
        self,
        task: str | Sequence[str],
        candidate_path: str | Path,
        config: Mapping[str, Any],
    ) -> BenchmarkExecutionResult:
        """Evaluate an already-packed candidate JSON file (direct-client API)."""
        # candidate_path 指向已打包的 JSON 文件；这里仅读取，不负责打包 kernel.py。
        task_ids = [task] if isinstance(task, str) else list(task)
        return self.evaluate_blob(Path(candidate_path).read_text(), task_ids, config)


def candidate_definition(candidate_blob: str) -> str:
    """Read only the definition required to render normalized transport failures."""
    # 只取结果字典需要的定义名称；候选完整格式的校验由评测适配器负责。
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
    # 没有正常评测报告时，也为每个请求的 workload 返回同样结构的失败记录。
    # 返回的是详细结果字典；外层 BenchmarkExecutionResult 由调用方构造。
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

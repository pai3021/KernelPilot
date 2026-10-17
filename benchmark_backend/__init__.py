"""Benchmark execution backends.

The harness talks to this package instead of teaching a coding runtime about
SSH, GPUs, or a particular evaluator location.
"""

# 导出统一接口及具体后端，供调用方显式选择；这里没有运行时注册表或评测调用。
from .base import (
    BenchmarkBackend,
    BenchmarkExecutionResult,
    ExecutionStatus,
    failure_results,
)
from .local import LocalBenchmarkBackend
from .ssh_remote import RemoteSSHBenchmarkBackend, RemoteSSHConfig

__all__ = [
    "BenchmarkBackend",
    "BenchmarkExecutionResult",
    "ExecutionStatus",
    "LocalBenchmarkBackend",
    "RemoteSSHBenchmarkBackend",
    "RemoteSSHConfig",
    "failure_results",
]

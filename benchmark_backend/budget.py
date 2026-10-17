"""Harness-level formal benchmark budget guard for an isolated branch."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from contextlib import contextmanager
from typing import Any


BUDGET_FILE_ENV = "EXPERIMENTPILOT_BENCHMARK_BUDGET_FILE"  # 分支预算账本的路径
INVOCATION_TOKEN_ENV = "EXPERIMENTPILOT_BENCHMARK_INVOCATION_TOKEN"  # 本次调用已扣额度的凭据


class BenchmarkBudgetExhausted(RuntimeError):
    """Raised before a second formal remote evaluation can reach a GPU."""


def _exhausted_message(data: dict[str, Any]) -> str:
    return (
        f"Benchmark budget exhausted: {data['benchmark_used']}/{data['benchmark_limit']}. "
        "Do not run another benchmark in this branch. Proceed using the existing result."
    )

# 锁：防止两个进程同时抢到最后一个名额
@contextmanager
def _locked(path: Path):
    """Use an exclusive sidecar lock around the small JSON ledger."""
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + 10
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out acquiring benchmark budget lock: {lock}")
            time.sleep(0.02)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass

# 创建预算账本
def initialize_budget(path: Path, *, limit: int = 1) -> None:
    # 用于创建新账本；本函数会覆盖目标文件，恢复历史预算应使用 read_budget。
    if limit < 1:
        raise ValueError("benchmark limit must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"benchmark_limit": limit, "benchmark_used": 0, "issued_tokens": []}, indent=2) + "\n")

# 读取，并检查这些字段的基本类型
def read_budget(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("benchmark_limit"), int) or not isinstance(data.get("benchmark_used"), int):
        raise ValueError(f"Malformed benchmark budget ledger: {path}")
    if not isinstance(data.get("issued_tokens"), list):
        raise ValueError(f"Malformed benchmark budget token ledger: {path}")
    return data

# 临时文件替换：避免账本只写了一半
def _write_budget(path: Path, data: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)

# 扣除一次额度，返回一个 token
def reserve_budget(path: Path) -> str:
    """Consume one formal evaluation and return its process-scoped token."""
    # 锁内完成读取、检查、扣除、写回；后续评测失败也不会在这里自动退回额度。
    with _locked(path):
        data = read_budget(path)
        if data["benchmark_used"] >= data["benchmark_limit"]:
            raise BenchmarkBudgetExhausted(_exhausted_message(data))
        token = uuid.uuid4().hex
        data["benchmark_used"] += 1
        data["issued_tokens"].append(token)
        _write_budget(path, data)
    return token

# 在正式入口申请一次额度
def begin_benchmark_scope_from_environment() -> None:
    """Reserve one run at the official runner seam, if this is a branch."""
    # 正式 runner 为一次完整比较申请额度；参考和候选的内部调用共用这个 token。
    raw_path = os.environ.get(BUDGET_FILE_ENV)
    if raw_path and not os.environ.get(INVOCATION_TOKEN_ENV):
        os.environ[INVOCATION_TOKEN_ENV] = reserve_budget(Path(raw_path))

# 在后端入口再次检查
def ensure_evaluation_authorized() -> None:
    """Guard the evaluator-facing seam, including direct runner bypasses."""
    raw_path = os.environ.get(BUDGET_FILE_ENV)
    if not raw_path:
        # 未传入预算账本时，不启用这层分支预算检查。
        return
    path = Path(raw_path)
    token = os.environ.get(INVOCATION_TOKEN_ENV)
    if token:
        # 已签发的 token 可以在同一次调用范围内复用，不会逐次后端调用扣除额度。
        if token in read_budget(path)["issued_tokens"]:
            return
        raise RuntimeError("Benchmark invocation token is not present in the branch budget ledger")
    # 带账本但没有 token 的直接后端调用，也必须先申请额度。
    os.environ[INVOCATION_TOKEN_ENV] = reserve_budget(path)

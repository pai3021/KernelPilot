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


BUDGET_FILE_ENV = "EXPERIMENTPILOT_BENCHMARK_BUDGET_FILE"
INVOCATION_TOKEN_ENV = "EXPERIMENTPILOT_BENCHMARK_INVOCATION_TOKEN"


class BenchmarkBudgetExhausted(RuntimeError):
    """Raised before a second formal remote evaluation can reach a GPU."""


def _exhausted_message(data: dict[str, Any]) -> str:
    return (
        f"Benchmark budget exhausted: {data['benchmark_used']}/{data['benchmark_limit']}. "
        "Do not run another benchmark in this branch. Proceed using the existing result."
    )


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


def initialize_budget(path: Path, *, limit: int = 1) -> None:
    if limit < 1:
        raise ValueError("benchmark limit must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"benchmark_limit": limit, "benchmark_used": 0, "issued_tokens": []}, indent=2) + "\n")


def read_budget(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("benchmark_limit"), int) or not isinstance(data.get("benchmark_used"), int):
        raise ValueError(f"Malformed benchmark budget ledger: {path}")
    if not isinstance(data.get("issued_tokens"), list):
        raise ValueError(f"Malformed benchmark budget token ledger: {path}")
    return data


def _write_budget(path: Path, data: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def reserve_budget(path: Path) -> str:
    """Consume one formal evaluation and return its process-scoped token."""
    with _locked(path):
        data = read_budget(path)
        if data["benchmark_used"] >= data["benchmark_limit"]:
            raise BenchmarkBudgetExhausted(_exhausted_message(data))
        token = uuid.uuid4().hex
        data["benchmark_used"] += 1
        data["issued_tokens"].append(token)
        _write_budget(path, data)
    return token


def begin_benchmark_scope_from_environment() -> None:
    """Reserve one run at the official runner seam, if this is a branch."""
    raw_path = os.environ.get(BUDGET_FILE_ENV)
    if raw_path and not os.environ.get(INVOCATION_TOKEN_ENV):
        os.environ[INVOCATION_TOKEN_ENV] = reserve_budget(Path(raw_path))


def ensure_evaluation_authorized() -> None:
    """Guard the evaluator-facing seam, including direct runner bypasses."""
    raw_path = os.environ.get(BUDGET_FILE_ENV)
    if not raw_path:
        return
    path = Path(raw_path)
    token = os.environ.get(INVOCATION_TOKEN_ENV)
    if token:
        if token in read_budget(path)["issued_tokens"]:
            return
        raise RuntimeError("Benchmark invocation token is not present in the branch budget ledger")
    os.environ[INVOCATION_TOKEN_ENV] = reserve_budget(path)

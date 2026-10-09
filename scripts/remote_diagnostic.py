#!/usr/bin/env python3
"""Execute one KernelBench candidate workload for a remote diagnostic tool.

This helper intentionally delegates execution to KernelBench's public
``eval_kernel_against_ref`` entry point with performance measurement disabled.
It neither serializes a benchmark result nor decides correctness; the calling
sanitizer is diagnostic evidence only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _task_path(kernelbench_path: str, task_id: str) -> Path:
    normalized = task_id.replace("\\", "/").removesuffix(".py")
    if normalized.startswith("/") or ".." in normalized.split("/"):
        raise ValueError(f"Invalid KernelBench task id: {task_id!r}")
    root = Path(kernelbench_path).resolve()
    nested = root / "KernelBench"
    if nested.is_dir():
        root = nested
    task = (root / f"{normalized}.py").resolve()
    if root not in task.parents or not task.is_file():
        raise FileNotFoundError(f"KernelBench task {task_id!r} was not found under {kernelbench_path}")
    return task


def _candidate_source(blob: str) -> tuple[dict, str]:
    data = json.loads(blob)
    if data.get("format") != "ako4x-kernelbench-v1":
        raise ValueError("Unsupported KernelBench solution blob")
    sources = data.get("sources")
    build = data.get("build")
    if not isinstance(sources, dict) or not isinstance(build, dict):
        raise ValueError("Incomplete KernelBench solution blob")
    entry = str(build.get("entry_point", "kernel.py::run")).split("::", 1)[0]
    source = sources.get(entry)
    if source is None and len(sources) == 1:
        source = next(iter(sources.values()))
    if not isinstance(source, str) or "ModelNew" not in source:
        raise ValueError("KernelBench candidate must define ModelNew")
    return data, source


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one KernelBench candidate for a diagnostic tool")
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    data, candidate_source = _candidate_source(str(request["candidate_blob"]))
    task = _task_path(str(request["kernelbench_path"]), str(request["task_id"]))

    import torch
    from kernelbench.eval import eval_kernel_against_ref

    language = str(data["build"].get("language", "python")).lower()
    backend = language if language in {"triton", "tilelang", "cute"} else "cuda"
    result = eval_kernel_against_ref(
        task.read_text(encoding="utf-8"),
        candidate_source,
        num_correct_trials=1,
        num_perf_trials=1,
        measure_performance=False,
        timing_method="cuda_event",
        verbose=False,
        device=torch.device("cuda:0"),
        backend=backend,
        precision=torch.float32,
        check_for_excessive_speedup=False,
    )
    if result is None or not getattr(result, "compiled", False):
        print("DIAGNOSTIC_CANDIDATE_DID_NOT_COMPILE")
        return 10
    print("DIAGNOSTIC_CANDIDATE_WORKLOAD_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

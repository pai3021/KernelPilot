#!/usr/bin/env python3
"""Run remote Compute Sanitizer without entering the formal benchmark path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_backend import RemoteSSHBenchmarkBackend
from scripts.pack_solution import pack_solution
from scripts.run_remote import load_remote_config


_VALID_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck", "all")


def _workloads() -> list[dict]:
    path = PROJECT_ROOT / "docs" / "workloads.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Remote sanitizer needs {path}")
    workloads = [json.loads(line)["workload"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not workloads:
        raise ValueError("Remote sanitizer found no workloads")
    return workloads


def _parse_indices(raw: str, count: int) -> list[int]:
    values: set[int] = set()
    for section in raw.split(","):
        section = section.strip()
        if not section:
            raise ValueError("empty workload index")
        if "-" in section:
            first, last = (int(value) for value in section.split("-", 1))
            if first > last:
                raise ValueError(f"invalid workload range: {section}")
            values.update(range(first, last + 1))
        else:
            values.add(int(section))
    indices = sorted(values)
    if any(index < 0 or index >= count for index in indices):
        raise ValueError(f"workload index must be between 0 and {count - 1}")
    return indices


def _write_artifact(directory: Path, prefix: str, result: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    artifact = directory / f"{stem}.json"
    artifact.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    for stream in ("stdout", "stderr"):
        (directory / f"{stem}.{stream}.log").write_text(str(result.get(stream, "")), encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Remote KernelBench Compute Sanitizer diagnostic")
    parser.add_argument("--index", default="0", help="Workload indices, e.g. 0 or 0,2-4 (default: 0)")
    parser.add_argument("--list", action="store_true", help="List child workloads and exit")
    parser.add_argument("--tool", default="memcheck", choices=_VALID_TOOLS)
    parser.add_argument("--timeout", type=int, default=300, help="Per-tool remote timeout in seconds")
    args = parser.parse_args()
    try:
        workloads = _workloads()
        if args.list:
            for index, workload in enumerate(workloads):
                print(json.dumps({"index": index, **workload}, sort_keys=True))
            return 0
        if args.timeout < 1:
            raise ValueError("--timeout must be positive")
        blob = pack_solution(quiet=True).read_text(encoding="utf-8")
        backend = RemoteSSHBenchmarkBackend(load_remote_config())
        exit_code = 0
        tools = _VALID_TOOLS[:-1] if args.tool == "all" else (args.tool,)
        for index in _parse_indices(args.index, len(workloads)):
            task_id = str(workloads[index]["uuid"])
            for tool in tools:
                result = backend.run_sanitizer(blob, task_id, tool=tool, timeout_seconds=args.timeout)
                result.update({"workload_index": index, "workload_axes": workloads[index].get("axes", {})})
                artifact = _write_artifact(PROJECT_ROOT / "artifacts" / "diagnostics" / "sanitize", f"w{index}-{tool}", result)
                print(json.dumps({"artifact": str(artifact), **result}, sort_keys=True))
                if result["status"] != "PASSED":
                    exit_code = 3
        return exit_code
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"protocol_version": 1, "diagnostic": "compute-sanitizer", "status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

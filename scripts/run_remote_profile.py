#!/usr/bin/env python3
"""Run an on-demand remote Nsight Systems timeline for one KernelBench workload."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_backend import RemoteSSHBenchmarkBackend
from scripts.pack_solution import pack_solution

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _workloads() -> list[dict]:
    path = PROJECT_ROOT / "docs" / "workloads.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Remote profiler needs {path}")
    workloads = [json.loads(line)["workload"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not workloads:
        raise ValueError("Remote profiler found no workloads")
    return workloads


def _load_remote_config() -> dict:
    """Read the existing [benchmark.remote] mapping without loading bench.sh."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Remote profiler needs {config_path}")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    values = dict(config.get("benchmark", {}))
    remote = config.get("benchmark", {}).get("remote", {}) or config.get("remote_benchmark", {})
    aliases = {
        "host": "remote_host", "python": "remote_python", "kernelbench": "remote_kernelbench",
        "runs_dir": "remote_runs_dir", "compute_sanitizer": "remote_compute_sanitizer", "nsys": "remote_nsys",
    }
    values.update({aliases.get(key, key): value for key, value in remote.items()})
    return values


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


def _number(value: str) -> float:
    return float(value.replace(",", ""))


def _parse_nsys_stats(path: Path) -> dict:
    """Parse only the CSV report rows emitted by the verified nsys version."""
    if not path.is_file():
        return {
            "top_cuda_kernels": None, "kernel_launch_count": None,
            "cuda_memcpy_count": None, "top_cuda_api_calls": None,
            "parse_status": {"stats": "missing"},
        }
    rows = list(csv.reader(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    kernel_rows: list[dict[str, str]] = []
    api_rows: list[dict[str, str]] = []
    parse_status = {"kernels": "not_present", "cuda_api": "not_present", "cuda_memcpy": "not_present"}
    index = 0
    while index < len(rows):
        header = rows[index]
        if "Total Time (ns)" not in header or "Name" not in header:
            index += 1
            continue
        data_rows: list[dict[str, str]] = []
        index += 1
        while index < len(rows) and rows[index] and len(rows[index]) == len(header):
            data_rows.append(dict(zip(header, rows[index])))
            index += 1
        if "Instances" in header:
            kernel_rows = data_rows
            parse_status["kernels"] = "parsed"
        elif "Num Calls" in header:
            api_rows = data_rows
            parse_status["cuda_api"] = "parsed"
        elif any("Memcpy" in value or "MemOps" in value for value in header):
            parse_status["cuda_memcpy"] = "parsed_but_schema_not_stable"
        index += 1

    def top(rows_to_sort: list[dict[str, str]], calls_key: str) -> list[dict] | None:
        if not rows_to_sort:
            return None
        parsed = []
        for row in rows_to_sort:
            try:
                parsed.append({
                    "name": row["Name"],
                    "total_time_ms": _number(row["Total Time (ns)"]) / 1_000_000,
                    "avg_time_us": _number(row["Avg (ns)"]) / 1_000,
                    "calls": int(_number(row[calls_key])),
                })
            except (KeyError, ValueError):
                continue
        return sorted(parsed, key=lambda item: item["total_time_ms"], reverse=True)[:10] or None

    kernels = top(kernel_rows, "Instances")
    try:
        launch_count = sum(int(_number(row["Instances"])) for row in kernel_rows)
    except (KeyError, ValueError):
        launch_count = None
    try:
        memcpy_count = sum(
            int(_number(row["Num Calls"]))
            for row in api_rows
            if "memcpy" in row.get("Name", "").lower()
        )
        if memcpy_count:
            parse_status["cuda_memcpy"] = "parsed_from_cuda_api"
        else:
            memcpy_count = None
    except (KeyError, ValueError):
        memcpy_count = None
    return {
        "top_cuda_kernels": kernels,
        "kernel_launch_count": launch_count,
        "cuda_memcpy_count": memcpy_count,
        "top_cuda_api_calls": top(api_rows, "Num Calls"),
        "parse_status": parse_status,
    }


def _artifact_dir(index: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "artifacts" / "diagnostics" / "profile" / f"w{index}-{stamp}-{uuid.uuid4().hex[:8]}"


def _write_summary(directory: Path, result: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    summary = directory / "summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    for stream in ("stdout", "stderr"):
        (directory / f"profile.{stream}.log").write_text(str(result.get(stream, "")), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Remote KernelBench Nsight Systems timeline diagnostic")
    parser.add_argument("--index", default="0", help="Workload indices, e.g. 0 or 0,2-4 (default: 0)")
    parser.add_argument("--list", action="store_true", help="List child workloads and exit")
    parser.add_argument("--timeout", type=int, default=300, help="Remote timeout in seconds")
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
        backend = RemoteSSHBenchmarkBackend(_load_remote_config())
        exit_code = 0
        for index in _parse_indices(args.index, len(workloads)):
            directory = _artifact_dir(index)
            result = backend.run_profiler(
                blob, str(workloads[index]["uuid"]), artifact_dir=directory, timeout_seconds=args.timeout,
            )
            result.update({"workload_index": index, "workload_axes": workloads[index].get("axes", {})})
            result.update(_parse_nsys_stats(Path(result["stats_path"])))
            summary = _write_summary(directory, result)
            print(json.dumps({"artifact": str(summary), **result}, sort_keys=True))
            if result["status"] != "PASS":
                exit_code = 3
        return exit_code
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "protocol_version": 1, "tool": "nsight-systems", "mode": "cuda_timeline",
            "status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}",
            "diagnostic_only": True, "formal_correctness_verdict": "NOT_RUN",
            "formal_performance_verdict": "NOT_RUN",
        }, sort_keys=True), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

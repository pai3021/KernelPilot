#!/usr/bin/env python3
"""Fixed remote child for ``RemoteSSHBenchmarkBackend``.

It receives a small JSON request and writes one JSON envelope.  It is not a
remote-shell service: the only operation is adapter.run on the supplied packed
candidate and requested KernelBench task IDs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one fixed KernelBench evaluation request")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(args.request.read_text())
    metadata = dict(request.get("metadata", {}))
    try:
        from scripts import benchmark_adapter as adapter

        results = adapter.run(
            request["candidate_blob"], request["task_ids"], request["params"],
            dataset_path=request["kernelbench_path"],
            capture_logs=bool(request.get("capture_logs", False)),
        )
        status = "SUCCESS"
        error_log = ""
    except Exception as exc:
        status = "INFRA_FAILURE"
        error_log = f"Remote evaluator setup failure: {type(exc).__name__}: {exc}"
        try:
            definition = str(json.loads(request.get("candidate_blob", "{}")).get("definition") or "remote-evaluation")
        except (TypeError, json.JSONDecodeError):
            definition = "remote-evaluation"
        results = {definition: {
            task_id: {
                "status": status, "solution": "remote-candidate", "axes": {},
                "max_abs_error": "NaN", "max_rel_error": "NaN",
                "error_log": error_log, "metadata": metadata,
            }
            for task_id in request.get("task_ids", [])
        }}
    metadata["duration_seconds"] = round(time.monotonic() - started, 6)
    _write_json(args.result, {
        "protocol_version": 1,
        "status": status,
        "error_log": error_log,
        "metadata": metadata,
        "results": results,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

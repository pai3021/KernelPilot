#!/usr/bin/env python3
"""Filesystem client for a parent-side remote benchmark bridge.

Codex workspace-write sandboxes can edit and execute local tools while denying
all socket syscalls.  This client sends only the already-approved benchmark
arguments to its parent runtime via files in the child workspace, then prints
the parent's ordinary ``run_remote.py`` output verbatim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import uuid


BRIDGE_DIR_ENV = "EXPERIMENTPILOT_BENCHMARK_BRIDGE_DIR"
BRIDGE_SERVER_ENV = "EXPERIMENTPILOT_BENCHMARK_BRIDGE_SERVER"


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = os.environ.get(BRIDGE_DIR_ENV)
    if not root or os.environ.get(BRIDGE_SERVER_ENV):
        raise RuntimeError("Benchmark bridge client requires a parent runtime bridge")
    bridge = Path(root)
    requests = bridge / "requests"
    responses = bridge / "responses"
    if not requests.is_dir() or not responses.is_dir():
        raise RuntimeError("Benchmark bridge is not available for this worker")

    request_id = uuid.uuid4().hex
    request_path = requests / f"{request_id}.json"
    response_path = responses / f"{request_id}.json"
    _write_json(request_path, {"protocol_version": 1, "args": arguments})

    deadline = time.monotonic() + int(os.environ.get("EXPERIMENTPILOT_BENCHMARK_BRIDGE_TIMEOUT", "360"))
    while time.monotonic() < deadline:
        if response_path.is_file():
            response = json.loads(response_path.read_text())
            sys.stdout.write(str(response.get("stdout", "")))
            sys.stderr.write(str(response.get("stderr", "")))
            return int(response.get("returncode", 1))
        time.sleep(0.05)
    raise TimeoutError("Benchmark bridge timed out waiting for the parent runtime")


if __name__ == "__main__":
    raise SystemExit(main())

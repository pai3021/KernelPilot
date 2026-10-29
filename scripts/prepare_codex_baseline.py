#!/usr/bin/env python3
"""Prepare a minimal native-Codex context without running a model or a GPU."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping
from uuid import uuid4


CLIENT = '''#!/usr/bin/env python3
"""Submit the current candidate to the fixed benchmark endpoint."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

root = Path(__file__).resolve().parent
endpoint = root / ".benchmark"
if not (endpoint / "requests").is_dir() or not (endpoint / "responses").is_dir():
    raise SystemExit("Benchmark endpoint is not running")
source = (root / "solution/kernel.py").read_bytes()
request_id = uuid.uuid4().hex
request = {"protocol_version": 1, "candidate_source": source.decode("utf-8"),
           "candidate_sha256": hashlib.sha256(source).hexdigest()}
target = endpoint / "requests" / (request_id + ".json")
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(request), encoding="utf-8")
temporary.replace(target)
deadline = time.monotonic() + 600
response_path = endpoint / "responses" / (request_id + ".json")
while time.monotonic() < deadline:
    if response_path.is_file():
        response = json.loads(response_path.read_text())
        print(json.dumps(response))
        raise SystemExit(0 if response.get("status") == "PASSED" else 1)
    time.sleep(0.05)
raise SystemExit("Benchmark response timed out; execution outcome is unknown")
'''


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _verified_source(facts: Mapping[str, Any], key: str) -> bytes:
    payload = facts[key]["source"].encode("utf-8")
    if not payload.strip() or hashlib.sha256(payload).hexdigest() != facts[key]["sha256"]:
        raise ValueError(f"Missing or mismatched source witness: {key}")
    return payload


def prepare_workspace(workspace: Path, facts: Mapping[str, Any], *, search_limit: int = 6) -> dict[str, Any]:
    """Expose problem facts and one fixed tool, without KernelPilot guidance."""
    if isinstance(search_limit, bool) or not isinstance(search_limit, int) or search_limit < 1:
        raise ValueError("search_limit must be a positive integer")
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise FileExistsError(f"Refusing to reuse baseline workspace: {workspace}")
    inherited = [str(parent / "AGENTS.md") for parent in workspace.parents if (parent / "AGENTS.md").exists()]
    if inherited:
        raise ValueError(f"Choose a neutral workspace without inherited project instructions: {inherited}")
    task_source = _verified_source(facts, "task")
    parent_source = _verified_source(facts, "parent_candidate")
    factual_context = {"task_id": facts["task_id"], "environment": facts["environment"],
                       "evaluation_policy": facts["evaluation_policy"]}
    factual_json = _json(factual_context)  # Validate before creating any files.
    problem = (
        "# Kernel optimization task\n\n"
        "Optimize solution/kernel.py for the task in reference.py on the specified hardware. "
        "Keep ModelNew and its forward signature. Implement handwritten Triton; torch may allocate "
        "buffers but reference/native high-performance operations such as torch.matmul are forbidden.\n\n"
        "## Fixed task and evaluation facts\n\n```json\n" + factual_json + "```\n\n"
        "## Evaluation interface\n\nRun `python3 evaluate.py` to evaluate the current solution/kernel.py. "
        f"You may submit at most {search_limit} search candidates within the run's time limit. "
        "The command returns correctness, latency and any compiler/runtime errors. "
        "Do not run other GPU work, modify reference.py or evaluate.py, change evaluation settings, "
        "or read other workspaces, historical solutions, memory, or external repository solutions. "
        "Only solution/kernel.py is editable.\n\n"
        "## Final submission\n\nLeave the candidate you choose to deliver in solution/kernel.py. "
        "You may retain the unchanged starting candidate as an explicit fallback if no correct improvement is found. "
        "It will receive a separate evaluation after your session ends. "
        "State what you implemented and the measured results.\n"
    )
    workspace.mkdir(parents=True)
    (workspace / "solution").mkdir()
    (workspace / "reference.py").write_bytes(task_source)
    (workspace / "solution/kernel.py").write_bytes(parent_source)
    (workspace / "PROBLEM.md").write_text(problem, encoding="utf-8")
    (workspace / "evaluate.py").write_text(CLIENT, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(workspace)], check=True, capture_output=True, timeout=10)
    visible_files = ("reference.py", "solution/kernel.py", "PROBLEM.md", "evaluate.py")
    return {"workspace": str(workspace), "profile": "minimal-native-codex", "search_limit": search_limit,
            "file_sha256": {name: hashlib.sha256((workspace / name).read_bytes()).hexdigest() for name in visible_files},
            "initial_parent_sha256": hashlib.sha256(parent_source).hexdigest(),
            "factual_context_sha256": hashlib.sha256(factual_json.encode()).hexdigest(),
            "model_or_gpu_calls": 0, "endpoint_status": "NOT_STARTED",
            "limits": ["Global Codex instructions/config still require launch-time auditing.",
                       "This minimal context does not provide OS-enforced read isolation."]}


def snapshot_submission(workspace: Path, observer_root: Path) -> dict[str, str]:
    """Freeze the delivered file; never replace it with the best earlier report."""
    workspace, observer_root = Path(workspace).resolve(), Path(observer_root).resolve()
    if observer_root.is_relative_to(workspace):
        raise ValueError("Observer snapshots must be outside the agent workspace")
    source = (workspace / "solution/kernel.py").resolve()
    if not source.is_relative_to(workspace):
        raise ValueError("Candidate source must stay inside its workspace")
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    observer_root.mkdir(parents=True, exist_ok=True)
    path = observer_root / (digest + ".kernel.py")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("Corrupt existing submission snapshot")
    return {"candidate_sha256": digest, "artifact_path": str(path), "selection": "delivered-file"}


class MinimalBenchmarkEndpoint:
    """External experiment observer: fixed evaluator, cap and source identity only.

    The callback is supplied by the experiment owner, never by the agent. This
    endpoint neither proposes a strategy nor selects or restores a candidate.
    Its conservative submission counter is separate from physical GPU dispatch
    records. An infrastructure fault makes the performance cohort incomplete.
    """

    FEEDBACK_FIELDS = frozenset({"status", "correct", "latency_ms", "reference_latency_ms",
                                 "speedup_factor", "error_log", "log"})

    def __init__(self, workspace: Path, observer_root: Path,
                 evaluate: Callable[[str], Mapping[str, Any]], *, search_limit: int = 6):
        self.workspace = Path(workspace).resolve()
        self.observer = Path(observer_root).resolve()
        if self.observer.is_relative_to(self.workspace):
            raise ValueError("Observer records must stay outside the agent workspace")
        if isinstance(search_limit, bool) or not isinstance(search_limit, int) or search_limit < 1:
            raise ValueError("search_limit must be a positive integer")
        self.evaluate, self.limit = evaluate, search_limit
        self.requests = self.workspace / ".benchmark/requests"
        self.responses = self.workspace / ".benchmark/responses"
        for directory in (self.requests, self.responses, self.observer):
            directory.mkdir(parents=True, exist_ok=True)
        self.ledger = self.observer / "submissions.json"

    @contextmanager
    def _locked(self):
        import fcntl  # Control-side execution uses native Linux/WSL.
        with (self.observer / "submissions.lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write(path: Path, value: Any):
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        temporary.write_text(_json(value), encoding="utf-8")
        temporary.replace(path)

    def process_request(self, request: Path) -> dict[str, Any]:
        request = Path(request)
        request_id = request.name.split(".")[0]
        if request.parent.resolve() != self.requests or not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise ValueError("Invalid benchmark request identity")
        value = json.loads(request.read_text())
        if (set(value) != {"protocol_version", "candidate_source", "candidate_sha256"}
                or value["protocol_version"] != 1 or not isinstance(value["candidate_source"], str)):
            raise ValueError("Only candidate source may cross the evaluation interface")
        source, digest = value["candidate_source"], value["candidate_sha256"]
        if not source.strip() or hashlib.sha256(source.encode()).hexdigest() != digest:
            raise ValueError("Candidate witness does not match submitted source")
        record_path = self.observer / (request_id + ".json")
        with self._locked():
            state = json.loads(self.ledger.read_text()) if self.ledger.exists() else {
                "search_limit": self.limit, "submission_attempts": 0}
            if state["search_limit"] != self.limit:
                raise ValueError("Cannot change an existing experiment cap")
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record["candidate_sha256"] != digest:
                    raise ValueError("Request identity was reused for a different candidate")
                feedback = record.get("feedback") or {
                    "status": "INFRA_FAILURE", "correct": None,
                    "error_log": "Previous submission outcome is unknown; it will not be rerun"}
            elif state.get("incomplete"):
                feedback = {"status": "INFRA_FAILURE", "correct": None,
                            "error_log": "This experiment is incomplete; stop further submissions"}
            elif state["submission_attempts"] >= self.limit:
                feedback = {"status": "BUDGET_EXHAUSTED", "correct": None,
                            "error_log": "Search submission allowance exhausted"}
            else:
                state["submission_attempts"] += 1
                self._write(self.ledger, state)
                record = {"candidate_source": source, "candidate_sha256": digest, "state": "SUBMITTED"}
                self._write(record_path, record)
                try:
                    result = dict(self.evaluate(source))
                    if not isinstance(result.get("status"), str):
                        raise ValueError("Evaluator did not return a status")
                    feedback = {key: result[key] for key in self.FEEDBACK_FIELDS if key in result}
                except Exception as exc:
                    feedback = {"status": "INFRA_FAILURE", "correct": None,
                                "error_log": f"Evaluator outcome unknown: {type(exc).__name__}: {exc}"}
                record.update({"state": "RETURNED", "feedback": feedback})
                self._write(record_path, record)
            if feedback["status"] == "INFRA_FAILURE":
                state["incomplete"] = True
                self._write(self.ledger, state)
        self._write(self.responses / (request_id + ".json"), feedback)
        return feedback

    def serve_once(self):
        for request in sorted(self.requests.glob("*.json")):
            claimed = request.with_suffix(".json.processing")
            try:
                request.replace(claimed)
            except FileNotFoundError:
                continue
            self.process_request(claimed)


def fixed_backend_evaluator(backend: Any, *, task_id: str,
                            evaluation_params: Mapping[str, Any]) -> Callable[[str], Mapping[str, Any]]:
    """Bind the same fixed oracle for search feedback and independent final grading.

    This callback stays outside the native agent context. Transport/backend
    reliability remains shared measurement infrastructure in this experiment.
    """
    frozen = _json(dict(evaluation_params))
    if not task_id.strip():
        raise ValueError("task_id must not be empty")
    def evaluate(source: str) -> Mapping[str, Any]:
        blob = _json({"format": "ako4x-kernelbench-v1", "name": "candidate",
                      "definition": task_id, "author": "experiment",
                      "build": {"language": "triton", "entry_point": "kernel.py::ModelNew"},
                      "sources": {"kernel.py": source}})
        execution = backend.evaluate_blob(blob, [task_id], json.loads(frozen), capture_logs=True)
        status = getattr(execution, "status", None)
        status = getattr(status, "value", status)
        metadata = getattr(execution, "metadata", {})
        if status == "INFRA_FAILURE" or (status == "TIMEOUT" and metadata.get("failure_stage")):
            return {"status": "INFRA_FAILURE", "correct": None,
                    "error_log": execution.error_log or "Fixed evaluator infrastructure failed"}
        workloads = [item for group in execution.results.values() for item in group.values()]
        if len(workloads) != 1:
            return {"status": "INFRA_FAILURE", "correct": None,
                    "error_log": "Fixed evaluator returned an unexpected workload count"}
        item = workloads[0]
        feedback = {key: item[key] for key in MinimalBenchmarkEndpoint.FEEDBACK_FIELDS if key in item}
        feedback["correct"] = execution.correct
        if execution.error_log and not feedback.get("error_log"):
            feedback["error_log"] = execution.error_log
        return feedback
    return evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--search-limit", type=int, default=6)
    args = parser.parse_args()
    if args.manifest_out.exists() or args.manifest_out.resolve().is_relative_to(args.workspace.resolve()):
        raise ValueError("Choose a new manifest outside the baseline workspace")
    value = prepare_workspace(args.workspace, json.loads(args.facts.read_text()), search_limit=args.search_limit)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("x") as stream:
        stream.write(_json(value))
    print(_json(value), end="")


if __name__ == "__main__":
    main()

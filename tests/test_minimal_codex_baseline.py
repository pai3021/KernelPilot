"""Check baseline information boundaries and delivered-candidate identity on CPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from scripts.prepare_codex_baseline import (
    MinimalBenchmarkEndpoint, fixed_backend_evaluator, prepare_workspace, snapshot_submission,
)


def witness(source):
    return {"source": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}


class MinimalCodexBaselineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / "native"
        self.facts = {"task_id": "level1/test", "task": witness("class Model: pass\n"),
                      "parent_candidate": witness("class ModelNew: pass\n"),
                      "environment": {"gpu": "test-only", "triton": "fixed"},
                      "evaluation_policy": {"atol": 1e-4, "rtol": 1e-4,
                                            "correct_trials": 3, "perf_trials": 20}}

    def tearDown(self):
        self.temp.cleanup()

    def test_minimal_context_contains_exact_facts_and_reference_but_no_harness_guides(self):
        record = prepare_workspace(self.workspace, self.facts)
        files = {str(path.relative_to(self.workspace)) for path in self.workspace.rglob("*")
                 if path.is_file() and ".git" not in path.relative_to(self.workspace).parts}
        self.assertEqual(files, {"PROBLEM.md", "reference.py", "evaluate.py", "solution/kernel.py"})
        self.assertEqual((self.workspace / "reference.py").read_text(), self.facts["task"]["source"])
        self.assertEqual(record["initial_parent_sha256"], self.facts["parent_candidate"]["sha256"])
        problem = (self.workspace / "PROBLEM.md").read_text()
        self.assertIn('"atol": 0.0001', problem)
        self.assertIn('"perf_trials": 20', problem)
        self.assertIn("at most 6", problem)
        for name in ("AGENTS.md", "CODEX_TASK.md", "KernelPilot", "Planner", "Skills"):
            self.assertNotIn(name, problem)
        self.assertEqual(record["model_or_gpu_calls"], 0)
        self.assertEqual(record["endpoint_status"], "NOT_STARTED")

    def test_task_source_is_data_and_is_never_executed_by_preparation(self):
        marker = self.root / "must-not-exist"
        source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        self.facts["task"] = witness(source)
        prepare_workspace(self.workspace, self.facts)
        self.assertFalse(marker.exists())

    def test_mismatched_source_invalid_budget_or_inherited_instructions_fail_before_creation(self):
        self.facts["task"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            prepare_workspace(self.workspace, self.facts)
        self.assertFalse(self.workspace.exists())
        for limit in (0, -1, True):
            with self.assertRaises(ValueError):
                prepare_workspace(self.workspace, self.facts, search_limit=limit)
        (self.root / "AGENTS.md").write_text("Project-specific instructions")
        self.facts["task"] = witness("class Model: pass\n")
        with self.assertRaisesRegex(ValueError, "neutral workspace"):
            prepare_workspace(self.workspace, self.facts)

    def test_preparation_never_overwrites_an_existing_workspace(self):
        prepare_workspace(self.workspace, self.facts)
        (self.workspace / "solution/kernel.py").write_text("# existing work\n")
        with self.assertRaises(FileExistsError):
            prepare_workspace(self.workspace, self.facts)
        self.assertEqual((self.workspace / "solution/kernel.py").read_text(), "# existing work\n")

    def test_final_submission_is_the_delivered_file_not_an_observer_selected_best(self):
        prepare_workspace(self.workspace, self.facts)
        (self.workspace / "previous-fast.py").write_text("# a previously measured fast candidate\n")
        current = b"# the agent deliberately leaves a different final candidate\n"
        (self.workspace / "solution/kernel.py").write_bytes(current)
        record = snapshot_submission(self.workspace, self.root / "observer")
        self.assertEqual(Path(record["artifact_path"]).read_bytes(), current)
        self.assertEqual(record["candidate_sha256"], hashlib.sha256(current).hexdigest())
        self.assertEqual(record["selection"], "delivered-file")
        with self.assertRaises(ValueError):
            snapshot_submission(self.workspace, self.workspace / "snapshots")

    def test_client_binds_the_submitted_source_and_preserves_evaluator_feedback(self):
        prepare_workspace(self.workspace, self.facts)
        endpoint = self.workspace / ".benchmark"
        (endpoint / "requests").mkdir(parents=True)
        (endpoint / "responses").mkdir()
        captured, faults = [], []
        feedback = {"status": "INCORRECT_NUMERICAL", "correct": False,
                    "error_log": "fixed-oracle numerical mismatch"}
        def fake_endpoint():
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    requests = list((endpoint / "requests").glob("*.json"))
                    if requests:
                        request = requests[0]
                        captured.append(json.loads(request.read_text()))
                        # A later mutable edit must not change the submitted payload.
                        (self.workspace / "solution/kernel.py").write_text("# later edit\n")
                        response = endpoint / "responses" / request.name
                        temporary = response.with_suffix(".tmp")
                        temporary.write_text(json.dumps(feedback))
                        temporary.replace(response)
                        return
                    time.sleep(0.01)
                raise AssertionError("client did not submit a request")
            except BaseException as exc:
                faults.append(exc)
        server = threading.Thread(target=fake_endpoint)
        server.start()
        try:
            result = subprocess.run([sys.executable, "evaluate.py"], cwd=self.workspace,
                                    text=True, capture_output=True, timeout=6)
        finally:
            server.join(timeout=6)
        self.assertEqual(faults, [])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout), feedback)
        self.assertEqual(captured[0]["candidate_source"], self.facts["parent_candidate"]["source"])
        self.assertEqual(captured[0]["candidate_sha256"], self.facts["parent_candidate"]["sha256"])
        self.assertEqual(set(captured[0]), {"protocol_version", "candidate_source", "candidate_sha256"})

    def test_client_without_an_endpoint_does_not_fake_a_benchmark_result(self):
        prepare_workspace(self.workspace, self.facts)
        result = subprocess.run([sys.executable, "evaluate.py"], cwd=self.workspace,
                                text=True, capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("not running", result.stderr)

    def request(self, endpoint, identity, source="class ModelNew: pass\n", **extra):
        path = endpoint.requests / (identity * 32 + ".json")
        path.write_text(json.dumps({"protocol_version": 1, "candidate_source": source,
                                    "candidate_sha256": hashlib.sha256(source.encode()).hexdigest(), **extra}))
        return path

    def test_external_observer_gives_feedback_without_strategy_or_best_candidate_assistance(self):
        prepare_workspace(self.workspace, self.facts)
        sources = []
        def evaluate(source):
            sources.append(source)
            return {"status": "PASSED", "correct": True, "latency_ms": 2.0,
                    "next_strategy": "privileged hint", "best_candidate": "a previous solution"}
        observer = self.root / "observer"
        endpoint = MinimalBenchmarkEndpoint(self.workspace, observer, evaluate)
        request = self.request(endpoint, "a")
        feedback = endpoint.process_request(request)
        self.assertEqual(feedback, {"status": "PASSED", "correct": True, "latency_ms": 2.0})
        restarted = MinimalBenchmarkEndpoint(self.workspace, observer, evaluate)
        self.assertEqual(restarted.process_request(request), feedback)
        self.assertEqual(len(sources), 1)
        self.assertEqual(json.loads((observer / "submissions.json").read_text())["submission_attempts"], 1)
        self.assertFalse((self.workspace / "observer").exists())

    def test_neutral_submission_cap_is_enforced_and_agent_cannot_supply_evaluation_settings(self):
        prepare_workspace(self.workspace, self.facts)
        called = []
        endpoint = MinimalBenchmarkEndpoint(self.workspace, self.root / "observer",
                                            lambda source: called.append(source) or {"status": "PASSED"},
                                            search_limit=1)
        with self.assertRaises(ValueError):
            endpoint.process_request(self.request(endpoint, "a", perf_trials=1))
        self.assertEqual(called, [])
        endpoint.process_request(self.request(endpoint, "b"))
        response = endpoint.process_request(self.request(endpoint, "c"))
        self.assertEqual(response["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(len(called), 1)

    def test_unknown_evaluation_outcome_remains_recorded_and_stops_more_work(self):
        prepare_workspace(self.workspace, self.facts)
        called = []
        def fail(source):
            called.append(source)
            raise RuntimeError("connection lost after dispatch")
        observer = self.root / "observer"
        endpoint = MinimalBenchmarkEndpoint(self.workspace, observer, fail)
        self.assertEqual(endpoint.process_request(self.request(endpoint, "a"))["status"], "INFRA_FAILURE")
        self.assertEqual(endpoint.process_request(self.request(endpoint, "b"))["status"], "INFRA_FAILURE")
        self.assertEqual(len(called), 1)
        self.assertTrue(json.loads((observer / "submissions.json").read_text())["incomplete"])

    def test_unknown_pending_request_is_not_executed_again_after_restart(self):
        prepare_workspace(self.workspace, self.facts)
        observer = self.root / "observer"
        called = []
        endpoint = MinimalBenchmarkEndpoint(self.workspace, observer,
                                            lambda source: called.append(source) or {"status": "PASSED"})
        request = self.request(endpoint, "a")
        value = json.loads(request.read_text())
        (observer / ("a" * 32 + ".json")).write_text(json.dumps({
            "candidate_sha256": value["candidate_sha256"], "candidate_source": value["candidate_source"],
            "state": "SUBMITTED"}))
        (observer / "submissions.json").write_text(json.dumps({"search_limit": 6, "submission_attempts": 1}))
        response = endpoint.process_request(request)
        self.assertEqual(response["status"], "INFRA_FAILURE")
        self.assertEqual(endpoint.process_request(self.request(endpoint, "b"))["status"], "INFRA_FAILURE")
        self.assertEqual(called, [])

    def test_fixed_backend_callback_preserves_task_source_and_frozen_trial_settings(self):
        calls = []
        class Backend:
            def evaluate_blob(inner, blob, tasks, params, **kwargs):
                calls.append((json.loads(blob), tasks, params, kwargs))
                return SimpleNamespace(correct=True, error_log="", results={"task": {"only": {
                    "status": "PASSED", "latency_ms": 1.2, "speedup_factor": 2,
                    "reference_latency_ms": 2.4, "best_candidate": "private advice"}}})
        params = {"correct_trials": 3, "perf_trials": 20, "timing_method": "cuda_event"}
        evaluate = fixed_backend_evaluator(Backend(), task_id="level1/test", evaluation_params=params)
        params["perf_trials"] = 1
        source = "class ModelNew: pass\n"
        feedback = evaluate(source)
        self.assertEqual(calls[0][0]["sources"], {"kernel.py": source})
        self.assertEqual(calls[0][1], ["level1/test"])
        self.assertEqual(calls[0][2]["perf_trials"], 20)
        self.assertEqual(calls[0][2]["correct_trials"], 3)
        self.assertTrue(calls[0][3]["capture_logs"])
        self.assertNotIn("best_candidate", feedback)
        self.assertEqual(feedback["status"], "PASSED")

    def test_transport_failure_cannot_be_presented_as_a_passing_candidate(self):
        for status in ("INFRA_FAILURE", "TIMEOUT"):
            with self.subTest(status=status):
                class Backend:
                    def evaluate_blob(inner, *args, **kwargs):
                        return SimpleNamespace(status=status, metadata={"failure_stage": "download"},
                                               correct=None, error_log="result transfer failed",
                                               results={"task": {"only": {"status": "PASSED"}}})
                evaluate = fixed_backend_evaluator(Backend(), task_id="task", evaluation_params={})
                self.assertEqual(evaluate("class ModelNew: pass")["status"], "INFRA_FAILURE")

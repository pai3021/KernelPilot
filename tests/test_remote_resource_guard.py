"""CPU-only failure injection for readiness, dispatch accounting and stopping."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from agent_runtime import AgentRunRequest, AgentRunResult, TaskContract
from agent_runtime.codex import CodexRuntime, _FilesystemBenchmarkBridge
from benchmark_backend import ExecutionStatus, RemoteSSHBenchmarkBackend, RemoteSSHConfig
from benchmark_backend.budget import (
    BUDGET_FILE_ENV, INVOCATION_TOKEN_ENV, BenchmarkBudgetExhausted,
    begin_benchmark_scope_from_environment, ensure_evaluation_authorized,
    initialize_budget, read_budget,
)
from benchmark_backend.provenance import capture_evaluated_candidate
from benchmark_backend.resource_guard import (
    RemoteResourceBlocked, block_path, block_remote, read_block,
)
from campaign.controller import TwoBranchController
from campaign.models import BranchSpec, RoundResult, RoundState


def config(**overrides):
    return dict(remote_host="unused-host", remote_repo="/repo", remote_python="/python",
                remote_kernelbench="/kb", remote_runs_dir="/runs", preferred_gpu=5,
                gpu_wait_timeout_seconds=2, gpu_poll_interval_seconds=1, **overrides)


BLOB = json.dumps({"definition": "task"})
RESULTS = {"task": {"task": {"status": "PASSED", "latency_ms": 1.0,
                              "reference_latency_ms": 2.0, "speedup_factor": 2.0}}}


class RemoteReadinessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger = self.root / "budget.json"
        initialize_budget(self.ledger, limit=2)
        self.environment = patch.dict(os.environ, {BUDGET_FILE_ENV: str(self.ledger)})
        self.environment.start()
        os.environ.pop(INVOCATION_TOKEN_ENV, None)
        self.backend = RemoteSSHBenchmarkBackend(config())
        self.commands = []
        self.uploads = []
        self.states = ["5, 12, 0\n"]
        self.clock = 0.0
        self.time_patch = patch("benchmark_backend.ssh_remote.time.monotonic", side_effect=lambda: self.clock)
        self.sleep_patch = patch("benchmark_backend.ssh_remote.time.sleep", side_effect=self.advance)
        self.time_patch.start()
        self.sleep_patch.start()
        self.backend._ssh = self.ssh
        self.backend._scp_to = self.upload
        self.backend._scp_from = self.download

    def tearDown(self):
        self.sleep_patch.stop()
        self.time_patch.stop()
        self.environment.stop()
        self.temp.cleanup()

    def advance(self, seconds):
        self.clock += seconds

    def ok(self):
        return subprocess.CompletedProcess([], 0, "", "")

    def ssh(self, command, *, timeout):
        self.commands.append(command)
        if command.startswith("nvidia-smi"):
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return subprocess.CompletedProcess([], 0, state, "")
        if "timeout --signal=TERM" in command:
            # The ledger must be committed before dispatch, never afterwards.
            self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)
        return self.ok()

    def upload(self, local, remote, *, timeout):
        self.uploads.append(remote)
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 0)
        return self.ok()

    def download(self, remote, local, *, timeout):
        # Simulate stale echoed request metadata; local dispatch evidence wins.
        local.write_text(json.dumps({"protocol_version": 1, "status": "SUCCESS", "results": RESULTS,
                                     "metadata": {"evaluation_dispatched": False}}))
        return self.ok()

    def evaluate(self):
        return self.backend.evaluate_blob(BLOB, ["task"], {"num_correct_trials": 3, "num_perf_trials": 20})

    def events(self):
        return [json.loads(line) for line in self.ledger.with_suffix(".json.remote-events.jsonl").read_text().splitlines()]

    def test_busy_then_ready_waits_without_spending_then_dispatches_once(self):
        self.states = ["5, 1500, 35\n", "5, 1200, 20\n", "5, 12, 0\n"]
        begin_benchmark_scope_from_environment()
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 0)
        result = self.evaluate()
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertTrue(result.metadata["evaluation_dispatched"])
        self.assertNotIn("failure_stage", result.results["task"]["task"]["metadata"])
        self.assertEqual(result.metadata["resource_wait_seconds"], 2)
        self.assertEqual(result.metadata["resource_polls"], 4)  # Includes the post-upload gate.
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)
        self.assertEqual(sum(event["event"] == "EVALUATION_DISPATCH" for event in self.events()), 1)
        self.assertIsNone(read_block(self.ledger))

    def test_busy_deadline_blocks_repeated_requests_and_new_process_without_charge(self):
        self.states = ["5, 1500, 35\n7, 12, 0\n"]  # Never switch the configured GPU.
        result = self.evaluate()
        self.assertEqual(result.status, ExecutionStatus.INFRA_FAILURE)
        self.assertEqual(result.metadata["failure_stage"], "resource_wait")
        self.assertFalse(result.metadata["evaluation_dispatched"])
        self.assertEqual(self.clock, 2)
        self.assertEqual(self.uploads, [])
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 0)
        original = block_path(self.ledger).read_bytes()
        queries = len(self.commands)
        os.environ.pop(INVOCATION_TOKEN_ENV, None)  # A different runner process cannot reset the stop.
        with self.assertRaises(RemoteResourceBlocked):
            self.evaluate()
        self.assertEqual(len(self.commands), queries)
        self.assertEqual(block_path(self.ledger).read_bytes(), original)

    def test_gpu_becomes_busy_during_upload_no_dispatch_no_charge(self):
        self.states = ["5, 12, 0\n", "5, 1500, 35\n"]
        result = self.evaluate()
        self.assertEqual(result.status, ExecutionStatus.INFRA_FAILURE)
        self.assertEqual(len(self.uploads), 2)
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 0)
        self.assertFalse(any("timeout --signal=TERM" in command for command in self.commands))

    def test_auto_selected_gpu_is_pinned_during_post_upload_wait(self):
        values = config()
        values["preferred_gpu"] = None
        self.backend.config = RemoteSSHConfig.from_mapping(values)
        self.states = ["5, 12, 0\n7, 12, 0\n", "5, 1500, 35\n7, 12, 0\n"]
        result = self.evaluate()
        self.assertEqual(result.status, ExecutionStatus.INFRA_FAILURE)
        self.assertEqual(result.metadata["physical_gpu_id"], 5)
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 0)
        self.assertFalse(any("timeout --signal=TERM" in command for command in self.commands))

    def test_query_transport_error_is_not_treated_as_busy_or_retried(self):
        def broken(command, *, timeout):
            self.commands.append(command)
            return subprocess.CompletedProcess([], 255, "", "connection refused")
        self.backend._ssh = broken
        result = self.evaluate()
        self.assertEqual(result.status, ExecutionStatus.INFRA_FAILURE)
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.clock, 0)
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 0)

    def test_preflight_and_upload_failures_do_not_commit_dispatch_budget(self):
        for stage in ("preflight", "upload"):
            with self.subTest(stage=stage):
                if block_path(self.ledger).exists():
                    block_path(self.ledger).unlink()
                if stage == "preflight":
                    self.backend._preflight = lambda gpu: (_ for _ in ()).throw(RuntimeError("bad environment"))
                else:
                    self.backend._preflight = lambda gpu: {}
                    self.backend._scp_to = lambda *a, **kw: subprocess.CompletedProcess([], 1, "", "permission denied")
                result = self.evaluate()
                self.assertEqual(result.status, ExecutionStatus.INFRA_FAILURE)
                self.assertEqual(result.metadata["failure_stage"], stage)
                self.assertEqual(read_budget(self.ledger)["benchmark_used"], 0)

    def test_reference_and_candidate_share_credit_but_both_dispatches_are_audited(self):
        self.assertEqual(self.evaluate().status, ExecutionStatus.SUCCESS)
        # A baseline has already consumed this invocation's one comparison credit.
        self.backend._scp_to = lambda *a, **kw: self.ok()
        self.assertEqual(self.evaluate().status, ExecutionStatus.SUCCESS)
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)
        self.assertEqual(sum(event["event"] == "EVALUATION_DISPATCH" for event in self.events()), 2)

    def test_busy_candidate_after_successful_baseline_does_not_refund_credit(self):
        self.assertEqual(self.evaluate().status, ExecutionStatus.SUCCESS)
        self.states = ["5, 1500, 35\n"]
        failed = self.evaluate()
        self.assertEqual(failed.status, ExecutionStatus.INFRA_FAILURE)
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)
        self.assertIsNotNone(read_block(self.ledger))

    def test_unknown_dispatch_outcome_stays_charged_and_is_not_rerun(self):
        original = self.backend._ssh
        def lost(command, *, timeout):
            if "timeout --signal=TERM" in command:
                self.commands.append(command)
                raise subprocess.TimeoutExpired("ssh", timeout)
            return original(command, timeout=timeout)
        self.backend._ssh = lost
        result = self.evaluate()
        self.assertEqual(result.status, ExecutionStatus.TIMEOUT)
        self.assertTrue(result.metadata["evaluation_dispatched"])
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)
        with self.assertRaises(RemoteResourceBlocked):
            self.evaluate()
        self.assertEqual(sum("timeout --signal=TERM" in command for command in self.commands), 1)

    def test_result_download_failure_keeps_dispatch_charge_and_stops(self):
        self.backend._scp_from = lambda *a, **kw: subprocess.CompletedProcess([], 1, "", "connection lost")
        result = self.evaluate()
        self.assertEqual(result.status, ExecutionStatus.INFRA_FAILURE)
        self.assertEqual(result.metadata["failure_stage"], "download")
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)
        self.assertIsNotNone(read_block(self.ledger))

    def test_candidate_correctness_failure_is_charged_but_not_an_infrastructure_stop(self):
        def incorrect(remote, local, *, timeout):
            local.write_text(json.dumps({"protocol_version": 1, "status": "SUCCESS",
                                         "results": {"task": {"task": {"status": "INCORRECT_NUMERICAL"}}}}))
            return self.ok()
        self.backend._scp_from = incorrect
        result = self.evaluate()
        self.assertEqual(result.status, ExecutionStatus.CORRECTNESS_FAILURE)
        self.assertFalse(result.correct)
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)
        self.assertIsNone(read_block(self.ledger))

    def test_exhausted_budget_fails_before_any_remote_query(self):
        ensure_evaluation_authorized()
        os.environ.pop(INVOCATION_TOKEN_ENV)
        ensure_evaluation_authorized()
        os.environ.pop(INVOCATION_TOKEN_ENV)
        with self.assertRaises(BenchmarkBudgetExhausted):
            self.evaluate()
        self.assertEqual(self.commands, [])

    def test_old_ledger_without_policy_keeps_runner_attempt_accounting(self):
        data = read_budget(self.ledger)
        data.pop("accounting_policy")
        self.ledger.write_text(json.dumps(data))
        begin_benchmark_scope_from_environment()
        self.states = ["5, 1500, 35\n"]
        self.evaluate()
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)
        self.assertNotIn("accounting_policy", read_budget(self.ledger))

    def test_invalid_wait_configuration_fails_before_remote_use(self):
        for name, value in (("gpu_wait_timeout_seconds", -1), ("gpu_poll_interval_seconds", 0)):
            values = config()
            values[name] = value
            with self.assertRaises(ValueError):
                RemoteSSHConfig.from_mapping(values)

    def test_wait_configuration_survives_spawn_config_loading(self):
        from spawn import load_remote_config, REMOTE_OPTIONAL_KEYS
        path = self.root / "remote.toml"
        path.write_text('[benchmark.remote]\nhost="unused"\nremote_repo="/repo"\npython="/python"\n'
                        'kernelbench="/kb"\nruns_dir="/runs"\ngpu_wait_timeout_seconds=7\n'
                        'gpu_poll_interval_seconds=2\n')
        values = load_remote_config(path)
        self.assertEqual(values["gpu_wait_timeout_seconds"], 7)
        self.assertEqual(values["gpu_poll_interval_seconds"], 2)
        self.assertIn("gpu_wait_timeout_seconds", REMOTE_OPTIONAL_KEYS)
        path.write_text(path.read_text().replace("gpu_poll_interval_seconds=2", "gpu_poll_interval_seconds=0"))
        with self.assertRaises(ValueError):
            load_remote_config(path)


class ResourceStopPropagationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger = self.root / ".ako/budget.json"
        initialize_budget(self.ledger)

    def tearDown(self):
        self.temp.cleanup()

    def test_bridge_refuses_blocked_scope_without_spawning_runner(self):
        with patch.dict(os.environ, {BUDGET_FILE_ENV: str(self.ledger)}):
            block_remote("busy", {"evaluation_dispatched": False})
        bridge = _FilesystemBenchmarkBridge(self.root, {BUDGET_FILE_ENV: str(self.ledger)}, timeout=10)
        bridge.requests.mkdir(parents=True)
        bridge.responses.mkdir()
        request = bridge.requests / "one.json.processing"
        request.write_text(json.dumps({"protocol_version": 1, "args": []}))
        with patch("agent_runtime.codex.subprocess.run") as run:
            bridge._serve_request(request)
        run.assert_not_called()
        self.assertTrue(bridge.blocked.is_set())
        self.assertIn("blocked", json.loads((bridge.responses / "one.json").read_text())["stderr"])

    def test_runtime_terminates_only_its_worker_after_bridge_reports_block(self):
        killed = threading.Event()
        class Process:
            pid = 55555
            returncode = None
            def communicate(inner, timeout=None):
                if not killed.wait(2):
                    raise AssertionError("blocked worker was not stopped promptly")
                inner.returncode = -9
                return "", ""
            def poll(inner):
                return inner.returncode
            def kill(inner):
                killed.set()
        def start_bridge(bridge):
            bridge.blocked.set()
        runtime = CodexRuntime(template_root=self.root, executable="unused-codex")
        with patch("agent_runtime.codex._FilesystemBenchmarkBridge.start", start_bridge), \
             patch("agent_runtime.codex._FilesystemBenchmarkBridge.close"), \
             patch("agent_runtime.codex.subprocess.Popen", return_value=Process()), \
             patch("agent_runtime.codex.os.getpgid", return_value=55555), \
             patch("agent_runtime.codex.os.killpg", side_effect=lambda *a: killed.set()):
            result = runtime.run(AgentRunRequest(self.root, "work", 5))
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.timed_out)
        self.assertTrue(result.metadata["remote_resource_blocked"])

    def test_runtime_refuses_to_restart_worker_with_an_existing_block(self):
        with patch.dict(os.environ, {BUDGET_FILE_ENV: str(self.ledger)}):
            block_remote("busy", {})
        runtime = CodexRuntime(template_root=self.root, executable="unused-codex")
        with patch("agent_runtime.codex.subprocess.Popen") as popen:
            result = runtime.run(AgentRunRequest(self.root, "work", 5,
                                                 environment={BUDGET_FILE_ENV: str(self.ledger)}))
        popen.assert_not_called()
        self.assertEqual(result.status, "blocked")

    def test_closed_bridge_does_not_start_another_already_queued_request(self):
        bridge = _FilesystemBenchmarkBridge(self.root, {}, timeout=10)
        bridge.requests.mkdir(parents=True)
        for name in ("a.json", "b.json"):
            (bridge.requests / name).write_text("{}")
        served = []
        def serve(path):
            served.append(path.name)
            bridge._stop.set()
        bridge._serve_request = serve
        bridge._serve()
        self.assertEqual(served, ["a.json.processing"])

    def test_controller_preserves_best_candidate_but_stops_before_next_branch(self):
        workspace = self.root / "worker"
        (workspace / "solution").mkdir(parents=True)
        (workspace / "solution/kernel.py").write_text("# measured candidate\n")
        (workspace / "config.toml").write_text('[build]\nlanguage="triton"\n')
        task = TaskContract("optimize", ("solution/kernel.py",), "bash scripts/bench.sh",
                            "fixed oracle", ("no evaluator edits",), ("stop on block",))
        specs = tuple(BranchSpec(name, "parent", "hypothesis", "custom_triton_elementwise",
                                 "reason", "signal", "work", "triton") for name in ("b1", "b2"))
        spawned = []
        def spawn(branch):
            spawned.append(branch)
            return workspace
        class Runtime:
            def prepare_workspace(inner, *args):
                pass
            def run(inner, request):
                witness = capture_evaluated_candidate(workspace)
                report = workspace / "trajectory/first/results.json"
                report.parent.mkdir(parents=True)
                report.write_text(json.dumps({"evaluated_candidate": witness, "results": RESULTS}))
                with patch.dict(os.environ, request.environment):
                    block_remote("GPU busy after earlier successful evaluation", {"evaluation_dispatched": False})
                return AgentRunResult("session", "blocked", -9, "", (), False, 0.1, "")
        controller = TwoBranchController(project_root=self.root, task=task, operator="op", dataset=self.root,
                                         parent_kernel=workspace / "solution/kernel.py", parent_id="parent",
                                         parent_speedup=1.0, spawn_child=spawn, branch_benchmark_limit=3)
        controller._plan = lambda: specs
        with patch("campaign.controller.RuntimeRegistry.create", return_value=Runtime()):
            result = controller.run()
        self.assertEqual(spawned, ["b1"])
        self.assertEqual(result.state, RoundState.RESOURCE_BLOCKED)
        self.assertIsNone(result.selected_branch_id)
        self.assertEqual(result.branches[0].status, "PASSED")
        self.assertEqual(result.branches[0].failure_type, "REMOTE_RESOURCE_BLOCKED")
        self.assertTrue(result.branches[0].candidate_path.is_file())
        self.assertEqual(RoundResult.from_dict(result.to_dict()).to_dict(), result.to_dict())
        state = json.loads((controller.round_dir / "round-state.json").read_text())
        self.assertEqual(state["unstarted_branches"], ["b2"])

    def test_corrupt_stop_record_fails_closed_and_first_cause_is_preserved(self):
        block_path(self.ledger).write_text("{")
        with patch.dict(os.environ, {BUDGET_FILE_ENV: str(self.ledger)}):
            block_remote("later reason", {})
        self.assertIn("Unreadable", read_block(self.ledger)["reason"])
        self.assertEqual(block_path(self.ledger).read_text(), "{")

    def test_controller_does_not_reset_an_existing_branch_ledger(self):
        ledger = self.root / ".ako/branch-budget.json"
        initialize_budget(ledger, limit=3)
        with patch.dict(os.environ, {BUDGET_FILE_ENV: str(ledger)}):
            os.environ.pop(INVOCATION_TOKEN_ENV, None)
            ensure_evaluation_authorized()
        before = ledger.read_bytes()
        task = TaskContract("optimize", ("solution/kernel.py",), "bench", "oracle", (), ())
        controller = TwoBranchController(project_root=self.root, task=task, operator="op", dataset=self.root,
                                         parent_kernel=self.root / "kernel.py", parent_id="parent",
                                         parent_speedup=1, spawn_child=lambda _: self.root)
        spec = BranchSpec("b1", "parent", "hypothesis", "custom_triton_elementwise",
                          "reason", "signal", "work", "triton")
        with patch("campaign.controller.RuntimeRegistry.create") as create:
            result = controller._run_branch(spec)
        create.assert_not_called()
        self.assertEqual(result.failure_type, "FileExistsError")
        self.assertEqual(ledger.read_bytes(), before)

"""Regression checks for the bounded Harness-versus-native experiment."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent_runtime import AgentRunResult, TaskContract
from benchmark_backend.budget import (
    BUDGET_FILE_ENV, INVOCATION_TOKEN_ENV, BenchmarkBudgetExhausted,
    begin_benchmark_scope_from_environment, ensure_evaluation_authorized,
    initialize_budget, read_budget,
)
from benchmark_backend.provenance import capture_evaluated_candidate
from campaign.controller import TwoBranchController, best_evaluated_report
from campaign.models import BranchSpec
from campaign.planner import BranchPlanner


ROOT = Path(__file__).resolve().parents[1]


def task():
    return TaskContract("optimize", ("solution/kernel.py",), "bash scripts/bench.sh",
                        "fixed oracle", ("no evaluator changes",), ("budget exhausted",))


class RunnerBudgetProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ledger = self.root / "budget.json"
        self.blob = self.root / "solution.json"
        self.blob.write_text("{}")
        # Only the expensive runner dependencies are replaced. The real main(),
        # argument parser and budget guard run, so ordering regressions surface.
        self.utils = types.ModuleType("scripts.bench_utils")
        for name in ("find_group_axis", "parse_int_filter", "run_ab_compare",
                     "run_and_report", "run_variance_check"):
            setattr(self.utils, name, Mock())
        self.utils.find_group_axis.return_value = ""
        spec = importlib.util.spec_from_file_location("budget_protocol_runner", ROOT / "scripts/run_remote.py")
        self.runner = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"scripts.bench_utils": self.utils}):
            spec.loader.exec_module(self.runner)
        self.backend = Mock()
        self.backend.evaluate_blob.side_effect = self.evaluate
        self.evaluations = 0

    def tearDown(self):
        self.tmp.cleanup()

    def evaluate(self, *args, **kwargs):
        ensure_evaluation_authorized()
        self.evaluations += 1
        return types.SimpleNamespace(results={})

    def invoke(self, args, callback, *, pack_error=None):
        self.utils.run_and_report.side_effect = callback
        self.utils.run_variance_check.side_effect = callback
        with patch.dict(os.environ, {BUDGET_FILE_ENV: str(self.ledger)}):
            for key in (INVOCATION_TOKEN_ENV, "EXPERIMENTPILOT_BENCHMARK_BRIDGE_DIR"):
                os.environ.pop(key, None)
            with patch.object(sys, "argv", ["run_remote.py", "--quiet", *args]), \
                 patch.object(sys, "stdout"), \
                 patch.object(self.runner, "_configure_console_output"), \
                 patch.object(self.runner, "load_remote_config", return_value={}), \
                 patch.object(self.runner, "RemoteSSHBenchmarkBackend", return_value=self.backend) as create, \
                 patch.object(self.runner, "capture_evaluated_candidate", return_value={}), \
                 patch.object(self.runner, "pack_solution", return_value=self.blob, side_effect=pack_error):
                self.runner.main()
                return create.call_count

    def test_first_and_smoke_each_consume_the_only_credit(self):
        for args in (["--first", "1"], ["--smoke"]):
            with self.subTest(args=args):
                initialize_budget(self.ledger, limit=1)
                self.invoke(args, lambda blob, run, **kw: run(blob, [], {}))
                self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)
                prior = self.evaluations
                with self.assertRaises(BenchmarkBudgetExhausted):
                    self.invoke([], lambda blob, run, **kw: run(blob, [], {}))
                self.assertEqual(self.evaluations, prior)

    def test_packing_failure_or_empty_workload_set_does_not_spend_dispatch_credit(self):
        initialize_budget(self.ledger, limit=1)
        with self.assertRaisesRegex(ValueError, "invalid candidate"):
            self.invoke([], lambda *a, **kw: None, pack_error=ValueError("invalid candidate"))
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 0)
        self.invoke([], lambda *a, **kw: None)
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 0)
        self.assertEqual(self.evaluations, 0)

    def test_reference_and_candidate_share_one_credit(self):
        initialize_budget(self.ledger, limit=1)
        def comparison(blob, run, **kwargs):
            run("reference", [], {})
            run(blob, [], {})
        self.invoke([], comparison)
        self.assertEqual(self.evaluations, 2)
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)

    def test_three_separate_feedback_invocations_stop_at_three(self):
        initialize_budget(self.ledger, limit=3)
        for _ in range(3):
            self.invoke([], lambda blob, run, **kw: run(blob, [], {}))
        with self.assertRaises(BenchmarkBudgetExhausted):
            self.invoke([], lambda blob, run, **kw: run(blob, [], {}))
        self.assertEqual(self.evaluations, 3)

    def test_variance_mode_counts_invocations_rather_than_actual_measurements(self):
        # Characterizes V1's accounting boundary. The experiment must disallow
        # this mode instead of presenting the ledger as physical GPU usage.
        initialize_budget(self.ledger, limit=1)
        def repeated(blob, run, **kwargs):
            for _ in range(kwargs["n_runs"]):
                run(blob, [], {})
        self.invoke(["--variance-check", "3"], repeated)
        self.assertEqual(self.evaluations, 3)
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)


class MultiEvaluationEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "solution").mkdir()
        self.kernel = self.root / "solution/kernel.py"
        self.kernel.write_text("# original\n")
        self.spec = BranchSpec("b1", "parent", "hypothesis", "custom_triton_elementwise",
                               "reason", "signal", "worker", "triton")

    def tearDown(self):
        self.tmp.cleanup()

    def controller(self, limit=1):
        return TwoBranchController(project_root=self.root, task=task(), operator="op",
                                   dataset=self.root, parent_kernel=self.kernel,
                                   parent_id="parent", parent_speedup=1.0,
                                   branch_benchmark_limit=limit)

    def report(self, label, status, speedup):
        self.kernel.write_text(f"# candidate {label}\n")
        witness = capture_evaluated_candidate(self.root)
        p = self.root / "trajectory" / label / "results.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"evaluated_candidate": witness,
                                "results": {"task": {"work": {"status": status, "speedup_factor": speedup}}}}))
        return p, witness

    def test_later_failure_keeps_earlier_best_passing_snapshot(self):
        best, witness = self.report("one", "PASSED", 1.5)
        self.report("two", "PASSED", 1.1)
        self.report("three", "COMPILE_ERROR", None)
        initialize_budget(self.root / ".ako/branch-budget.json", limit=3)
        result = self.controller(3)._branch_result(self.spec, self.root, None, None)
        self.assertEqual(result.status, "PASSED")
        self.assertEqual(result.speedup_factor, 1.5)
        self.assertEqual(result.candidate_path, Path(witness["artifact_path"]))
        self.assertEqual(result.evaluated_candidate_sha256, witness["candidate_sha256"])
        self.assertEqual(result.artifacts["benchmark_result"], best)
        self.assertEqual(result.artifacts["selected_report_provenance"], best)
        self.assertNotIn("evaluated_candidate_provenance", result.artifacts)
        self.assertNotEqual(hashlib.sha256(self.kernel.read_bytes()).hexdigest(), result.evaluated_candidate_sha256)

    def test_missing_corrupt_or_unbound_snapshots_are_never_selected(self):
        _, witness = self.report("one", "PASSED", 2.0)
        Path(witness["artifact_path"]).write_text("# corrupt\n")
        self.report("two", "INCORRECT_NUMERICAL", 99.0)
        p, _ = self.report("three", "PASSED", 100.0)
        data = json.loads(p.read_text()); data.pop("evaluated_candidate"); p.write_text(json.dumps(data))
        self.assertIsNone(best_evaluated_report(self.root))

    def test_all_failed_evaluations_preserve_the_actual_failure_report(self):
        self.report("one", "COMPILE_ERROR", None)
        last, _ = self.report("two", "INCORRECT_NUMERICAL", None)
        result = self.controller(3)._branch_result(self.spec, self.root, None, None)
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.benchmark_result["status"], "INCORRECT_NUMERICAL")
        self.assertEqual(result.artifacts["benchmark_result"], last)
        self.assertEqual(result.failure_type, "BENCHMARK_OR_CORRECTNESS_FAILURE")

    def test_corrupt_passing_evidence_never_becomes_a_promotable_fallback(self):
        _, witness = self.report("one", "PASSED", 2.0)
        Path(witness["artifact_path"]).write_text("# corrupt\n")
        result = self.controller(3)._branch_result(self.spec, self.root, None, None)
        self.assertEqual(result.status, "FAILED")
        self.assertIsNone(result.speedup_factor)

    def test_default_limit_and_opt_in_worker_and_planner_guidance(self):
        self.assertEqual(self.controller().branch_benchmark_limit, 1)
        self.assertIn("at most 3", self.controller(3)._worker_prompt(self.spec, 3))
        planner = BranchPlanner("codex", self.root, benchmark_limit=3)
        prompt = planner._prompt(parent_id="parent", parent_speedup=1.0, task=task(),
                                 retry=False, branch_ids=("b1", "b2"), planning_context="")
        self.assertIn("at most 3", prompt)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.controller(invalid)

    def test_controller_materializes_the_requested_limit_for_worker(self):
        (self.root / "config.toml").write_text('[build]\nlanguage = "triton"\n')
        seen = []
        class Runtime:
            def prepare_workspace(inner, workspace, contract):
                pass
            def run(inner, request):
                seen.append(read_budget(Path(request.environment[BUDGET_FILE_ENV]))["benchmark_limit"])
                return AgentRunResult("session", "completed", 0, "done", (), False, 0.1, "")
        controller = self.controller(3)
        controller._spawn_child = lambda branch: self.root
        with patch("campaign.controller.RuntimeRegistry.create", return_value=Runtime()):
            controller._run_branch(self.spec)
        self.assertEqual(seen, [3])

    def test_trajectory_persists_pre_evaluation_witness_after_a_working_copy_edit(self):
        import builtins
        witness = capture_evaluated_candidate(self.root)
        self.kernel.write_text("# an unmeasured later edit\n")
        module_spec = importlib.util.spec_from_file_location("pilot_test_bench_utils", ROOT / "scripts/bench_utils.py")
        utils = importlib.util.module_from_spec(module_spec)
        real_open = builtins.open
        def redirected(path, *args, **kwargs):
            if Path(path) == ROOT / "config.toml":
                path = ROOT / "tests/fixtures/config.toml"
            return real_open(path, *args, **kwargs)
        with patch("builtins.open", side_effect=redirected):
            module_spec.loader.exec_module(utils)
        utils.PROJECT_ROOT = self.root
        with patch.object(sys, "stdout"):
            utils.save_trajectory({"task": {"work": {"status": "PASSED", "speedup_factor": 1.2}}},
                                  {"name": "candidate", "definition": "task"}, label="witness")
        path = next((self.root / "trajectory").glob("*/results.json"))
        self.assertEqual(json.loads(path.read_text())["evaluated_candidate"], witness)
        self.assertEqual(best_evaluated_report(self.root), path)

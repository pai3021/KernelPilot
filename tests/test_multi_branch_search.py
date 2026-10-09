from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_runtime import AgentEvent, AgentRunResult, TaskContract
from benchmark_backend.budget import (
    BUDGET_FILE_ENV,
    INVOCATION_TOKEN_ENV,
    BenchmarkBudgetExhausted,
    begin_benchmark_scope_from_environment,
    ensure_evaluation_authorized,
    initialize_budget,
    read_budget,
)
from campaign.build_language import (
    BuildLanguageConfigurationError,
    TRITON_STRATEGY_FAMILIES,
    materialize_build_language,
    read_build_language,
    resolve_build_language,
    validate_build_language,
)
from campaign.controller import TwoBranchController
from campaign.models import BranchResult, BranchSpec, RoundResult, RoundState
from campaign.planner import BranchPlanner
from campaign.ranking import rank_candidates
from campaign.state import CampaignState, ParentSnapshot, RoundRecord


def _spec(branch_id: str, family: str, build_language: str | None = None) -> BranchSpec:
    return BranchSpec(branch_id, "parent", f"hypothesis {branch_id}", family, "reason", "signal", "worker prompt", build_language)


def _result(branch_id: str, speedup: float | None, status: str = "PASSED") -> BranchResult:
    benchmark = None if speedup is None else {"status": status, "speedup_factor": speedup}
    return BranchResult(branch_id, "codex", "session", status, Path("kernel.py"), "", benchmark, 1, "done", None)


class BranchModelAndRankingTest(unittest.TestCase):
    def test_spec_requires_complete_nonempty_contract(self):
        with self.assertRaises(ValueError):
            BranchSpec.from_dict({"branch_id": "b1"})

    def test_correctness_is_a_hard_gate(self):
        decision = rank_candidates([_result("b1", 1.9, "FAILED"), _result("b2", 1.0)], parent_speedup=1.1)
        self.assertEqual(decision.decision, "KEEP_CURRENT_PARENT")
        self.assertIsNone(decision.selected_branch_id)

    def test_only_strict_improvement_promotes(self):
        self.assertEqual(rank_candidates([_result("b1", 1.1)], parent_speedup=1.1).decision, "KEEP_CURRENT_PARENT")
        promoted = rank_candidates([_result("b1", 1.2), _result("b2", 1.3)], parent_speedup=1.1)
        self.assertEqual((promoted.decision, promoted.selected_branch_id), ("PROMOTE", "b2"))

    def test_completed_round_round_trips_for_recovery_without_benchmark(self):
        original = RoundResult(
            "round-001", RoundState.PROMOTED, "reference", 1.0,
            (_result("b1", 1.1), _result("b2", 0.9)), "b1", "PROMOTE",
            {"workspace": Path("campaign/round-001")},
        )
        restored = RoundResult.from_dict(original.to_dict())
        self.assertEqual(restored.to_dict(), original.to_dict())
        self.assertEqual(restored.branches[0].speedup_factor, 1.1)


class PlannerParsingTest(unittest.TestCase):
    def test_parser_requires_exact_two_diverse_families(self):
        payload = {"branches": [_spec("b1", "custom_kernel", "python").to_dict(), _spec("b2", "scheduling", "python").to_dict()]}
        parsed = BranchPlanner.parse(json.dumps(payload), parent_id="parent")
        self.assertEqual([item.branch_id for item in parsed], ["b1", "b2"])
        payload["branches"][1]["strategy_family"] = "custom_kernel"
        with self.assertRaisesRegex(ValueError, "strategy_family"):
            BranchPlanner.parse(json.dumps(payload), parent_id="parent")

    def test_round_two_prompt_carries_structured_prior_feedback_and_new_ids(self):
        task = TaskContract("objective", ("solution/kernel.py",), "bash scripts/bench.sh", "oracle", ("no evaluator edit",), ("stop",))
        prompt = BranchPlanner("codex", Path.cwd())._prompt(
            parent_id="phase2a-r001-b2", parent_speedup=1.0303, task=task, retry=False,
            branch_ids=("b3", "b4"), planning_context="b1: strategy_family=custom_triton_gemm; status=COMPILE_ERROR.\nb2: strategy_family=native_dispatch_layout_optimization; status=PASSED.",
        )
        self.assertIn('"branch_id":"b3"', prompt)
        self.assertIn("custom_triton_gemm", prompt)
        self.assertIn("native_dispatch_layout_optimization", prompt)
        payload = {"branches": [_spec("b3", "native_dispatch_refinement", "python").to_dict(), _spec("b4", "alternative_schedule", "python").to_dict()]}
        self.assertEqual({item.branch_id for item in BranchPlanner.parse(json.dumps(payload), parent_id="parent", branch_ids=("b3", "b4"))}, {"b3", "b4"})


class BuildLanguageResolverTest(unittest.TestCase):
    def _config(self, root: Path, language: str = "python") -> Path:
        path = root / "config.toml"
        path.write_text(f"[build]\nlanguage = \"{language}\"\n\n[benchmark]\nbackend = \"ssh\"\n")
        return path

    def test_observed_triton_families_resolve_explicitly(self):
        self.assertIn("custom_triton_elementwise", TRITON_STRATEGY_FAMILIES)
        self.assertIn("triton_fused_elementwise_kernel", TRITON_STRATEGY_FAMILIES)
        for family in TRITON_STRATEGY_FAMILIES:
            self.assertEqual(resolve_build_language(family), "triton")

    def test_non_triton_and_unknown_families_preserve_existing_config(self):
        self.assertIsNone(resolve_build_language("framework_dispatch"))
        self.assertIsNone(resolve_build_language("custom_cuda_elementwise"))
        self.assertIsNone(resolve_build_language("unrecognized_future_strategy"))

    def test_planner_declared_language_handles_new_triton_family_without_matching(self):
        self.assertEqual(resolve_build_language("custom_triton_outer_product", "triton"), "triton")
        with self.assertRaisesRegex(ValueError, "Unsupported planner build_language"):
            resolve_build_language("custom_triton_outer_product", "unknown")

    def test_materialize_updates_only_build_language(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            materialize_build_language(config, "custom_triton_elementwise")
            self.assertEqual(read_build_language(config), "triton")
            self.assertIn('[benchmark]\nbackend = "ssh"', config.read_text())

    def test_preflight_reports_strategy_expected_and_actual(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory), "python")
            with self.assertRaisesRegex(BuildLanguageConfigurationError, "custom_triton_elementwise.*triton.*python"):
                validate_build_language(config, "custom_triton_elementwise")

    def test_missing_child_config_is_a_harness_configuration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            with self.assertRaisesRegex(BuildLanguageConfigurationError, "custom_triton_elementwise.*triton.*None"):
                materialize_build_language(config, "custom_triton_elementwise")


class CampaignStateTest(unittest.TestCase):
    def test_round_boundary_state_persists_parent_lineage_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign-state.json"
            parent = ParentSnapshot("phase2a-r001-b2", "b2", Path("candidate.py"), {"status": "PASSED", "speedup_factor": 1.0303}, "phase2a-r001")
            state = CampaignState("campaign", "task", 1, parent, parent, parent.benchmark_result, [RoundRecord("phase2a-r001", "baseline", {"speedup_factor": 1.0042}, (), (), "b2", "PROMOTE", parent.parent_id)], 4, 2)
            state.save(path)
            restored = CampaignState.load(path)
            self.assertEqual(restored.current_parent.parent_id, "phase2a-r001-b2")
            self.assertEqual((restored.round_index, restored.benchmark_budget_used, restored.benchmark_budget_total), (1, 2, 4))
            self.assertEqual(restored.round_history[0].selected_parent, "phase2a-r001-b2")


class BudgetGuardTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ledger = Path(self.tempdir.name) / "branch-budget.json"
        initialize_budget(self.ledger)
        self.old_file = os.environ.get(BUDGET_FILE_ENV)
        self.old_token = os.environ.get(INVOCATION_TOKEN_ENV)
        os.environ[BUDGET_FILE_ENV] = str(self.ledger)
        os.environ.pop(INVOCATION_TOKEN_ENV, None)

    def tearDown(self):
        if self.old_file is None:
            os.environ.pop(BUDGET_FILE_ENV, None)
        else:
            os.environ[BUDGET_FILE_ENV] = self.old_file
        if self.old_token is None:
            os.environ.pop(INVOCATION_TOKEN_ENV, None)
        else:
            os.environ[INVOCATION_TOKEN_ENV] = self.old_token
        self.tempdir.cleanup()

    def test_runner_scope_allows_reference_and_candidate_backend_calls_once(self):
        begin_benchmark_scope_from_environment()
        ensure_evaluation_authorized()
        ensure_evaluation_authorized()
        self.assertEqual(read_budget(self.ledger)["benchmark_used"], 1)

    def test_new_process_scope_is_blocked_after_budget(self):
        begin_benchmark_scope_from_environment()
        os.environ.pop(INVOCATION_TOKEN_ENV, None)
        with self.assertRaises(BenchmarkBudgetExhausted):
            begin_benchmark_scope_from_environment()


class ControllerFailureIsolationTest(unittest.TestCase):
    def test_custom_child_label_prefix_is_retained_for_later_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = TaskContract("objective", ("solution/kernel.py",), "bash scripts/bench.sh", "oracle", ("no evaluator edit",), ("stop",))
            controller = TwoBranchController(project_root=root, task=task, operator="op", dataset=root, parent_kernel=root / "parent.py", parent_id="parent", parent_speedup=1.0, child_label_prefix="phase2b-campaign")
            self.assertEqual(controller.child_label_prefix, "phase2b-campaign")

    def test_one_branch_spawn_failure_does_not_stop_other_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_b2 = root / "b2"
            (child_b2 / "solution").mkdir(parents=True)
            (child_b2 / "solution" / "kernel.py").write_text("# candidate\n")

            def spawn(branch_id: str) -> Path:
                if branch_id == "b1":
                    raise RuntimeError("synthetic spawn failure")
                return child_b2

            class FakeRuntime:
                def prepare_workspace(self, workspace, task):
                    (workspace / "CODEX_TASK.md").write_text(task.objective)

                def run(self, request):
                    report = request.workspace / "trajectory" / "one" / "results.json"
                    report.parent.mkdir(parents=True)
                    report.write_text(json.dumps({"results": {"task": {"work": {"status": "PASSED", "speedup_factor": 1.2}}}}))
                    return AgentRunResult("b2-session", "completed", 0, "branch result", (AgentEvent("session_completed"),), False, 0.1, "")

            task = TaskContract("objective", ("solution/kernel.py",), "bash scripts/bench.sh", "oracle", ("no evaluator edit",), ("stop",))
            controller = TwoBranchController(
                project_root=root, task=task, operator="op", dataset=root, parent_kernel=child_b2 / "solution" / "kernel.py",
                parent_id="parent", parent_speedup=1.0, round_id="test-round", spawn_child=spawn,
            )
            with patch.object(controller, "_plan", return_value=(_spec("b1", "custom_kernel"), _spec("b2", "scheduling"))), patch(
                "campaign.controller.RuntimeRegistry.create", return_value=FakeRuntime()
            ):
                result = controller.run()
            self.assertEqual(result.state, RoundState.PROMOTED)
            self.assertEqual(result.branches[0].failure_type, "RuntimeError")
            self.assertEqual(result.branches[1].status, "PASSED")
            self.assertEqual(result.selected_branch_id, "b2")
            self.assertEqual((child_b2 / "CODEX_TASK.md").read_text(), "objective")

    def test_triton_preflight_prevents_worker_and_budget_use_on_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            (child / "solution").mkdir(parents=True)
            (child / "solution" / "kernel.py").write_text("# candidate\n")
            (child / "config.toml").write_text("[build]\nlanguage = \"python\"\n")

            class NeverRunRuntime:
                def prepare_workspace(self, workspace, task):
                    raise AssertionError("worker must not start after preflight failure")

                def run(self, request):
                    raise AssertionError("worker must not run after preflight failure")

            task = TaskContract("objective", ("solution/kernel.py",), "bash scripts/bench.sh", "oracle", ("no evaluator edit",), ("stop",))
            controller = TwoBranchController(
                project_root=root, task=task, operator="op", dataset=root, parent_kernel=child / "solution" / "kernel.py",
                parent_id="parent", parent_speedup=1.0, round_id="test-round", spawn_child=lambda _: child,
            )
            with patch("campaign.controller.materialize_build_language", return_value="triton"), patch(
                "campaign.controller.RuntimeRegistry.create", return_value=NeverRunRuntime()
            ):
                result = controller._run_branch(_spec("b1", "custom_triton_elementwise", "triton"))
            self.assertEqual(result.failure_type, "BuildLanguageConfigurationError")
            self.assertEqual(result.benchmark_calls, 0)
            self.assertFalse((child / ".ako" / "branch-budget.json").exists())

    def test_triton_language_is_materialized_before_worker_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            (child / "solution").mkdir(parents=True)
            (child / "solution" / "kernel.py").write_text("# candidate\n")
            (child / "config.toml").write_text("[build]\nlanguage = \"python\"\n")
            seen: list[str | None] = []

            class FakeRuntime:
                def prepare_workspace(self, workspace, task):
                    self.task = task

                def run(self, request):
                    seen.append(read_build_language(request.workspace / "config.toml"))
                    report = request.workspace / "trajectory" / "one" / "results.json"
                    report.parent.mkdir(parents=True)
                    report.write_text(json.dumps({"results": {"task": {"work": {"status": "PASSED", "speedup_factor": 1.2}}}}))
                    return AgentRunResult("session", "completed", 0, "done", (), False, 0.1, "")

            task = TaskContract("objective", ("solution/kernel.py",), "bash scripts/bench.sh", "oracle", ("no evaluator edit",), ("stop",))
            controller = TwoBranchController(
                project_root=root, task=task, operator="op", dataset=root, parent_kernel=child / "solution" / "kernel.py",
                parent_id="parent", parent_speedup=1.0, round_id="test-round", spawn_child=lambda _: child,
            )
            with patch("campaign.controller.RuntimeRegistry.create", return_value=FakeRuntime()):
                result = controller._run_branch(_spec("b1", "custom_triton_elementwise", "triton"))
            self.assertEqual(seen, ["triton"])
            self.assertEqual(read_build_language(child / "config.toml"), "triton")
            self.assertEqual(result.status, "PASSED")
            self.assertEqual(task.editable_files, ("solution/kernel.py",))


if __name__ == "__main__":
    unittest.main()

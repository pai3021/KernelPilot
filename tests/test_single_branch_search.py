"""CPU-only acceptance of a genuine one-plan, one-worker feedback control."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_runtime import AgentRunResult, TaskContract
from benchmark_backend.budget import (
    BUDGET_FILE_ENV, INVOCATION_TOKEN_ENV, BenchmarkBudgetExhausted,
    ensure_evaluation_authorized, read_budget,
)
from benchmark_backend.provenance import capture_evaluated_candidate
from campaign.controller import TwoBranchController
from campaign.models import BranchSpec, RoundState
from campaign.planner import BranchPlanner
from campaign.task_facts import build_task_facts


def contract():
    return TaskContract("Optimize the fixed task using handwritten Triton.", ("solution/kernel.py",),
                        "bash scripts/bench.sh", "Keep the fixed oracle and inputs.",
                        ("Do not change the evaluator or budget.",), ("Stop on infrastructure blockage.",))


def spec(branch="b1"):
    return BranchSpec(branch, "reference", "Try an initial layout, then adapt using feedback.",
                      "custom_triton_elementwise", "task facts", "lower latency", "Use the fixed runner.", "triton")


class SingleBranchSearchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = self.root / "parent.py"
        self.parent.write_text("class ModelNew: pass\n")
        self.facts = build_task_facts(task_id="task", task_source=self.parent, parent_source=self.parent,
                                     environment={"gpu": "test-only"}, evaluation_policy={"fixed": True})

    def tearDown(self):
        self.temp.cleanup()

    def planner_prompt(self, ids):
        return BranchPlanner("codex", self.root, benchmark_limit=6,
                             task_facts=self.facts, allow_strategy_adaptation=True)._prompt(
            parent_id="reference", parent_speedup=1, task=contract(), retry=False,
            branch_ids=ids, planning_context="")

    def test_single_plan_schema_does_not_generate_and_discard_a_second_hypothesis(self):
        prompt = self.planner_prompt(("b1",))
        schema = json.loads(prompt.split("Schema:\n", 1)[1].split("\n\n", 1)[0])
        self.assertEqual(len(schema["branches"]), 1)
        self.assertEqual(schema["branches"][0]["branch_id"], "b1")
        self.assertIn("Plan exactly one", prompt)
        self.assertNotIn("values must differ", prompt)
        self.assertIn("at most 6", prompt)
        self.assertIn(self.facts["task"]["sha256"], prompt)
        self.assertIn("Workers may revise strategy", prompt)

    def test_single_plan_rejects_wrong_count_identity_and_parent(self):
        good = {"branches": [spec().to_dict()]}
        self.assertEqual(len(BranchPlanner.parse(json.dumps(good), parent_id="reference", branch_ids=("b1",))), 1)
        for invalid in ({"branches": []}, {"branches": [spec().to_dict(), spec("b2").to_dict()]},
                        {"branches": [spec("b2").to_dict()]}):
            with self.assertRaises(ValueError):
                BranchPlanner.parse(json.dumps(invalid), parent_id="reference", branch_ids=("b1",))
        with self.assertRaises(ValueError):
            BranchPlanner.parse(json.dumps(good), parent_id="different", branch_ids=("b1",))

    def test_default_two_branch_contract_still_requires_diverse_directions(self):
        prompt = self.planner_prompt(("b1", "b2"))
        schema = json.loads(prompt.split("Schema:\n", 1)[1].split("\n\n", 1)[0])
        self.assertEqual(len(schema["branches"]), 2)
        self.assertIn("values must differ", prompt)
        with self.assertRaises(ValueError):
            BranchPlanner.parse(json.dumps({"branches": [spec().to_dict()]}), parent_id="reference")
        with self.assertRaisesRegex(ValueError, "strategy_family"):
            BranchPlanner.parse(json.dumps({"branches": [spec().to_dict(), spec("b2").to_dict()]}),
                                parent_id="reference")

    def test_invalid_branch_allocation_fails_before_any_runtime_is_created(self):
        for ids in ((), ("b1", "b2", "b3"), ("b1", "b1"), ("",), (False,)):
            with self.subTest(ids=ids), patch("campaign.planner.RuntimeRegistry.create") as create:
                planner = BranchPlanner("codex", self.root)
                with self.assertRaises(ValueError):
                    planner.plan(workspace=self.root, parent_id="reference", parent_speedup=1,
                                 task=contract(), branch_ids=ids)
                create.assert_not_called()
                with self.assertRaises(ValueError):
                    TwoBranchController(project_root=self.root, task=contract(), operator="task", dataset=self.root,
                                         parent_kernel=self.parent, parent_id="reference", parent_speedup=1,
                                         branch_ids=ids)

    def test_one_planner_one_worker_six_feedback_slots_preserves_earlier_best(self):
        requests, spawned = [], []
        def spawn(branch):
            spawned.append(branch)
            workspace = self.root / branch
            (workspace / "solution").mkdir(parents=True)
            (workspace / "solution/kernel.py").write_bytes(self.parent.read_bytes())
            (workspace / "config.toml").write_text('[build]\nlanguage="triton"\n')
            return workspace
        class Runtime:
            def prepare_workspace(inner, workspace, task):
                pass
            def run(inner, request):
                requests.append(request)
                if request.workspace.name == "planner":
                    return AgentRunResult("planner", "completed", 0,
                                          json.dumps({"branches": [spec().to_dict()]}), (), False, 0.1, "")
                ledger = Path(request.environment[BUDGET_FILE_ENV])
                self.assertEqual(read_budget(ledger)["benchmark_limit"], 6)
                self.assertEqual(request.timeout, 1800)
                self.assertEqual(json.loads((request.workspace / "TASK_FACTS.json").read_text()), self.facts)
                for number, speedup in enumerate((None, 1.4, 1.2, 1.8, 1.3, None), start=1):
                    # Simulate six distinct official Runner processes without GPU work.
                    with patch.dict(os.environ, request.environment):
                        os.environ.pop(INVOCATION_TOKEN_ENV, None)
                        ensure_evaluation_authorized()
                    kernel = request.workspace / "solution/kernel.py"
                    kernel.write_text(f"# measured attempt {number}\n")
                    witness = capture_evaluated_candidate(request.workspace)
                    report = request.workspace / "trajectory" / str(number) / "results.json"
                    report.parent.mkdir(parents=True)
                    report.write_text(json.dumps({"evaluated_candidate": witness, "results": {
                        "task": {"task": {"status": "PASSED" if speedup else "COMPILE_ERROR",
                                          "speedup_factor": speedup}}}}))
                with patch.dict(os.environ, request.environment):
                    os.environ.pop(INVOCATION_TOKEN_ENV, None)
                    with self.assertRaises(BenchmarkBudgetExhausted):
                        ensure_evaluation_authorized()
                return AgentRunResult("worker", "completed", 0, "done", (), False, 0.1, "")
        controller = TwoBranchController(project_root=self.root, task=contract(), operator="task", dataset=self.root,
                                         parent_kernel=self.parent, parent_id="reference", parent_speedup=1,
                                         branch_ids=("b1",), branch_benchmark_limit=6, worker_timeout=1800,
                                         planning_facts=self.facts, allow_strategy_adaptation=True, spawn_child=spawn)
        with patch("campaign.planner.RuntimeRegistry.create", return_value=Runtime()):
            result = controller.run()
        self.assertEqual(len(requests), 2)  # One real planner schema, one continuous worker context.
        self.assertEqual(spawned, ["b1"])
        self.assertEqual(result.state, RoundState.PROMOTED)
        self.assertEqual(len(result.branches), 1)
        branch = result.branches[0]
        self.assertEqual(branch.benchmark_calls, 6)
        self.assertEqual(branch.speedup_factor, 1.8)
        self.assertEqual(branch.candidate_path.read_text(), "# measured attempt 4\n")
        self.assertEqual(result.selected_branch_id, "b1")
        self.assertEqual(read_budget(self.root / "b1/.ako/branch-budget.json")["accounting_policy"], "dispatch")

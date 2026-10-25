from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_runtime import TaskContract
from campaign.controller import TwoBranchController
from campaign.models import BranchSpec
from campaign.planner import BranchPlanner
from campaign.task_facts import build_task_facts


def contract():
    return TaskContract("optimize", ("solution/kernel.py",), "bash scripts/bench.sh",
                        "fixed oracle", ("no evaluator changes",), ("budget exhausted",))


class InformedPlanningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "task.py"
        self.parent = self.root / "parent.py"
        self.marker = self.root / "MUST_NOT_EXIST"
        self.source.write_text(f"from pathlib import Path\nPath({str(self.marker)!r}).touch()\nN = 4096\n")
        self.parent.write_text("class ModelNew: pass\n")
        self.facts = build_task_facts(task_id="matmul", task_source=self.source,
                                     parent_source=self.parent, environment={"gpu": "RTX4090"},
                                     evaluation_policy={"atol": 1e-4, "correct_trials": 3})

    def tearDown(self):
        self.temp.cleanup()

    def test_fact_builder_reads_without_executing_and_binds_both_sources(self):
        self.assertFalse(self.marker.exists())
        self.assertIn("N = 4096", self.facts["task"]["source"])
        for path, key in ((self.source, "task"), (self.parent, "parent_candidate")):
            self.assertEqual(self.facts[key]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(self.facts["evaluation_policy"]["correct_trials"], 3)

    def test_missing_empty_or_unpersistable_facts_fail_before_agent_start(self):
        for path in (self.root / "missing.py", self.root / "empty.py"):
            if path.name == "empty.py":
                path.write_text("")
            with self.assertRaises((OSError, ValueError)):
                build_task_facts(task_id="task", task_source=path, parent_source=self.parent,
                                 environment={}, evaluation_policy={})
        with self.assertRaises(ValueError):
            build_task_facts(task_id="task", task_source=self.source, parent_source=self.parent,
                             environment={"unknown": float("nan")}, evaluation_policy={})

    def test_planner_receives_real_sources_without_needing_file_tools(self):
        planner = BranchPlanner("codex", self.root, benchmark_limit=3,
                                task_facts=self.facts, allow_strategy_adaptation=True)
        prompt = planner._prompt(parent_id="parent", parent_speedup=1.0, task=contract(),
                                 retry=False, branch_ids=("b1", "b2"), planning_context="")
        self.assertIn("N = 4096", prompt)
        self.assertIn("class ModelNew: pass", prompt)
        self.assertIn(self.facts["task"]["sha256"], prompt)
        self.assertIn("do not impose restrictions absent from the task contract", prompt)
        self.assertNotIn("pursue only its assigned hypothesis", prompt)
        self.assertIn("at most 3", prompt)

    def test_adaptation_overrides_strategy_lock_but_preserves_hard_constraints(self):
        spec = BranchSpec("b1", "parent", "hypothesis", "grouped", "reason", "signal",
                          "Never switch the initial strategy", "triton")
        prompt = TwoBranchController._worker_prompt(spec, 3, True)
        self.assertIn("you may change algorithm family", prompt)
        self.assertIn("overrides strategy-locking language", prompt)
        self.assertIn("never overrides the oracle", prompt)
        self.assertIn("at most 3", prompt)
        legacy = BranchPlanner("codex", self.root)._prompt(
            parent_id="p", parent_speedup=1, task=contract(), retry=False,
            branch_ids=("b1", "b2"), planning_context="")
        self.assertIn("pursue only its assigned hypothesis", legacy)

    def test_controller_snapshots_and_passes_the_facts_to_the_planner(self):
        captured = {}
        specs = tuple(BranchSpec(b, "parent", "hypothesis", b, "reason", "signal", "worker", "triton")
                      for b in ("b1", "b2"))
        class Planner:
            def __init__(inner, **kwargs):
                captured.update(kwargs)
            def plan(inner, **kwargs):
                return specs, ()
        controller = TwoBranchController(project_root=self.root, task=contract(), operator="task",
                         dataset=self.root, parent_kernel=self.parent, parent_id="parent",
                         parent_speedup=1.0, planning_facts=self.facts, allow_strategy_adaptation=True)
        self.facts["environment"]["gpu"] = "mutated later"
        with patch("campaign.controller.BranchPlanner", Planner):
            controller._plan()
        self.assertEqual(captured["task_facts"]["environment"]["gpu"], "RTX4090")
        self.assertTrue(captured["allow_strategy_adaptation"])
        saved=json.loads((controller.round_dir / "planning-facts.json").read_text())
        self.assertEqual(saved, captured["task_facts"])

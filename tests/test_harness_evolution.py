from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from harness_evolution import EvolutionCoordinator, HarnessProposal, RetrospectiveContext, SelfEvolutionConfig, load_config
from harness_evolution.gate import ProposalGate
from harness_evolution.retrospective import Retrospective, RetrospectiveOutputError
from agent_runtime import AgentRunResult, RuntimeRegistry
from campaign.multi_round import TwoRoundCampaign


def context() -> RetrospectiveContext:
    return RetrospectiveContext(
        campaign_id="test-campaign", task_signature={"benchmark": "KernelBench"},
        round_summary={"result": "complete"},
        known_harness_events=({"event_id": "observed", "evidence": "actual artifact"},),
    )


def proposal(*, target: str = "templates/skills/bench/SKILL.md", resolution_key: str = "") -> HarnessProposal:
    return HarnessProposal(
        "proposal-1", "A concrete workflow gap made a branch waste one action.", ("campaign/round-001/branch-b1",),
        "guidance", target, "Existing guidance lacks the observed warning.",
        "Add a concise, evidence-bound warning before the affected workflow step.",
        "Future workers recognize this validated pitfall.", "low", "workflow", ("EXP-1",), resolution_key,
    )


class HarnessEvolutionTest(unittest.TestCase):
    def test_defaults_enable_advanced_path_but_off_path_does_nothing(self):
        self.assertTrue(SelfEvolutionConfig().enabled)
        self.assertEqual(SelfEvolutionConfig().runtime, "codex")
        self.assertEqual(load_config(Path.cwd() / "harness_evolution/defaults.toml").runtime, "codex")
        self.assertTrue(TwoRoundCampaign(project_root=Path.cwd(), campaign_id="unit-default").self_evolution.enabled)
        with tempfile.TemporaryDirectory() as directory:
            coordinator = EvolutionCoordinator(project_root=Path(directory), config=SelfEvolutionConfig(enabled=False))
            result = coordinator.after_campaign(context())
            self.assertEqual(result["status"], "DISABLED")
            self.assertFalse(result["retrospective_invoked"])
            self.assertFalse((Path(directory) / "artifacts/evolution").exists())

    def _runtime_result(self, message: str) -> AgentRunResult:
        return AgentRunResult("session", "completed", 0, message, (), False, 0.01, "")

    def test_codex_retrospective_uses_registry_and_parses_proposal(self):
        runtime = MagicMock()
        runtime.run.return_value = self._runtime_result(json.dumps({"proposal": proposal().to_dict()}))
        with tempfile.TemporaryDirectory() as directory, patch("harness_evolution.retrospective.RuntimeRegistry.create", return_value=runtime) as create:
            observed = Retrospective(project_root=Path(directory), runtime_name="codex").run(workspace=Path(directory) / "r", context=context())
        self.assertEqual(observed, proposal())
        create.assert_called_once_with("codex", template_root=Path(directory))
        self.assertEqual(runtime.run.call_count, 1)

    def test_codex_retrospective_accepts_no_proposal(self):
        runtime = MagicMock()
        runtime.run.return_value = self._runtime_result('{"proposal": null}')
        with tempfile.TemporaryDirectory() as directory, patch("harness_evolution.retrospective.RuntimeRegistry.create", return_value=runtime):
            self.assertIsNone(Retrospective(project_root=Path(directory), runtime_name="codex").run(workspace=Path(directory) / "r", context=context()))
        self.assertEqual(runtime.run.call_count, 1)

    def test_invalid_payload_gets_one_repair_only(self):
        runtime = MagicMock()
        runtime.run.side_effect = [self._runtime_result("not json"), self._runtime_result('{"proposal": null}')]
        with tempfile.TemporaryDirectory() as directory, patch("harness_evolution.retrospective.RuntimeRegistry.create", return_value=runtime):
            observed = Retrospective(project_root=Path(directory), runtime_name="codex").run(workspace=Path(directory) / "r", context=context())
        self.assertIsNone(observed)
        self.assertEqual(runtime.run.call_count, 2)
        self.assertEqual(runtime.run.call_args_list[1].args[0].artifact_label, "retrospective-repair")

    def test_second_invalid_payload_is_bounded_failure(self):
        runtime = MagicMock()
        runtime.run.side_effect = [self._runtime_result("invalid"), self._runtime_result("still invalid")]
        with tempfile.TemporaryDirectory() as directory, patch("harness_evolution.retrospective.RuntimeRegistry.create", return_value=runtime):
            with self.assertRaises(RetrospectiveOutputError):
                Retrospective(project_root=Path(directory), runtime_name="codex").run(workspace=Path(directory) / "r", context=context())
        self.assertEqual(runtime.run.call_count, 2)

    def test_claude_runtime_remains_registered_as_alternative(self):
        self.assertEqual(RuntimeRegistry.create("claude", template_root=Path.cwd()).name, "claude")

    def test_proposal_round_trip(self):
        self.assertEqual(HarnessProposal.from_dict(proposal().to_dict()), proposal())
        payload = proposal().to_dict()
        payload["evidence"] = "one concrete artifact"
        self.assertEqual(HarnessProposal.from_dict(payload).evidence, ("one concrete artifact",))

    def test_frozen_and_security_targets_are_rejected(self):
        gate = ProposalGate(project_root=Path.cwd())
        self.assertFalse(gate.evaluate(proposal(target="benchmark_backend/ssh_remote.py")).scope_passed)
        self.assertTrue(gate.evaluate(proposal(target="templates/skills/bench/SKILL.md", resolution_key="",)).scope_passed)
        security = HarnessProposal("proposal-2", "Evidence-backed issue", ("artifact",), "guidance", "templates/skills/bench/SKILL.md", "current", "Add SSH credential guidance.", "effect", "low", "workflow")
        self.assertEqual(gate.evaluate(security).decision, "REJECT")

    def test_missing_evidence_and_already_resolved_are_rejected(self):
        gate = ProposalGate(project_root=Path.cwd())
        with self.assertRaises(ValueError):
            HarnessProposal("proposal-3", "problem", (), "guidance", "templates/task.md", "current", "change", "effect", "low", "workflow")
        decision = gate.evaluate(proposal(resolution_key="harness_level_benchmark_budget"))
        self.assertEqual(decision.decision, "REJECT_AS_ALREADY_RESOLVED")

    def test_no_proposal_records_ledger_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("harness_evolution.coordinator.Retrospective.run", return_value=None):
                result = EvolutionCoordinator(project_root=root, config=SelfEvolutionConfig()).after_campaign(context())
            self.assertEqual(result["status"], "NO_PROPOSAL")
            self.assertFalse((root / "harness_evolution").exists())
            ledger = root / "artifacts/evolution/harness-ledger.jsonl"
            self.assertEqual(json.loads(ledger.read_text())["decision"], "NO_PROPOSAL")

    def test_accepted_proposal_is_scoped_text_mutation_and_increments_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "templates/skills/bench/SKILL.md"
            target.parent.mkdir(parents=True)
            target.write_text("# guidance\n")
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_contract.py").write_text("import unittest\nclass T(unittest.TestCase):\n def test_ok(self): self.assertTrue(True)\n")
            with patch("harness_evolution.coordinator.Retrospective.run", return_value=proposal()):
                result = EvolutionCoordinator(project_root=root, config=SelfEvolutionConfig()).after_campaign(context())
            self.assertEqual(result["status"], "ACCEPT")
            self.assertIn("evidence-bound warning", target.read_text())
            self.assertEqual(json.loads((root / "artifacts/evolution/harness-version.json").read_text())["version"], "H1")
            self.assertTrue((root / "artifacts/evolution/test-campaign-retrospective/candidate/templates/skills/bench/SKILL.md").is_file())

    def test_rejected_proposal_does_not_mutate_harness_or_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "templates/skills/bench/SKILL.md"
            target.parent.mkdir(parents=True)
            target.write_text("# unchanged\n")
            memory = root / "artifacts/memory/experiences.jsonl"
            memory.parent.mkdir(parents=True)
            memory.write_text('{"immutable": true}\n')
            before = (target.read_bytes(), memory.read_bytes())
            with patch("harness_evolution.coordinator.Retrospective.run", return_value=proposal(resolution_key="harness_level_benchmark_budget")):
                result = EvolutionCoordinator(project_root=root, config=SelfEvolutionConfig()).after_campaign(context())
            self.assertEqual(result["status"], "REJECT_AS_ALREADY_RESOLVED")
            self.assertEqual(before, (target.read_bytes(), memory.read_bytes()))


if __name__ == "__main__":
    unittest.main()

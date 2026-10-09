"""Phase 2B: import Phase 2A history and execute only adaptive round two."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agent_runtime import TaskContract
from benchmark_backend.budget import initialize_budget, read_budget, reserve_budget
from experience_memory import ExperienceStore
from harness_evolution import EvolutionCoordinator, RetrospectiveContext, SelfEvolutionConfig

from .controller import TwoBranchController
from .models import RoundResult
from .state import CampaignState, ParentSnapshot, RoundRecord


class TwoRoundCampaign:
    """A bounded campaign: two historical Round-1 branches plus Round-2 only."""

    def __init__(self, *, project_root: Path, campaign_id: str, worker_runtime: str = "codex", planner_runtime: str = "codex", self_evolution: SelfEvolutionConfig | None = None) -> None:
        self.project_root = Path(project_root)
        self.campaign_id = campaign_id
        self.worker_runtime = worker_runtime
        self.planner_runtime = planner_runtime
        self.self_evolution = self_evolution or SelfEvolutionConfig()
        self.root = self.project_root / "artifacts" / "phase2b" / campaign_id
        self.state_path = self.root / "campaign-state.json"
        self.budget_path = self.root / "campaign-budget.json"
        self.lineage_path = self.root / "lineage.json"

    def _seed_from_phase2a(self) -> CampaignState:
        source = self.project_root / "artifacts" / "phase2a" / "phase2a-r001" / "round-result-recovered.json"
        data = json.loads(source.read_text())
        if data.get("promotion_decision") != "PROMOTE" or data.get("selected_branch_id") != "b2":
            raise ValueError("Phase 2A recovered result must promote b2 before Phase 2B can continue")
        branches = data.get("branches")
        if not isinstance(branches, list):
            raise ValueError("Phase 2A recovered result lacks branches")
        winner = next((item for item in branches if item.get("branch_id") == "b2"), None)
        if not isinstance(winner, Mapping) or not isinstance(winner.get("benchmark_result"), Mapping):
            raise ValueError("Phase 2A winner has no standard benchmark result")
        workspace_name = Path(str(winner["artifacts"]["workspace"])).name
        candidate = self.project_root / "artifacts" / "children" / workspace_name / "solution" / "kernel.py"
        if not candidate.is_file():
            raise FileNotFoundError(f"Promoted Phase 2A candidate is absent: {candidate}")
        parent = ParentSnapshot("phase2a-r001-b2", "b2", candidate, dict(winner["benchmark_result"]), "phase2a-r001")
        record = RoundRecord(
            round_id="phase2a-r001", parent_id=str(data["parent_id"]),
            parent_result={"speedup_factor": data["parent_speedup"]},
            branch_specs=tuple(json.loads((self.project_root / "artifacts" / "phase2a" / "phase2a-r001" / "branch-specs.json").read_text())),
            branch_results=tuple(branches), winner="b2", promotion_decision="PROMOTE", selected_parent=parent.parent_id,
        )
        return CampaignState(
            campaign_id=self.campaign_id, task_id="1_Square_matrix_multiplication_", round_index=1,
            initial_parent=parent, current_parent=parent, best_result=dict(winner["benchmark_result"]),
            round_history=[record], benchmark_budget_total=4, benchmark_budget_used=2,
        )

    def _initialize(self) -> CampaignState:
        if self.root.exists():
            raise FileExistsError(f"campaign artifact directory already exists: {self.root}")
        self.root.mkdir(parents=True)
        state = self._seed_from_phase2a()
        initialize_budget(self.budget_path, limit=4)
        ledger = read_budget(self.budget_path)
        ledger.update({"benchmark_used": 2, "issued_tokens": ["imported-phase2a-b1", "imported-phase2a-b2"], "round_limit": 2, "rounds_completed": 1})
        self.budget_path.write_text(json.dumps(ledger, indent=2) + "\n")
        state.save(self.state_path)
        self.lineage_path.write_text(json.dumps({"initial_parent": parent_dict(state.initial_parent), "round_001_winner": parent_dict(state.current_parent)}, indent=2) + "\n")
        return state

    @staticmethod
    def _planning_context(state: CampaignState) -> str:
        history = state.round_history[-1]
        feedback = [
            f"Current promoted parent: {state.current_parent.parent_id}",
            f"Current benchmark: correctness {state.best_result.get('status')}, speedup {state.best_result.get('speedup_factor')}x.",
            "Previous branches:",
        ]
        specs_by_id = {str(spec.get("branch_id")): spec for spec in history.branch_specs}
        for result in history.branch_results:
            benchmark = result.get("benchmark_result") if isinstance(result, Mapping) else {}
            spec = specs_by_id.get(str(result.get("branch_id")), {})
            feedback.append(f"- {result.get('branch_id')}: strategy_family={spec.get('strategy_family')}; status={result.get('failure_type') or benchmark.get('status')}.")
        feedback.extend(["One branch should exploit the successful parent direction; the other should explore a materially different direction.", "Do not blindly repeat the same failed implementation."])
        return "\n".join(feedback)

    def run_round_two(self, *, resume: bool = False) -> CampaignState:
        if resume:
            if not self.state_path.is_file():
                raise FileNotFoundError(f"round-boundary resume needs {self.state_path}")
            state = CampaignState.load(self.state_path)
            if state.round_index != 1 or state.status != "RUNNING":
                raise ValueError("campaign is not at a resumable Round-1 boundary")
        else:
            state = self._initialize()
        task = TaskContract(
            objective="Optimize KernelBench Level 1 Square Matrix Multiplication (N=4096).",
            editable_files=("solution/kernel.py",), benchmark_command='bash scripts/bench.sh --label "branch result"',
            correctness_contract="Candidate must pass the fixed KernelBench correctness oracle before latency ranking.",
            forbidden_changes=("Do not modify evaluator, oracle, reference implementation, remote backend, thresholds, task definition, or budget ledger.",),
            stop_conditions=("Stop after the branch budget's one formal benchmark evaluation.",),
        )
        def reserve_campaign_slot(_spec: Any) -> None:
            reserve_budget(self.budget_path)
        controller = TwoBranchController(
            project_root=self.project_root, task=task, operator="1_Square_matrix_multiplication_",
            dataset=self.project_root / "artifacts" / "deps" / "KernelBench", parent_kernel=state.current_parent.candidate_path,
            parent_id=state.current_parent.parent_id, parent_speedup=float(state.best_result["speedup_factor"]),
            worker_runtime=self.worker_runtime, planner_runtime=self.planner_runtime, round_id="round-002",
            artifact_root=self.root, child_label_prefix=f"phase2b-{self.campaign_id}", branch_ids=("b3", "b4"),
            planning_context=self._planning_context(state), before_branch=reserve_campaign_slot,
        )
        result = controller.run()
        self._record_round_two(state, result)
        if self.self_evolution.enabled and self.self_evolution.after_campaign:
            EvolutionCoordinator(project_root=self.project_root, config=self.self_evolution).after_campaign(
                self._retrospective_context(state)
            )
        return state

    def _retrospective_context(self, state: CampaignState) -> RetrospectiveContext:
        """Bounded campaign evidence for optional Mode-3-style reflection."""
        records = ExperienceStore(self.project_root / "artifacts/memory/experiences.jsonl").list()[:3]
        latest = state.round_history[-1]
        branches = tuple(item for item in latest.branch_results if isinstance(item, Mapping))
        succeeded = tuple(item for item in branches if (item.get("benchmark_result") or {}).get("status") == "PASSED")
        failed = tuple(item for item in branches if item not in succeeded)
        return RetrospectiveContext(
            campaign_id=state.campaign_id,
            task_signature={"benchmark": "KernelBench", "task_id": state.task_id, "hardware": "RTX4090"},
            round_summary={"round_id": latest.round_id, "promotion_decision": latest.promotion_decision, "selected_parent": latest.selected_parent},
            successful_branches=succeeded,
            failed_branches=failed,
            promotion_history=tuple({"round_id": item.round_id, "decision": item.promotion_decision, "winner": item.winner} for item in state.round_history),
            budget_summary={"used": state.benchmark_budget_used, "total": state.benchmark_budget_total},
            relevant_experiences=tuple({"experience_id": record.experience_id, "outcome": record.outcome, "lesson": record.lesson} for record in records),
            known_harness_events=(),
        )

    def _record_round_two(self, state: CampaignState, result: RoundResult) -> None:
        round_path = self.root / "round-002.json"
        round_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        spec_path = self.root / "round-002" / "branch-specs.json"
        specs = tuple(json.loads(spec_path.read_text())) if spec_path.is_file() else ()
        selected_parent = f"{result.round_id}-{result.selected_branch_id}" if result.selected_branch_id else state.current_parent.parent_id
        state.round_history.append(RoundRecord(result.round_id, result.parent_id, {"speedup_factor": result.parent_speedup}, specs, tuple(item.to_dict() for item in result.branches), result.selected_branch_id, result.promotion_decision, selected_parent))
        ledger = read_budget(self.budget_path)
        ledger["rounds_completed"] = 2
        self.budget_path.write_text(json.dumps(ledger, indent=2) + "\n")
        state.benchmark_budget_used = int(ledger["benchmark_used"])
        if result.selected_branch_id:
            winner = next(item for item in result.branches if item.branch_id == result.selected_branch_id)
            state.current_parent = ParentSnapshot(f"{result.round_id}-{winner.branch_id}", winner.branch_id, winner.candidate_path, dict(winner.benchmark_result or {}), result.round_id, winner.diff)
            state.best_result = dict(winner.benchmark_result or {})
        state.round_index = 2
        state.status = "COMPLETED"
        state.termination_reason = "MAX_ROUNDS" if state.benchmark_budget_used < state.benchmark_budget_total else "BUDGET_EXHAUSTED"
        state.save(self.state_path)
        self.lineage_path.write_text(json.dumps({"initial_parent": parent_dict(state.initial_parent), "round_001_winner": parent_dict(state.round_history and state.initial_parent), "final_parent": parent_dict(state.current_parent)}, indent=2) + "\n")
        (self.root / "final-result.json").write_text(json.dumps({"campaign_id": state.campaign_id, "status": state.status, "termination_reason": state.termination_reason, "best_parent": parent_dict(state.current_parent), "benchmark_budget_used": state.benchmark_budget_used}, indent=2) + "\n")


def parent_dict(parent: ParentSnapshot) -> dict[str, Any]:
    return parent.to_dict()

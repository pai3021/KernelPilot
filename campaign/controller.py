"""Serial 1-round × 2-branch controller for Phase 2A."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from agent_runtime import AgentRunRequest, RuntimeRegistry, TaskContract

from benchmark_backend.budget import BUDGET_FILE_ENV, initialize_budget, read_budget
from .build_language import materialize_build_language, validate_build_language
from .models import BranchResult, BranchSpec, RoundResult, RoundState
from .planner import BranchPlanner, PlanningFailure
from .ranking import rank_candidates


def workload_result_from_report(path: Path) -> Mapping[str, Any] | None:
    """Return the one standard KernelBench workload result from a report."""
    try:
        report = json.loads(path.read_text())
        outer = report.get("results", {})
        if not isinstance(outer, Mapping):
            return None
        for group in outer.values():
            if isinstance(group, Mapping):
                for workload in group.values():
                    if isinstance(workload, Mapping):
                        return workload
    except (OSError, json.JSONDecodeError):
        return None
    return None


def latest_report(workspace: Path) -> Path | None:
    reports = list((workspace / "trajectory").glob("*/results.json"))
    return max(reports, key=lambda path: path.stat().st_mtime) if reports else None


class TwoBranchController:
    """One deterministic round; runtimes are looked up only through registry."""

    def __init__(
        self,
        *,
        project_root: Path,
        task: TaskContract,
        operator: str,
        dataset: Path,
        parent_kernel: Path,
        parent_id: str,
        parent_speedup: float,
        worker_runtime: str = "codex",
        planner_runtime: str = "codex",
        runtime_model: str | None = None,
        runtime_reasoning_effort: str | None = None,
        round_id: str = "round-001",
        spawn_child: Callable[[str], Path] | None = None,
        artifact_root: Path | None = None,
        child_label_prefix: str = "phase2a",
        branch_ids: tuple[str, str] = ("b1", "b2"),
        planning_context: str = "",
        before_branch: Callable[[BranchSpec], None] | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.task = task
        self.operator = operator
        self.dataset = Path(dataset)
        self.parent_kernel = Path(parent_kernel)
        self.parent_id = parent_id
        self.parent_speedup = parent_speedup
        self.worker_runtime = worker_runtime
        self.planner_runtime = planner_runtime
        self.runtime_model = runtime_model
        self.runtime_reasoning_effort = runtime_reasoning_effort
        self.round_id = round_id
        self.round_dir = (Path(artifact_root) / round_id) if artifact_root else self.project_root / "artifacts" / "phase2a" / round_id
        self._spawn_child = spawn_child or self._spawn_via_existing_harness
        self.branch_ids = branch_ids
        self.planning_context = planning_context
        self.before_branch = before_branch
        self.child_label_prefix = child_label_prefix

    def _record_state(self, state: RoundState, **extra: Any) -> None:
        self.round_dir.mkdir(parents=True, exist_ok=True)
        (self.round_dir / "round-state.json").write_text(
            json.dumps({"round_id": self.round_id, "state": state.value, **extra}, indent=2) + "\n"
        )

    def _spawn_via_existing_harness(self, branch_id: str) -> Path:
        label = f"{self.child_label_prefix}-{self.round_id}-{branch_id}"
        command = [
            sys.executable, "spawn.py", "--operator", self.operator, "--backend", "ssh",
            "--gpu", "rtx4090", "--agent", self.worker_runtime, "--dataset", str(self.dataset),
            "--kernel", str(self.parent_kernel), "--name", label,
        ]
        completed = subprocess.run(command, cwd=self.project_root, text=True, capture_output=True, check=False)
        spawn_log = self.round_dir / f"{branch_id}-spawn.log"
        spawn_log.write_text(completed.stdout + completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"child spawn failed for {branch_id}: {completed.stderr.strip() or completed.stdout.strip()}")
        workspace = self.project_root / "artifacts" / "children" / f"kernelpilot-run-{label}"
        if not workspace.is_dir():
            raise RuntimeError(f"spawn reported success but workspace was absent: {workspace}")
        return workspace

    def _planner_workspace(self) -> Path:
        workspace = self.round_dir / "planner"
        workspace.mkdir(parents=True, exist_ok=True)
        # Codex exec requires a Git workspace. This contains planner-only task
        # instructions and no candidate, so planning cannot mutate a branch.
        if not (workspace / ".git").exists():
            completed = subprocess.run(["git", "init", "-q"], cwd=workspace, text=True, capture_output=True, check=False)
            if completed.returncode:
                raise RuntimeError(f"planner workspace git init failed: {completed.stderr.strip()}")
        return workspace

    def _plan(self) -> tuple[BranchSpec, BranchSpec]:
        planner = BranchPlanner(runtime_name=self.planner_runtime, template_root=self.project_root,
                                model=self.runtime_model, reasoning_effort=self.runtime_reasoning_effort)
        specs, attempts = planner.plan(
            workspace=self._planner_workspace(), parent_id=self.parent_id,
            parent_speedup=self.parent_speedup, task=self.task, branch_ids=self.branch_ids,
            planning_context=self.planning_context,
        )
        for index, attempt in enumerate(attempts, start=1):
            (self.round_dir / f"planner-attempt-{index}.json").write_text(json.dumps(attempt.to_dict(), indent=2) + "\n")
        (self.round_dir / "branch-specs.json").write_text(json.dumps([spec.to_dict() for spec in specs], indent=2) + "\n")
        return specs

    @staticmethod
    def _worker_prompt(spec: BranchSpec) -> str:
        return (
            f"You are Worker {spec.branch_id}. Work only on this assigned branch.\n\n"
            f"Hypothesis: {spec.hypothesis}\nStrategy family: {spec.strategy_family}\n"
            f"Reason: {spec.reason}\nExpected signal: {spec.expected_signal}\n\n"
            f"Planner guidance: {spec.worker_prompt}\n\n"
            "Read CODEX_TASK.md, the current candidate, and necessary task files. Make a real, correctness-preserving "
            "optimization attempt only in solution/kernel.py. Run the provided benchmark to evaluate it. The Harness "
            "enforces one formal benchmark evaluation; after it is consumed, do not attempt another evaluation. Do not "
            "modify the evaluator, oracle, reference implementation, remote backend, thresholds, task definition, or "
            "budget ledger. In your final response state: changed code, correctness, reference latency, candidate latency, "
            "speedup, and whether this branch supports or rejects its hypothesis."
        )

    def _branch_result(self, spec: BranchSpec, workspace: Path, run: Any | None, error: Exception | None) -> BranchResult:
        budget_path = workspace / ".ako" / "branch-budget.json"
        try:
            calls = int(read_budget(budget_path)["benchmark_used"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            calls = 0
        report_path = latest_report(workspace)
        benchmark = workload_result_from_report(report_path) if report_path else None
        candidate = workspace / "solution" / "kernel.py"
        candidate_sha256: str | None = None
        evaluated_candidate_path: Path | None = None
        evaluated_at: str | None = None
        provenance_path = workspace / ".ako" / "evaluated-candidate.json"
        try:
            provenance = json.loads(provenance_path.read_text())
            candidate_sha256 = str(provenance["candidate_sha256"])
            evaluated_candidate_path = Path(str(provenance["artifact_path"]))
            evaluated_at = str(provenance["evaluation_timestamp"])
            if evaluated_candidate_path.is_file():
                candidate = evaluated_candidate_path
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            pass
        diff_result = subprocess.run(["git", "diff", "--", "solution/kernel.py"], cwd=workspace, text=True, capture_output=True, check=False)
        artifacts: dict[str, Path] = {"workspace": workspace, "budget_ledger": budget_path}
        if report_path:
            artifacts["benchmark_result"] = report_path
        if provenance_path.is_file():
            artifacts["evaluated_candidate_provenance"] = provenance_path
        if evaluated_candidate_path is not None:
            artifacts["evaluated_candidate_snapshot"] = evaluated_candidate_path
        if run is not None:
            artifacts.update({f"runtime_{name}": Path(value) for name, value in run.artifacts.items()})
        status = "PASSED" if benchmark and benchmark.get("status") == "PASSED" else "FAILED"
        failure: str | None = None
        if error is not None:
            failure = type(error).__name__
        elif run is not None and run.timed_out:
            failure = "AGENT_TIMEOUT"
        elif run is not None and run.status != "completed":
            failure = "AGENT_FAILURE"
        elif benchmark is None:
            failure = "NO_BENCHMARK_RESULT"
        elif benchmark.get("status") != "PASSED":
            failure = "BENCHMARK_OR_CORRECTNESS_FAILURE"
        return BranchResult(
            branch_id=spec.branch_id, runtime=self.worker_runtime,
            session_id="" if run is None else run.session_id, status=status, candidate_path=candidate,
            diff=diff_result.stdout, benchmark_result=benchmark, benchmark_calls=calls,
            final_message="" if run is None else run.final_message, failure_type=failure,
            evaluated_candidate_sha256=candidate_sha256, evaluated_candidate_path=evaluated_candidate_path, artifacts=artifacts,
            evaluated_at=evaluated_at,
        )

    def _run_branch(self, spec: BranchSpec) -> BranchResult:
        workspace: Path | None = None
        run: Any | None = None
        try:
            workspace = self._spawn_child(spec.branch_id)
            if self.before_branch is not None:
                self.before_branch(spec)
            # Strategy metadata belongs to the planner, while child build
            # configuration belongs to the Harness.  Materialize and verify it
            # before a worker can spend either its formal benchmark budget or a
            # remote GPU evaluation.
            config_path = workspace / "config.toml"
            materialize_build_language(config_path, spec.strategy_family, spec.build_language)
            validate_build_language(config_path, spec.strategy_family, spec.build_language)
            ledger = workspace / ".ako" / "branch-budget.json"
            initialize_budget(ledger, limit=1)
            runtime = RuntimeRegistry.create(self.worker_runtime, template_root=self.project_root)
            # spawn materializes only a generic child contract.  The controller
            # owns the round-specific TaskContract, so make it the authoritative
            # runtime-native instruction immediately before the worker starts.
            runtime.prepare_workspace(workspace, self.task)
            run = runtime.run(AgentRunRequest(
                workspace=workspace, prompt=self._worker_prompt(spec), timeout=900,
                environment={BUDGET_FILE_ENV: str(ledger), **({} if self.runtime_reasoning_effort is None else {"AGENT_RUNTIME_REASONING_EFFORT": self.runtime_reasoning_effort})},
                model=self.runtime_model, artifact_label=f"branch-{spec.branch_id}",
            ))
            return self._branch_result(spec, workspace, run, None)
        except Exception as exc:  # Failure isolation is the controller contract.
            if workspace is None:
                workspace = self.round_dir / "failed-workspaces" / spec.branch_id
                workspace.mkdir(parents=True, exist_ok=True)
            return self._branch_result(spec, workspace, run, exc)

    def run(self) -> RoundResult:
        if self.round_dir.exists():
            raise FileExistsError(f"round artifact directory already exists: {self.round_dir}")
        self._record_state(RoundState.PLANNING)
        try:
            specs = self._plan()
        except PlanningFailure as exc:
            self._record_state(RoundState.FAILED, error=str(exc))
            result = RoundResult(self.round_id, RoundState.FAILED, self.parent_id, self.parent_speedup, (), None, "PLANNING_FAILURE", {"round_dir": self.round_dir})
            (self.round_dir / "round-result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
            return result
        self._record_state(RoundState.BRANCHES_READY)
        self._record_state(RoundState.RUNNING)
        branches = tuple(self._run_branch(spec) for spec in specs)
        self._record_state(RoundState.EVALUATING)
        decision = rank_candidates(branches, parent_speedup=self.parent_speedup)
        final_state = RoundState.PROMOTED if decision.decision == "PROMOTE" else RoundState.NO_IMPROVEMENT
        self._record_state(final_state, ranking=asdict(decision))
        result = RoundResult(
            self.round_id, final_state, self.parent_id, self.parent_speedup, branches,
            decision.selected_branch_id, decision.decision,
            {"round_dir": self.round_dir, "branch_specs": self.round_dir / "branch-specs.json", "ranking": self.round_dir / "round-state.json"},
        )
        (self.round_dir / "round-result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        return result

"""Structured two-branch planning through the existing AgentRuntime seam."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agent_runtime import AgentRunRequest, RuntimeRegistry, TaskContract

from .models import BranchSpec


class PlanningFailure(RuntimeError):
    """The planner exhausted its one structured-output retry."""


@dataclass
class BranchPlanner:
    runtime_name: str
    template_root: Path
    timeout: int = 240
    model: str | None = None
    reasoning_effort: str | None = None

    def _prompt(self, *, parent_id: str, parent_speedup: float, task: TaskContract, retry: bool, branch_ids: tuple[str, str], planning_context: str) -> str:
        correction = "\nYour preceding response was invalid. Return only the required JSON object." if retry else ""
        prompt = (
            "You are the KernelPilot BranchPlanner. Do not edit files and do not run commands. "
            "Return exactly one JSON object, with no Markdown fence or prose.\n\n"
            "Schema:\n"
            f'{{"branches":[{{"branch_id":"{branch_ids[0]}","parent_id":"...","hypothesis":"...",'
            '"strategy_family":"...","build_language":"python|triton|cuda|cpp|tilelang|cute","reason":"...","expected_signal":"...","worker_prompt":"..."},'
            f'{{"branch_id":"{branch_ids[1]}","parent_id":"...","hypothesis":"...","strategy_family":"...",'
            '"build_language":"python|triton|cuda|cpp|tilelang|cute","reason":"...","expected_signal":"...","worker_prompt":"..."}]}\n\n'
            f"Parent id: {parent_id}\nParent speedup: {parent_speedup:.6f}x\n"
            f"Task objective: {task.objective}\n"
            f"Editable files: {', '.join(task.editable_files)}\n"
            f"Benchmark command: {task.benchmark_command}\n"
        )
        prompt += (f"\nStructured prior-round summary (use this feedback; do not repeat it verbatim):\n{planning_context}\n" if planning_context else "")
        prompt += (
            "Plan exactly two practical and clearly different optimization directions. "
            "The two strategy_family values must differ materially, not merely a parameter change. "
            "Set build_language to the actual candidate implementation language; Triton kernels must use triton. "
            "Each worker_prompt must direct one isolated worker to pursue only its assigned hypothesis, "
            "preserve the evaluator/oracle/backend/thresholds, use at most one benchmark, and summarize the result."
            + correction
        )
        return prompt

    @staticmethod
    def parse(text: str, *, parent_id: str, branch_ids: tuple[str, str] = ("b1", "b2")) -> tuple[BranchSpec, BranchSpec]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"planner did not return a JSON object: {exc.msg}") from exc
        if not isinstance(payload, Mapping) or set(payload) != {"branches"}:
            raise ValueError("planner JSON must contain only a branches field")
        raw_branches = payload["branches"]
        if not isinstance(raw_branches, list) or len(raw_branches) != 2:
            raise ValueError("planner must return exactly two branches")
        if any(not isinstance(item, Mapping) or not isinstance(item.get("build_language"), str) or not item["build_language"].strip() for item in raw_branches):
            raise ValueError("planner must declare build_language for every branch")
        branches = tuple(BranchSpec.from_dict(item) for item in raw_branches if isinstance(item, Mapping))
        if len(branches) != 2:
            raise ValueError("each planner branch must be an object")
        if {branch.branch_id for branch in branches} != set(branch_ids):
            raise ValueError(f"branch ids must be exactly {branch_ids[0]} and {branch_ids[1]}")
        if any(branch.parent_id != parent_id for branch in branches):
            raise ValueError("every branch must refer to the requested parent")
        if branches[0].strategy_family == branches[1].strategy_family:
            raise ValueError("branch strategy_family values must differ")
        return branches  # type: ignore[return-value]

    def plan(self, *, workspace: Path, parent_id: str, parent_speedup: float, task: TaskContract, branch_ids: tuple[str, str] = ("b1", "b2"), planning_context: str = "") -> tuple[tuple[BranchSpec, BranchSpec], tuple[Any, ...]]:
        runtime = RuntimeRegistry.create(self.runtime_name, template_root=self.template_root)
        runtime.prepare_workspace(workspace, task)
        attempts: list[Any] = []
        for attempt in range(2):
            result = runtime.run(AgentRunRequest(
                workspace=workspace,
                prompt=self._prompt(parent_id=parent_id, parent_speedup=parent_speedup, task=task, retry=attempt == 1, branch_ids=branch_ids, planning_context=planning_context),
                timeout=self.timeout,
                model=self.model,
                environment={} if self.reasoning_effort is None else {"AGENT_RUNTIME_REASONING_EFFORT": self.reasoning_effort},
                artifact_label=f"planner-{attempt + 1}",
            ))
            attempts.append(result)
            if result.status != "completed":
                continue
            try:
                return self.parse(result.final_message, parent_id=parent_id, branch_ids=branch_ids), tuple(attempts)
            except ValueError:
                continue
        raise PlanningFailure("PLANNING_FAILURE: planner did not produce two valid, diverse BranchSpecs after one retry")

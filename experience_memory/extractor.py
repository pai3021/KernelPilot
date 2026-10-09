"""Extract final-campaign branch evidence into the append-only memory store."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from campaign.models import BranchResult, BranchSpec, RoundResult

from .models import ExperienceRecord, TaskSignature
from .store import ExperienceStore


def _outcome(branch: BranchResult, selected: bool, decision: str) -> str:
    result = branch.benchmark_result or {}
    if selected and decision == "PROMOTE":
        return "PROMOTED"
    status = str(result.get("status", ""))
    if status == "PASSED":
        return "VALID_NOT_PROMOTED"
    if status == "COMPILE_ERROR":
        return "COMPILE_FAILURE"
    if status == "INCORRECT_NUMERICAL":
        return "CORRECTNESS_FAILURE"
    if status == "TIMEOUT":
        return "TIMEOUT"
    if status == "RUNTIME_ERROR":
        return "RUNTIME_FAILURE"
    return "INFRA_FAILURE"


def extract_round(
    store: ExperienceStore,
    *,
    signature: TaskSignature,
    campaign_id: str,
    round_result: RoundResult,
    specs: Iterable[BranchSpec],
) -> list[ExperienceRecord]:
    """Persist evaluated candidates only; never learn from an unevaluated edit."""
    by_id = {spec.branch_id: spec for spec in specs}
    created: list[ExperienceRecord] = []
    for branch in round_result.branches:
        if not branch.evaluated_candidate_sha256:
            continue
        spec = by_id.get(branch.branch_id)
        if spec is None:
            continue
        outcome = _outcome(
            branch,
            branch.branch_id == round_result.selected_branch_id,
            round_result.promotion_decision,
        )
        result = dict(branch.benchmark_result or {})
        lesson = (
            f"{spec.strategy_family} produced {result.get('speedup_factor')}x under the fixed evaluator."
            if outcome in {"PROMOTED", "VALID_NOT_PROMOTED"}
            else f"{spec.strategy_family} ended as {outcome}; do not blindly replay this exact attempt."
        )
        refs = {name: str(path) for name, path in branch.artifacts.items()}
        record = ExperienceRecord(
            experience_id=f"EXP-{campaign_id}-{round_result.round_id}-{branch.branch_id}",
            task_signature=signature,
            source_campaign=campaign_id,
            source_round=round_result.round_id,
            source_branch=branch.branch_id,
            parent_id=round_result.parent_id,
            hypothesis=spec.hypothesis,
            strategy_family=spec.strategy_family,
            action_summary=spec.worker_prompt[:280],
            evaluation=result,
            outcome=outcome,
            failure_type=branch.failure_type,
            failure_reason=str(result.get("error_log", "")) or branch.failure_type,
            lesson=lesson,
            candidate_sha256=branch.evaluated_candidate_sha256,
            artifact_refs=refs,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        if store.append(record):
            created.append(record)
    return created

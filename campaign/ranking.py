"""Deterministic Phase 2A candidate ranking; no LLM judge participates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import BranchResult


@dataclass(frozen=True)
class RankingDecision:
    selected_branch_id: str | None
    decision: str
    reason: str


def rank_candidates(branches: Iterable[BranchResult], *, parent_speedup: float) -> RankingDecision:
    valid = [branch for branch in branches if branch.correctness_passed and branch.speedup_factor is not None]
    if not valid:
        return RankingDecision(None, "KEEP_CURRENT_PARENT", "No branch returned a correctness-passing benchmark result.")
    winner = max(valid, key=lambda branch: (branch.speedup_factor or float("-inf"), branch.branch_id))
    if (winner.speedup_factor or float("-inf")) <= parent_speedup:
        return RankingDecision(
            None,
            "KEEP_CURRENT_PARENT",
            f"Best valid branch {winner.branch_id} ({winner.speedup_factor:.6f}x) does not exceed parent ({parent_speedup:.6f}x).",
        )
    return RankingDecision(
        winner.branch_id,
        "PROMOTE",
        f"{winner.branch_id} is correctness-passing and improves {parent_speedup:.6f}x to {winner.speedup_factor:.6f}x.",
    )

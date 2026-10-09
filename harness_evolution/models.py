"""Serializable evidence and proposal records for the narrow V1 evolution path."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class RetrospectiveContext:
    campaign_id: str
    task_signature: Mapping[str, Any]
    round_summary: Mapping[str, Any]
    successful_branches: tuple[Mapping[str, Any], ...] = ()
    failed_branches: tuple[Mapping[str, Any], ...] = ()
    promotion_history: tuple[Mapping[str, Any], ...] = ()
    budget_summary: Mapping[str, Any] = field(default_factory=dict)
    relevant_experiences: tuple[Mapping[str, Any], ...] = ()
    known_harness_events: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HarnessProposal:
    proposal_id: str
    problem: str
    evidence: tuple[str, ...]
    target_type: str
    target_path: str
    current_behavior: str
    proposed_change: str
    expected_effect: str
    risk: str
    scope_class: str
    source_experiences: tuple[str, ...] = ()
    resolution_key: str = ""

    REQUIRED = (
        "proposal_id", "problem", "evidence", "target_type", "target_path",
        "current_behavior", "proposed_change", "expected_effect", "risk", "scope_class",
    )

    def __post_init__(self) -> None:
        if not self.proposal_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("proposal_id must be alphanumeric with - or _")
        if not self.evidence:
            raise ValueError("proposal requires concrete evidence")
        for name in self.REQUIRED:
            if not getattr(self, name):
                raise ValueError(f"HarnessProposal.{name} must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HarnessProposal":
        missing = [name for name in cls.REQUIRED if name not in value]
        if missing:
            raise ValueError(f"proposal missing fields: {', '.join(missing)}")
        data = dict(value)
        raw_evidence = data["evidence"]
        data["evidence"] = (raw_evidence,) if isinstance(raw_evidence, str) else tuple(str(item) for item in raw_evidence)
        data["source_experiences"] = tuple(str(item) for item in data.get("source_experiences", ()))
        data["resolution_key"] = str(data.get("resolution_key", ""))
        return cls(**data)


@dataclass(frozen=True)
class GateDecision:
    decision: str
    reason: str
    scope_passed: bool
    evidence_passed: bool
    regression_passed: bool | None
    harness_version_before: str
    harness_version_after: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

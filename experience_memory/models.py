from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

@dataclass(frozen=True)
class TaskSignature:
    benchmark: str
    task_id: str
    task_level: str
    operator_family: str
    operation_type: str
    input_signature: str
    shape_signature: str
    dtype: str
    hardware: str
    evaluation_contract: str
    def to_dict(self) -> dict[str, str]: return asdict(self)
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSignature": return cls(**{key: str(value[key]) for key in cls.__dataclass_fields__})

@dataclass(frozen=True)
class ExperienceRecord:
    experience_id: str
    task_signature: TaskSignature
    source_campaign: str
    source_round: str
    source_branch: str
    parent_id: str
    hypothesis: str
    strategy_family: str
    action_summary: str
    evaluation: Mapping[str, Any]
    outcome: str
    failure_type: str | None
    failure_reason: str | None
    lesson: str
    candidate_sha256: str
    artifact_refs: Mapping[str, str]
    created_at: str
    def dedupe_key(self) -> str: return "|".join((self.source_campaign, self.source_round, self.source_branch, self.candidate_sha256))
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["task_signature"] = self.task_signature.to_dict(); return data
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperienceRecord":
        data = dict(value); data["task_signature"] = TaskSignature.from_dict(data["task_signature"]); return cls(**data)

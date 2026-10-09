from __future__ import annotations
from dataclasses import dataclass
from .models import ExperienceRecord, TaskSignature
from .store import ExperienceStore

@dataclass(frozen=True)
class MemoryHit:
    record: ExperienceRecord
    score: int
    summary: str

class ExperienceRetriever:
    def __init__(self, store: ExperienceStore) -> None: self.store = store
    @staticmethod
    def score(source: TaskSignature, target: TaskSignature) -> int:
        if source.benchmark != target.benchmark or source.hardware != target.hardware: return -1
        # A bounded context is useful only when it is operator-relevant.  A
        # matching dtype/level alone must not make a GEMM lesson appear in a
        # normalization or elementwise campaign.
        if (source.operator_family != target.operator_family
                and source.operation_type != target.operation_type):
            return -1
        score = 0
        score += 4 if source.operator_family == target.operator_family else 0
        score += 3 if source.operation_type == target.operation_type else 0
        score += 1 if source.dtype == target.dtype else 0
        score += 1 if source.task_level == target.task_level else 0
        return score
    def query(self, target: TaskSignature, *, success_limit: int = 2, failure_limit: int = 1) -> list[MemoryHit]:
        hits = [MemoryHit(item, self.score(item.task_signature, target), f"{item.experience_id} | {item.strategy_family} | {item.outcome} | {item.lesson}") for item in self.store.list()]
        hits = [item for item in hits if item.score >= 0]
        hits.sort(key=lambda item: (-item.score, item.record.experience_id))
        success = [item for item in hits if item.record.outcome in {"PROMOTED", "VALID_NOT_PROMOTED"}][:success_limit]
        failure = [item for item in hits if item.record.outcome not in {"PROMOTED", "VALID_NOT_PROMOTED", "INFRA_FAILURE"}][:failure_limit]
        return success + failure

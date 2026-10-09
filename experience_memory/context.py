from __future__ import annotations
from .retrieval import MemoryHit

class ExperienceContextBuilder:
    def __init__(self, *, max_full_records: int = 2, max_characters: int = 3500) -> None:
        self.max_full_records, self.max_characters = max_full_records, max_characters
    def build(self, hits: list[MemoryHit]) -> tuple[str, list[str]]:
        summaries = ["Cross-task experience memory (advisory priors; evaluate the current task independently):"]
        summaries.extend(f"- score={hit.score}: {hit.summary}" for hit in hits)
        selected: list[MemoryHit] = []
        for outcome_set in (("PROMOTED", "VALID_NOT_PROMOTED"), ("CORRECTNESS_FAILURE", "COMPILE_FAILURE", "RUNTIME_FAILURE", "TIMEOUT")):
            selected.extend([hit for hit in hits if hit.record.outcome in outcome_set][:1])
        selected = selected[:self.max_full_records]
        if selected: summaries.append("Selected full records:")
        for hit in selected:
            record = hit.record
            summaries.append(f"- {record.experience_id}: hypothesis={record.hypothesis}; action={record.action_summary}; lesson={record.lesson}; evidence={record.evaluation.get('status')} {record.evaluation.get('speedup_factor', 'invalid')}x.")
        text = "\n".join(summaries)[:self.max_characters]
        return text, [hit.record.experience_id for hit in selected]

from __future__ import annotations
import json
from pathlib import Path
from collections.abc import Callable

from .models import ExperienceRecord

class ExperienceStore:
    def __init__(self, path: Path) -> None: self.path = Path(path)
    def list(self) -> list[ExperienceRecord]:
        if not self.path.is_file(): return []
        return [ExperienceRecord.from_dict(json.loads(line)) for line in self.path.read_text().splitlines() if line.strip()]
    def append(self, record: ExperienceRecord) -> bool:
        existing = {item.dedupe_key() for item in self.list()}
        if record.dedupe_key() in existing: return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return True
    def get(self, experience_id: str) -> ExperienceRecord | None:
        return next((item for item in self.list() if item.experience_id == experience_id), None)

    def query(self, predicate: Callable[[ExperienceRecord], bool] | None = None) -> list[ExperienceRecord]:
        """Return records matching a caller-supplied metadata predicate.

        Ranking deliberately stays in :mod:`experience_memory.retrieval`; this
        small method makes the persistent-store surface complete without
        coupling it to a particular similarity policy.
        """
        records = self.list()
        return records if predicate is None else [record for record in records if predicate(record)]

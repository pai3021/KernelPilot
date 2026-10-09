"""Serializable Phase 2A planning, branch, and round records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class RoundState(str, Enum):
    PLANNING = "PLANNING"
    BRANCHES_READY = "BRANCHES_READY"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    RANKED = "RANKED"
    PROMOTED = "PROMOTED"
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    FAILED = "FAILED"


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    parent_id: str
    hypothesis: str
    strategy_family: str
    reason: str
    expected_signal: str
    worker_prompt: str
    build_language: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if field_name == "build_language":
                continue
            if not str(value).strip():
                raise ValueError(f"BranchSpec.{field_name} must not be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BranchSpec":
        required = ("branch_id", "parent_id", "hypothesis", "strategy_family", "reason", "expected_signal", "worker_prompt")
        missing = [name for name in required if not isinstance(value.get(name), str) or not value[name].strip()]
        if missing:
            raise ValueError(f"BranchSpec missing non-empty fields: {', '.join(missing)}")
        build_language = value.get("build_language")
        if build_language is not None and (not isinstance(build_language, str) or not build_language.strip()):
            raise ValueError("BranchSpec.build_language must be a non-empty string when present")
        return cls(**{name: str(value[name]).strip() for name in required},
                   build_language=None if build_language is None else build_language.strip())

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class BranchResult:
    branch_id: str
    runtime: str
    session_id: str
    status: str
    candidate_path: Path
    diff: str
    benchmark_result: Mapping[str, Any] | None
    benchmark_calls: int
    final_message: str
    failure_type: str | None
    evaluated_candidate_sha256: str | None = None
    evaluated_candidate_path: Path | None = None
    evaluated_at: str | None = None
    artifacts: Mapping[str, Path] = field(default_factory=dict)

    @property
    def correctness_passed(self) -> bool:
        return bool(self.benchmark_result and self.benchmark_result.get("status") == "PASSED")

    @property
    def speedup_factor(self) -> float | None:
        if not self.benchmark_result:
            return None
        value = self.benchmark_result.get("speedup_factor")
        return float(value) if isinstance(value, (int, float)) else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate_path"] = str(self.candidate_path)
        data["evaluated_candidate_path"] = str(self.evaluated_candidate_path) if self.evaluated_candidate_path else None
        data["artifacts"] = {name: str(path) for name, path in self.artifacts.items()}
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BranchResult":
        """Restore a persisted branch result without re-running its benchmark."""
        return cls(
            branch_id=str(value["branch_id"]),
            runtime=str(value["runtime"]),
            session_id=str(value["session_id"]),
            status=str(value["status"]),
            candidate_path=Path(str(value["candidate_path"])),
            diff=str(value["diff"]),
            benchmark_result=value.get("benchmark_result"),
            benchmark_calls=int(value["benchmark_calls"]),
            final_message=str(value["final_message"]),
            failure_type=None if value.get("failure_type") is None else str(value["failure_type"]),
            evaluated_candidate_sha256=None if value.get("evaluated_candidate_sha256") is None else str(value["evaluated_candidate_sha256"]),
            evaluated_candidate_path=None if value.get("evaluated_candidate_path") is None else Path(str(value["evaluated_candidate_path"])),
            evaluated_at=None if value.get("evaluated_at") is None else str(value["evaluated_at"]),
            artifacts={str(name): Path(str(path)) for name, path in (value.get("artifacts") or {}).items()},
        )


@dataclass(frozen=True)
class RoundResult:
    round_id: str
    state: RoundState
    parent_id: str
    parent_speedup: float
    branches: tuple[BranchResult, ...]
    selected_branch_id: str | None
    promotion_decision: str
    artifacts: Mapping[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "state": self.state.value,
            "parent_id": self.parent_id,
            "parent_speedup": self.parent_speedup,
            "branches": [branch.to_dict() for branch in self.branches],
            "selected_branch_id": self.selected_branch_id,
            "promotion_decision": self.promotion_decision,
            "artifacts": {name: str(path) for name, path in self.artifacts.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoundResult":
        """Restore a completed round from its immutable campaign artifact."""
        return cls(
            round_id=str(value["round_id"]),
            state=RoundState(str(value["state"])),
            parent_id=str(value["parent_id"]),
            parent_speedup=float(value["parent_speedup"]),
            branches=tuple(BranchResult.from_dict(item) for item in value["branches"]),
            selected_branch_id=None if value.get("selected_branch_id") is None else str(value["selected_branch_id"]),
            promotion_decision=str(value["promotion_decision"]),
            artifacts={str(name): Path(str(path)) for name, path in (value.get("artifacts") or {}).items()},
        )

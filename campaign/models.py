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
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str                      # 分支id
    parent_id: str                      # 从哪个已有版本出发
    hypothesis: str                     # 本次优化的假设
    strategy_family: str                # 优化策略类型
    reason: str                         # 为什么选择这个策略
    expected_signal: str                # 预期的性能信号
    worker_prompt: str                  # 交给worker 的提示词
    build_language: str | None = None   # 候选准备使用的语言

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
    branch_id: str                   # 分支id
    runtime: str                     # 哪个 coding agent
    session_id: str                  # 会话id
    status: str                      # 分支评测状态：PASSED / FAILED；Agent 运行失败原因另记在 failure_type
    candidate_path: Path             # 候选代码位置，但优先指向快照
    diff: str
    benchmark_result: Mapping[str, Any] | None # 实际评测结果，包括正确性状态、耗时、加速比等
    benchmark_calls: int             # 评测调用次数
    final_message: str               # Agent 的最终回复
    failure_type: str | None         # 失败类型：Agent 超时、没有评测报告、正确性失败
    # 实际评测代码的哈希、快照位置和评测时间
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
    round_id: str                    # 轮次编号
    state: RoundState                # 这一轮最后处于什么状态
    parent_id: str                   # 本轮从哪个代码版本出发
    parent_speedup: float            # 父候选加速比
    branches: tuple[BranchResult, ...] # 这一轮各分支的 BranchResult
    selected_branch_id: str | None   # 选中了哪个分支；没有晋升候选时可以为空
    promotion_decision: str          # 晋升候选、保留当前父版本等决策
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

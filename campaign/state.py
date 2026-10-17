"""Small, serializable state for a single two-round campaign only."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ParentSnapshot:
    parent_id: str          # 父候选的唯一标识符
    source_branch: str      # 父候选的源分支名称
    candidate_path: Path    # 父候选的代码路径
    benchmark_result: Mapping[str, Any]  # 父候选的评测结果
    promoted_round: str     # 父候选晋升的轮次编号
    diff: str = ""          # 该候选在其工作区中记录的代码差异

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self); value["candidate_path"] = str(self.candidate_path); return value

# RoundRecord：把一轮经过放进历史。
# 它记录：
# - 这一轮从哪个父版本出发。
# - 提出了哪些分支计划。
# - 各分支得到什么结果。
# - 选中了谁、做了什么决策。
# - 最终保留哪个版本作为后续起点。
@dataclass(frozen=True)
class RoundRecord:
    round_id: str
    parent_id: str
    parent_result: Mapping[str, Any]
    branch_specs: tuple[Mapping[str, Any], ...]
    branch_results: tuple[Mapping[str, Any], ...]
    winner: str | None
    promotion_decision: str
    selected_parent: str

    def to_dict(self) -> dict[str, Any]: return asdict(self)

# 整个 campaign 的当前状态
@dataclass
class CampaignState:
    campaign_id: str
    task_id: str
    round_index: int
    initial_parent: ParentSnapshot
    current_parent: ParentSnapshot
    best_result: Mapping[str, Any]
    round_history: list[RoundRecord] = field(default_factory=list)
    benchmark_budget_total: int = 4
    benchmark_budget_used: int = 0
    status: str = "RUNNING"
    termination_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"campaign_id": self.campaign_id, "task_id": self.task_id, "round_index": self.round_index,
                "initial_parent": self.initial_parent.to_dict(), "current_parent": self.current_parent.to_dict(),
                "best_result": dict(self.best_result), "round_history": [item.to_dict() for item in self.round_history],
                "benchmark_budget_total": self.benchmark_budget_total, "benchmark_budget_used": self.benchmark_budget_used,
                "status": self.status, "termination_reason": self.termination_reason}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "CampaignState":
        data = json.loads(path.read_text())
        def snapshot(value: Mapping[str, Any]) -> ParentSnapshot:
            return ParentSnapshot(str(value["parent_id"]), str(value["source_branch"]), Path(str(value["candidate_path"])), dict(value["benchmark_result"]), str(value["promoted_round"]), str(value.get("diff", "")))
        history = [RoundRecord(str(item["round_id"]), str(item["parent_id"]), dict(item["parent_result"]), tuple(item["branch_specs"]), tuple(item["branch_results"]), item.get("winner"), str(item["promotion_decision"]), str(item["selected_parent"])) for item in data.get("round_history", [])]
        return cls(str(data["campaign_id"]), str(data["task_id"]), int(data["round_index"]), snapshot(data["initial_parent"]), snapshot(data["current_parent"]), dict(data["best_result"]), history, int(data["benchmark_budget_total"]), int(data["benchmark_budget_used"]), str(data["status"]), data.get("termination_reason"))

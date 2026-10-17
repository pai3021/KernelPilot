from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

# 任务特征签名
@dataclass(frozen=True)
class TaskSignature:
    benchmark: str              # 评测基准
    task_id: str                # 任务 ID
    task_level: str             # 任务层级
    operator_family: str        # 算子类别
    operation_type: str         # 算子类型
    input_signature: str        # 输入描述
    shape_signature: str        # 形状描述
    dtype: str                  # 数据类型
    hardware: str               # 硬件环境
    evaluation_contract: str    # 评测约定
    def to_dict(self) -> dict[str, str]: return asdict(self)
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSignature": return cls(**{key: str(value[key]) for key in cls.__dataclass_fields__})

# 结构化经验记录
@dataclass(frozen=True)
class ExperienceRecord:
    # 1. 身份与来源：这条经验从哪里来
    experience_id: str                  # 经验编号
    task_signature: TaskSignature       # 任务条件
    source_campaign: str                # 所属 campaign
    source_round: str                   # 所属 round
    source_branch: str                  # 所属 branch
    parent_id: str                      # 父候选 id
    # 2. 优化方案：当时准备做什么
    hypothesis: str                     # 核心假设
    strategy_family: str                # 优化策略类别
    action_summary: str                 # 尝试内容摘要；当前提取器取 Planner 的 worker_prompt 前 280 字符
    # 3. 实测结果：这次尝试发生了什么
    evaluation: Mapping[str, Any]       # 评测结果
    outcome: str                        # 结果分类
    failure_type: str | None            # 失败类型
    failure_reason: str | None          # 失败原因
    # 4. 经验与证据：得到了什么观察，依据是什么
    lesson: str                         # 经验总结
    candidate_sha256: str               # 候选代码的 SHA-256
    artifact_refs: Mapping[str, str]    # 相关产物的引用
    created_at: str
    # 判断是否重复保存同一次尝试
    def dedupe_key(self) -> str: return "|".join((self.source_campaign, self.source_round, self.source_branch, self.candidate_sha256))
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["task_signature"] = self.task_signature.to_dict(); return data
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperienceRecord":
        data = dict(value); data["task_signature"] = TaskSignature.from_dict(data["task_signature"]); return cls(**data)

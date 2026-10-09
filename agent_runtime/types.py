"""Small, serializable V1 contracts shared by all agent runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimeCapabilities:
    # 能力描述而非行为实现：上层可据此判断 resume、hooks、sandbox 等是否可用。
    supports_resume: bool = False
    supports_structured_events: bool = False
    supports_native_project_instructions: bool = False
    supports_native_skills: bool = False
    supports_hooks: bool = False
    supports_native_sandbox: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class TaskContract:
    """Runtime-neutral task facts; renderers own their native instruction file."""

    # 一份任务的长期规则：做什么、允许改什么、怎样评测、何时停止。
    # 它不携带任何 claude/codex 命令行参数。

    objective: str
    editable_files: tuple[str, ...]
    benchmark_command: str
    correctness_contract: str
    forbidden_changes: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    rendered_instructions: str = ""

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("TaskContract.objective must not be empty")
        if not self.editable_files:
            raise ValueError("TaskContract.editable_files must not be empty")
        if not self.benchmark_command.strip():
            raise ValueError("TaskContract.benchmark_command must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentSession:
    # 统一会话句柄。session_id 在 Claude 中是 UUID，在 Codex 中通常是 thread_id；
    # 上层只负责保存和传回，不解释其供应商含义。
    runtime: str
    session_id: str
    workspace: Path
    status: str
    started_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["workspace"] = str(self.workspace)
        return data


@dataclass(frozen=True)
class AgentRunRequest:
    # “这一次调用”的参数，与描述长期任务规则的 TaskContract 分开。
    workspace: Path
    prompt: str
    timeout: int
    environment: Mapping[str, str] = field(default_factory=dict)
    model: str | None = None
    sandbox_policy: str | None = None
    artifact_label: str = "run"

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("AgentRunRequest.prompt must not be empty")
        if self.timeout <= 0:
            raise ValueError("AgentRunRequest.timeout must be positive")
        if not self.artifact_label.replace("_", "").replace("-", "").isalnum():
            raise ValueError("AgentRunRequest.artifact_label must be alphanumeric")


@dataclass(frozen=True)
class AgentEvent:
    # 公共 kind 供上层判断事件类别，data 保留原始 JSON 方便诊断和兼容新字段。
    kind: str
    timestamp: str | None = None
    message: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentRunResult:
    # Runtime 边界的统一输出；上层无需再解析 Claude/Codex 的原始 JSONL。
    session_id: str
    status: str
    exit_code: int
    final_message: str
    events: tuple[AgentEvent, ...]
    timed_out: bool
    duration: float
    stderr_tail: str
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["events"] = [event.to_dict() for event in self.events]
        data["artifacts"] = {key: str(path) for key, path in self.artifacts.items()}
        return data

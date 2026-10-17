"""Small, serializable V1 contracts shared by all agent runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


# 阅读顺序：TaskContract（公共规则）→ AgentRunRequest（本次指令）
# → AgentRunResult（运行结果）；AgentSession 用来继续已有会话。
# 这些 dataclass 负责携带数据，执行 CLI 的代码在 codex.py / claude.py。
@dataclass(frozen=True)
class RuntimeCapabilities:
    # 这里描述本项目适配器已经接入的能力，不代表对应产品的全部能力。
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

    objective: str                      # 任务目标
    editable_files: tuple[str, ...]     # 允许修改的文件列表
    benchmark_command: str              # 评测命令行
    correctness_contract: str           # 正确性门槛
    forbidden_changes: tuple[str, ...]  # 禁止改动哪些东西
    stop_conditions: tuple[str, ...]    # 停止条件
    # 同一任务的多个分支可共享上述规则，具体方案另放进运行请求的 prompt。
    # editable_files 等字段会写入任务说明；这个数据类本身不实施访问限制。
    # 完整模板正文供 Claude 渲染器优先使用；Codex 渲染器按结构化字段生成说明。
    rendered_instructions: str = ""

    def __post_init__(self) -> None:
        # dataclass 自动生成的构造函数会调用这里，校验任务的必要信息。
        # 正确性结果由 benchmark 决定，不由这个数据类判定。
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
    # 已有会话的标识，不等于优化分支；下一轮也可以选择启动新会话。
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
    # Planner、Worker、复盘 Agent 都可以使用它，角色由任务和 prompt 指定。
    # timeout 单位为秒；environment 会覆盖或补充父进程的同名环境变量。
    # “这一次调用”的参数，与描述长期任务规则的 TaskContract 分开。
    workspace: Path
    prompt: str
    timeout: int
    environment: Mapping[str, str] = field(default_factory=dict)
    model: str | None = None
    # 字段是否生效取决于适配器：当前 Codex 固定 workspace-write，未读此字段；
    # Claude 的 build_command 当前未使用 model / sandbox_policy。
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
    # JSONL 是每行一个 JSON；适配器将两种 CLI 的记录映射到公共事件类型。
    # events 是本次执行后解析出的记录，当前实现没有提供实时事件订阅接口。
    # 公共 kind 供上层判断事件类别，data 保留原始 JSON 方便诊断和兼容新字段。
    kind: str                       # 统一的事件类型，工具调用、工具结果等等
    timestamp: str | None = None
    message: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentRunResult:
    # status=completed 表示 CLI 正常退出，不代表候选正确或性能提升。
    # final_message 是 Agent 的文字回复，评测结论仍需读取 benchmark 报告。
    # Runtime 边界的统一输出；上层无需再解析 Claude/Codex 的原始 JSONL。
    session_id: str                 # 会话标识
    status: str                     # 当前运行状态：completed / failed / timed_out
    exit_code: int                  # CLI 退出码，非零表示失败
    final_message: str              # Agent 最终回复的文本
    events: tuple[AgentEvent, ...]  # 运行期间的事件记录，按时间顺序排列
    timed_out: bool                 # 是否超时
    duration: float                 # 本次运行耗时（秒），使用 monotonic 时钟计算
    stderr_tail: str                # 标准错误输出的末尾部分
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["events"] = [event.to_dict() for event in self.events]
        # Path 转成字符串后，控制器才能把产物位置一起写入 JSON 记录。
        data["artifacts"] = {key: str(path) for key, path in self.artifacts.items()}
        return data

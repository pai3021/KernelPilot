"""Abstract interface for a coding-agent runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .types import AgentRunRequest, AgentRunResult, AgentSession, RuntimeCapabilities, TaskContract


class AgentRuntime(ABC):
    """How an agent runs, deliberately separate from benchmark execution."""

    # 这是 Harness 向上层暴露的统一“遥控器”。Campaign 等编排代码
    # 只调用下面这些生命周期方法，不直接拼 Claude/Codex 的命令行参数。
    # 各 CLI 的客观差异不会消失，而是被限制在各自的 Runtime Adapter 内。

    name: str

    @property
    @abstractmethod
    def capabilities(self) -> RuntimeCapabilities:
        """Report capabilities implemented by this runtime today."""

        # 不强求两个后端能力完全相同；只如实声明当前实现支持什么。

    @abstractmethod
    def prepare_workspace(self, workspace: Path, task: TaskContract) -> tuple[Path, ...]:
        """Materialize runtime-native project files into a common child workspace."""

        # 将统一 TaskContract 翻译成 CLAUDE.md 或 AGENTS.md/CODEX_TASK.md。

    @abstractmethod
    def run(self, request: AgentRunRequest) -> AgentRunResult:
        """Start a new agent session."""

        # 启动新会话，并把原生 CLI 输出转换成统一 AgentRunResult。

    @abstractmethod
    def resume(self, session: AgentSession, request: AgentRunRequest) -> AgentRunResult:
        """Continue one session using the runtime's native protocol."""

        # “继续会话”语义统一，具体 --resume/exec resume 命令由 Adapter 翻译。

    @abstractmethod
    def terminate(self, session: AgentSession) -> None:
        """Terminate an active session if one is known to this runtime."""

        # 超时时清理 Runtime 创建的进程及其子进程。

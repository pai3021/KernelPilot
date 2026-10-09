"""Runtime-neutral coding-agent execution boundary for KernelPilot."""

from .base import AgentRuntime
from .registry import RuntimeRegistry
from .types import (
    AgentEvent,
    AgentRunRequest,
    AgentRunResult,
    AgentSession,
    RuntimeCapabilities,
    TaskContract,
)

# Import concrete implementations once so `RuntimeRegistry.create("claude")`
# and `RuntimeRegistry.create("codex")` work for every public-package
# consumer, not only callers that happened to import an adapter module first.
# 中文理解：这里利用“导入模块会执行模块末尾 register()”的副作用完成注册，
# 保证调用者只 import agent_runtime 也能按名称创建两个具体 Adapter。
from . import claude as _claude  # noqa: F401,E402
from . import codex as _codex  # noqa: F401,E402

__all__ = [
    "AgentEvent",
    "AgentRunRequest",
    "AgentRunResult",
    "AgentRuntime",
    "AgentSession",
    "RuntimeCapabilities",
    "RuntimeRegistry",
    "TaskContract",
]

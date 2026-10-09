"""Runtime construction registry for verified runtime implementations."""

from __future__ import annotations

from collections.abc import Callable

from .base import AgentRuntime


class RuntimeRegistry:
    # 名称 -> Runtime 工厂的注册表。这样上层只按配置 create("codex")，
    # 不需要到处编写 if runtime == "codex" / elif runtime == "claude"。
    _factories: dict[str, Callable[..., AgentRuntime]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., AgentRuntime]) -> None:
        # 具体 Adapter 在模块末尾注册自己，例如 codex -> CodexRuntime。
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("runtime name must not be empty")
        cls._factories[normalized] = factory

    @classmethod
    def create(cls, name: str, **kwargs: object) -> AgentRuntime:
        # **kwargs 原样交给具体工厂，Registry 本身不理解各 Runtime 的初始化细节。
        normalized = name.strip().lower()
        try:
            return cls._factories[normalized](**kwargs)
        except KeyError as exc:
            known = ", ".join(sorted(cls._factories)) or "(none)"
            raise ValueError(f"unknown runtime {name!r}; registered runtimes: {known}") from exc

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._factories))

"""Deterministic scope, evidence, and regression gates for one proposal."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .models import GateDecision, HarnessProposal


# 当前版本可接受的模板/指南目标前缀；不表示允许修改整个 templates 目录。
MUTABLE_PREFIXES = ("templates/skills/", "templates/agent/", "templates/task.md", "templates/retrospective.md", "templates/closed-loop-scope.md")
# 核心执行、评测和编排代码不接受这条自演进流程的修改提案。
FROZEN_PREFIXES = ("agent_runtime/", "benchmark_backend/", "campaign/", "experience_memory/", "master/", "scripts/", "spawn.py", "templates/benchmark/")
# 对路径和建议文字做关键词过滤；这里采用字符串匹配。
SECURITY_TERMS = ("ssh", "credential", "token", "secret", "sandbox", "approval", "security")
# 已有程序化控制的已知问题，避免再次用提示词建议重复解决。
RESOLVED_CONTROLS = {"harness_level_benchmark_budget": "benchmark_backend/budget.py already programmatically enforces one formal evaluation per branch"}


def scope_reason(proposal: HarnessProposal) -> str | None:
    # 返回拒绝原因；None 表示通过本项检查。路径只统一分隔符并去掉开头斜杠。
    path = proposal.target_path.replace("\\", "/").lstrip("/")
    if any(path.startswith(prefix) for prefix in FROZEN_PREFIXES):
        return "SCOPE_REJECT: frozen or core-Python target"
    if any(term in f"{path} {proposal.proposed_change}".lower() for term in SECURITY_TERMS):
        return "SCOPE_REJECT: security boundary"
    if not any(path.startswith(prefix) for prefix in MUTABLE_PREFIXES):
        return "SCOPE_REJECT: target is outside the V1 mutable allowlist"
    return None


def evidence_reason(proposal: HarnessProposal) -> str | None:
    # 基础文本检查：证据非空、描述有一定长度；不读取报告核实事实或因果关系。
    if not proposal.evidence or any(not item.strip() for item in proposal.evidence):
        return "EVIDENCE_REJECT: concrete campaign evidence is required"
    if len(proposal.problem.strip()) < 12 or len(proposal.proposed_change.strip()) < 12:
        return "EVIDENCE_REJECT: problem/change is not specific enough"
    return None


class ProposalGate:
    def __init__(self, *, project_root: Path, version_before: str = "H0") -> None:
        self.project_root = Path(project_root)
        self.version_before = version_before

    def evaluate(self, proposal: HarnessProposal) -> GateDecision:
        # 1. 先检查范围，失败就返回；此时 evidence_passed=False 不代表已经检查过证据。
        reason = scope_reason(proposal)
        if reason:
            return GateDecision("REJECT", reason, False, False, None, self.version_before, self.version_before)
        # 2. 检查证据条目和问题/修改文字是否满足基本要求。
        reason = evidence_reason(proposal)
        if reason:
            return GateDecision("REJECT", reason, True, False, None, self.version_before, self.version_before)
        # 3. 匹配已解决问题的标识，例如正式评测预算已有代码约束。
        if proposal.resolution_key in RESOLVED_CONTROLS:
            return GateDecision("REJECT_AS_ALREADY_RESOLVED", RESOLVED_CONTROLS[proposal.resolution_key], True, True, None, self.version_before, self.version_before)
        # 通过静态规则，只进入待回归状态；此方法不应用提案，也不增加 Harness 版本。
        return GateDecision("PENDING_REGRESSION", "scope and evidence accepted", True, True, None, self.version_before, self.version_before)

    def run_regression(self) -> tuple[bool, str]:
        # 使用控制进程的 Python，在原项目目录运行单元测试；SSH 控制流程应从 WSL 启动。
        # coordinator 另存的候选模板没有在此替换进项目，不能据此宣称已验证新模板效果。
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=self.project_root, text=True, capture_output=True, check=False,
        )
        # 返回是否通过，以及最后 12000 个字符的测试输出，供上层保存检查记录。
        return completed.returncode == 0, (completed.stdout + completed.stderr)[-12000:]

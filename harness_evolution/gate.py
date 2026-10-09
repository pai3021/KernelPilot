"""Deterministic scope, evidence, and regression gates for one proposal."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .models import GateDecision, HarnessProposal


MUTABLE_PREFIXES = ("templates/skills/", "templates/agent/", "templates/task.md", "templates/retrospective.md", "templates/closed-loop-scope.md")
FROZEN_PREFIXES = ("agent_runtime/", "benchmark_backend/", "campaign/", "experience_memory/", "master/", "scripts/", "spawn.py", "templates/benchmark/")
SECURITY_TERMS = ("ssh", "credential", "token", "secret", "sandbox", "approval", "security")
RESOLVED_CONTROLS = {"harness_level_benchmark_budget": "benchmark_backend/budget.py already programmatically enforces one formal evaluation per branch"}


def scope_reason(proposal: HarnessProposal) -> str | None:
    path = proposal.target_path.replace("\\", "/").lstrip("/")
    if any(path.startswith(prefix) for prefix in FROZEN_PREFIXES):
        return "SCOPE_REJECT: frozen or core-Python target"
    if any(term in f"{path} {proposal.proposed_change}".lower() for term in SECURITY_TERMS):
        return "SCOPE_REJECT: security boundary"
    if not any(path.startswith(prefix) for prefix in MUTABLE_PREFIXES):
        return "SCOPE_REJECT: target is outside the V1 mutable allowlist"
    return None


def evidence_reason(proposal: HarnessProposal) -> str | None:
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
        reason = scope_reason(proposal)
        if reason:
            return GateDecision("REJECT", reason, False, False, None, self.version_before, self.version_before)
        reason = evidence_reason(proposal)
        if reason:
            return GateDecision("REJECT", reason, True, False, None, self.version_before, self.version_before)
        if proposal.resolution_key in RESOLVED_CONTROLS:
            return GateDecision("REJECT_AS_ALREADY_RESOLVED", RESOLVED_CONTROLS[proposal.resolution_key], True, True, None, self.version_before, self.version_before)
        return GateDecision("PENDING_REGRESSION", "scope and evidence accepted", True, True, None, self.version_before, self.version_before)

    def run_regression(self) -> tuple[bool, str]:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=self.project_root, text=True, capture_output=True, check=False,
        )
        return completed.returncode == 0, (completed.stdout + completed.stderr)[-12000:]

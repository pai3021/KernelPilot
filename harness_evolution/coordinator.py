"""One optional post-campaign retrospective, proposal gate, and ledger entry."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import SelfEvolutionConfig
from .gate import ProposalGate
from .ledger import append_ledger, next_version, read_version, write_version
from .models import GateDecision, HarnessProposal, RetrospectiveContext
from .retrospective import Retrospective


class EvolutionCoordinator:
    """Keep self-evolution an explicit post-campaign feature, never a prerequisite."""

    def __init__(self, *, project_root: Path, config: SelfEvolutionConfig) -> None:
        self.project_root = Path(project_root)
        self.config = config
        # 复盘产物、当前 Harness 版本和跨次处理历史分别保存；构造时尚未调用 Agent。
        self.root = self.project_root / "artifacts" / "evolution"
        self.version_path = self.root / "harness-version.json"
        self.ledger_path = self.root / "harness-ledger.jsonl"

    def after_campaign(self, context: RetrospectiveContext) -> dict[str, Any]:
        # campaign 结束后的入口：功能关闭则直接返回，不发起复盘。
        if not self.config.enabled or not self.config.after_campaign:
            return {"status": "DISABLED", "mutated": False, "retrospective_invoked": False}
        version_before = read_version(self.version_path)
        if not self.version_path.is_file():
            write_version(self.version_path, version_before)
        proposal_root = self.root / f"{context.campaign_id}-retrospective"
        # 每个 campaign 使用独立目录，已有证据不在这里覆盖重跑。
        if proposal_root.exists():
            raise FileExistsError(f"evolution evidence already exists: {proposal_root}")
        proposal_root.mkdir(parents=True)
        (proposal_root / "context.json").write_text(json.dumps(context.to_dict(), indent=2) + "\n")
        try:
            # 1. 保存输入后调用复盘 Agent，得到一条可选提案；Agent 不负责应用修改。
            proposal = Retrospective(project_root=self.project_root, runtime_name=self.config.runtime).run(
                workspace=proposal_root / "retrospective", context=context,
            )
        except (RuntimeError, ValueError) as exc:
            # Reflection is optional post-campaign work.  A malformed or failed
            # retrospective must retain evidence and not erase a completed
            # deterministic campaign.
            # 这里捕获到的复盘失败单独记账，已完成的 campaign 结果仍保留。
            (proposal_root / "retrospective-error.txt").write_text(str(exc) + "\n")
            decision = GateDecision("RETROSPECTIVE_FAILURE", str(exc), False, False, None, version_before, version_before)
            self._write_decision(proposal_root, context, None, decision)
            return {"status": decision.decision, "mutated": False, "retrospective_invoked": True, "artifact_root": proposal_root}
        if proposal is None:
            # 没有足够依据时不强行改进，也不增加版本；记录本次没有提案。
            decision = GateDecision("NO_PROPOSAL", "retrospective found no sufficiently evidenced mutable gap", True, True, None, version_before, version_before)
            self._write_decision(proposal_root, context, None, decision)
            return {"status": decision.decision, "mutated": False, "retrospective_invoked": True, "artifact_root": proposal_root}
        # 2. 有提案才进入确定性的检查与应用流程。
        return self.gate_proposal(context=context, proposal=proposal, artifact_root=proposal_root, retrospective_invoked=True)

    def gate_proposal(self, *, context: RetrospectiveContext, proposal: HarnessProposal, artifact_root: Path, retrospective_invoked: bool = False) -> dict[str, Any]:
        """Gate a retained structured proposal without re-running an agent.

        Used for replay recovery when a historical runtime response was valid
        but an adapter bug prevented its first deterministic gate evaluation.
        """
        # 也可直接传入保留的历史提案重做检查，无需重新调用复盘 Agent。
        proposal_root = Path(artifact_root)
        proposal_root.mkdir(parents=True, exist_ok=True)
        version_before = read_version(self.version_path)
        if not self.version_path.is_file():
            write_version(self.version_path, version_before)
        (proposal_root / "context.json").write_text(json.dumps(context.to_dict(), indent=2) + "\n")
        (proposal_root / "proposal.json").write_text(json.dumps(proposal.to_dict(), indent=2) + "\n")
        # 先检查修改范围、基础证据要求，以及问题是否已有控制措施。
        gate = ProposalGate(project_root=self.project_root, version_before=version_before)
        decision = gate.evaluate(proposal)
        regression_log = ""
        if decision.decision == "PENDING_REGRESSION":
            # 保存追加建议后的候选文件；原目标暂不修改。
            self._materialize_variant(proposal_root, proposal)
            # 当前回归仍在原项目运行，没有加载上面的候选文件，不验证新指南的实际收益。
            passed, regression_log = gate.run_regression()
            if passed:
                # 检查通过后才追加到真实目标，更新 H0/H1 等逻辑版本；这里不创建 Git 提交。
                self._apply(proposal)
                version_after = next_version(version_before)
                write_version(self.version_path, version_after)
                decision = GateDecision("ACCEPT", "scope, evidence, and regression gates passed", True, True, True, version_before, version_after)
            else:
                decision = GateDecision("REJECT", "REGRESSION_REJECT", True, True, False, version_before, version_before)
        (proposal_root / "regression-output.txt").write_text(regression_log)
        # 无论接受还是拒绝，都保留决策及版本变化，返回 mutated 表示是否已应用。
        self._write_decision(proposal_root, context, proposal, decision)
        return {"status": decision.decision, "mutated": decision.decision == "ACCEPT", "retrospective_invoked": retrospective_invoked, "artifact_root": proposal_root}

    def _materialize_variant(self, root: Path, proposal: HarnessProposal) -> None:
        # 为审阅保留“原文件 + 新指导文字”的副本；不是工作区切换或补丁执行。
        source = self.project_root / proposal.target_path
        if not source.is_file():
            raise FileNotFoundError(f"mutable proposal target is absent: {source}")
        candidate = root / "candidate" / proposal.target_path
        candidate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, candidate)
        candidate.write_text(candidate.read_text() + "\n\n" + proposal.proposed_change.strip() + "\n")

    def _apply(self, proposal: HarnessProposal) -> None:
        """The V1 accepted mutation is intentionally append-only guidance text."""
        # proposed_change 按普通文字追加，不解析成 diff，也不执行其中的命令。
        target = self.project_root / proposal.target_path
        with target.open("a", encoding="utf-8") as handle:
            handle.write("\n\n" + proposal.proposed_change.strip() + "\n")

    def _write_decision(self, root: Path, context: RetrospectiveContext, proposal: HarnessProposal | None, decision: GateDecision) -> None:
        # 本次目录保存完整 GateDecision；全局 JSONL 账本追加一条简要处理历史。
        (root / "gate-result.json").write_text(json.dumps(decision.to_dict(), indent=2) + "\n")
        append_ledger(self.ledger_path, {
            "proposal_id": None if proposal is None else proposal.proposal_id,
            "campaign_id": context.campaign_id,
            "target": None if proposal is None else proposal.target_path,
            "decision": decision.decision,
            "reason": decision.reason,
            "regression_status": decision.regression_passed,
            "harness_version_before": decision.harness_version_before,
            "harness_version_after": decision.harness_version_after,
        })

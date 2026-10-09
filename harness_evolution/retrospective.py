"""Campaign-level structured retrospective through the existing runtime seam."""
from __future__ import annotations

import json
import re
from pathlib import Path

from agent_runtime import AgentRunRequest, RuntimeRegistry, TaskContract

from .models import HarnessProposal, RetrospectiveContext


class Retrospective:
    def __init__(self, *, project_root: Path, runtime_name: str, timeout: int = 300) -> None:
        self.project_root = Path(project_root)
        self.runtime_name = runtime_name
        self.timeout = timeout

    @staticmethod
    def prompt(context: RetrospectiveContext) -> str:
        evidence = json.dumps(context.to_dict(), indent=2, sort_keys=True)
        return (
            "You are a KernelPilot campaign-level retrospective. You do not edit files, run benchmarks, "
            "or change candidates. Inspect only the structured evidence below. Return exactly one JSON object and no Markdown.\n\n"
            "Either return {\"proposal\": null} when no small, evidence-grounded mutable harness improvement exists, "
            "or {\"proposal\": {proposal_id, problem, evidence, target_type, target_path, current_behavior, "
            "proposed_change, expected_effect, risk, scope_class, source_experiences, resolution_key}}.\n\n"
            "At most one proposal. Allowed target paths are guidance/template files under templates/. Never target "
            "AgentRuntime, benchmark backend, evaluator, scoring, task identity, credentials, SSH, sandbox, or core campaign Python. "
            "Historical problems may already be fixed: use resolution_key='harness_level_benchmark_budget' for prompt-only "
            "benchmark-budget failures, so the gate can detect an existing protection. A proposal without concrete evidence is invalid.\n\n"
            f"Campaign evidence:\n{evidence}"
        )

    def run(self, *, workspace: Path, context: RetrospectiveContext) -> HarnessProposal | None:
        """Run one strict JSON attempt and, at most, one bounded repair.

        Runtime differences end at :class:`AgentRuntime`: both Codex and
        Claude produce the same final text payload, which is validated here
        against the single HarnessProposal schema.
        """
        workspace.mkdir(parents=True, exist_ok=True)
        runtime = RuntimeRegistry.create(self.runtime_name, template_root=self.project_root)
        task = TaskContract(
            objective="Review structured campaign evidence and return one optional harness proposal.",
            editable_files=("PROPOSAL.json",), benchmark_command="No benchmark permitted.",
            correctness_contract="Return structured JSON only.",
            forbidden_changes=("Do not edit harness, candidate, evaluator, runtime, or benchmark files.",),
            stop_conditions=("Stop after one structured response.",),
        )
        runtime.prepare_workspace(workspace, task)
        prompt = self.prompt(context)
        parse_error: ValueError | None = None
        for attempt in (1, 2):
            label = "retrospective" if attempt == 1 else "retrospective-repair"
            result = runtime.run(AgentRunRequest(
                workspace=workspace, prompt=prompt, timeout=self.timeout, artifact_label=label,
            ))
            (workspace / f"retrospective-attempt-{attempt}.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
            # Preserve the historical artifact name for replay tooling.  It
            # always points to the latest attempt, never silently discarding a
            # malformed initial answer.
            (workspace / "retrospective-result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
            if result.status != "completed":
                raise RuntimeError(f"retrospective runtime failed: {result.stderr_tail}")
            try:
                return self.parse_payload(result.final_message)
            except ValueError as exc:
                parse_error = exc
                if attempt == 2:
                    break
                prompt = self.repair_prompt(context, str(exc))
        raise RetrospectiveOutputError(f"retrospective remained invalid after one repair: {parse_error}")

    @staticmethod
    def parse_payload(message: str) -> HarnessProposal | None:
        """Accept only a JSON object, optionally wrapped in a JSON code fence."""
        text = message.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("retrospective did not return one JSON object") from exc
        if not isinstance(payload, dict) or set(payload) != {"proposal"}:
            raise ValueError("retrospective output must contain only proposal")
        raw = payload["proposal"]
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("proposal must be object or null")
        return HarnessProposal.from_dict(raw)

    @staticmethod
    def repair_prompt(context: RetrospectiveContext, error: str) -> str:
        return (
            "Your prior retrospective response was invalid: " + error + ". "
            "This is the one repair attempt. Return exactly one JSON object with only the key 'proposal'; "
            "use null for no proposal, or a valid HarnessProposal object. No Markdown or prose.\n\n"
            "Campaign evidence:\n" + json.dumps(context.to_dict(), indent=2, sort_keys=True)
        )


class RetrospectiveOutputError(ValueError):
    """Both bounded structured-output attempts failed validation."""

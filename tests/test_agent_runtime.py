from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent_runtime import AgentRunResult, RuntimeRegistry, TaskContract
import agent_runtime.claude  # noqa: F401 - registration is part of the public seam


class AgentRuntimeContractTest(unittest.TestCase):
    def test_registry_exposes_both_verified_runtime_implementations(self):
        self.assertEqual(RuntimeRegistry.names(), ("claude", "codex"))
        with self.assertRaisesRegex(ValueError, "unknown runtime"):
            RuntimeRegistry.create("unknown")

    def test_task_contract_rejects_missing_required_facts(self):
        with self.assertRaisesRegex(ValueError, "objective"):
            TaskContract("", ("solution/",), "bash scripts/bench.sh", "oracle", (), ())
        with self.assertRaisesRegex(ValueError, "editable_files"):
            TaskContract("objective", (), "bash scripts/bench.sh", "oracle", (), ())

    def test_task_contract_is_plain_serializable_data(self):
        task = TaskContract(
            "objective", ("solution/",), "bash scripts/bench.sh", "oracle", ("no evaluator edits",), ("one bench",)
        )
        self.assertEqual(task.to_dict()["benchmark_command"], "bash scripts/bench.sh")
        self.assertEqual("solution/", task.to_dict()["editable_files"][0])

    def test_master_phase_one_uses_runtime_seam_not_a_cli_command(self):
        from master.master import run_sub_phase1

        with TemporaryDirectory() as tmp:
            child = Path(tmp)
            (child / "solution").mkdir()
            (child / "solution" / "kernel.py").write_text("seed\n")
            (child / ".ako").mkdir()
            transcript = child / ".ako" / "phase1-transcript.jsonl"
            stderr = child / ".ako" / "phase1-stderr.log"
            transcript.write_text('{"type":"result"}\n')
            stderr.write_text("")

            class FakeRuntime:
                def run(self, request):
                    self.request = request
                    return AgentRunResult(
                        session_id="runtime-session", status="completed", exit_code=0,
                        final_message="done", events=(), timed_out=False, duration=1.0,
                        stderr_tail="", artifacts={"transcript": transcript, "stderr": stderr},
                    )

            runtime = FakeRuntime()
            with patch("master.master.RuntimeRegistry.create", return_value=runtime) as create:
                result = run_sub_phase1(child, "controlled prompt", timeout=9)
            create.assert_called_once_with("claude", template_root=Path(__file__).resolve().parents[1])
            self.assertEqual(runtime.request.prompt, "controlled prompt")
            self.assertEqual(runtime.request.artifact_label, "phase1")
            self.assertEqual((result.session_id, result.exit_status), ("runtime-session", 0))

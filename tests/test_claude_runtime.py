from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_runtime import AgentRunRequest, AgentSession, TaskContract
from agent_runtime.claude import ClaudeCodeRuntime


ROOT = Path(__file__).resolve().parents[1]


class _Process:
    pid = 1234

    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls = 0

    def communicate(self, timeout=None):
        self.calls += 1
        return self.stdout, self.stderr

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


def _task() -> TaskContract:
    return TaskContract(
        objective="Keep the existing task semantics.",
        editable_files=("solution/", "config.toml"),
        benchmark_command="bash scripts/bench.sh --first 1",
        correctness_contract="oracle available",
        forbidden_changes=("do not edit evaluator",),
        stop_conditions=("one benchmark",),
        rendered_instructions="# Kernel Optimization Task\n\nPreserved body.\n",
    )


class ClaudeCodeRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name)
        (self.workspace / "scripts").mkdir()
        self.runtime = ClaudeCodeRuntime(
            template_root=ROOT,
            permissions={"allow": ["Bash(*)", "Edit(*)", "Read(*)"]},
            executable="claude-test",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_capabilities_match_verified_native_features(self):
        capabilities = self.runtime.capabilities
        self.assertTrue(capabilities.supports_resume)
        self.assertTrue(capabilities.supports_structured_events)
        self.assertTrue(capabilities.supports_native_project_instructions)
        self.assertTrue(capabilities.supports_native_skills)
        self.assertTrue(capabilities.supports_hooks)
        self.assertFalse(capabilities.supports_native_sandbox)

    def test_prepare_workspace_creates_only_claude_native_material(self):
        created = self.runtime.prepare_workspace(self.workspace, _task())
        self.assertIn(self.workspace / "CLAUDE.md", created)
        self.assertEqual((self.workspace / "CLAUDE.md").read_text(), _task().rendered_instructions)
        self.assertTrue((self.workspace / ".claude" / "skills").is_dir())
        self.assertTrue((self.workspace / ".claude" / "hooks").is_dir())
        settings = json.loads((self.workspace / ".claude" / "settings.local.json").read_text())
        self.assertEqual(settings["permissions"]["allow"], ["Bash(*)", "Edit(*)", "Read(*)"])

    def test_command_construction_hides_cli_flags_from_request(self):
        request = AgentRunRequest(self.workspace, "do work", 30, artifact_label="phase1")
        self.assertEqual(
            self.runtime.build_command(request, session_id="session-a", resume=False),
            ["claude-test", "--print", "--verbose", "--output-format", "stream-json", "--session-id", "session-a", "do work"],
        )
        self.assertEqual(
            self.runtime.build_command(request, session_id="session-a", resume=True)[:3],
            ["claude-test", "--resume", "session-a"],
        )

    def test_run_normalizes_transcript_events_and_artifacts(self):
        transcript = "\n".join([
            json.dumps({"type": "system", "timestamp": "t0"}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}}),
            json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "content": "PASS"}]}}),
            json.dumps({"type": "result", "result": "reject: no speedup"}),
        ])
        with patch("agent_runtime.claude.subprocess.Popen", return_value=_Process(stdout=transcript)) as popen:
            result = self.runtime.run(AgentRunRequest(self.workspace, "do work", 30, artifact_label="phase1"))
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual([event.kind for event in result.events], ["session_started", "tool_call", "tool_result", "session_completed"])
        self.assertEqual(result.final_message, "reject: no speedup")
        self.assertEqual(popen.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertTrue(result.artifacts["transcript"].is_file())
        self.assertTrue((self.workspace / ".ako" / "session-id.txt").is_file())

    def test_process_failure_and_missing_executable_are_normalized(self):
        with patch("agent_runtime.claude.subprocess.Popen", return_value=_Process(stderr="bad cli", returncode=2)):
            result = self.runtime.run(AgentRunRequest(self.workspace, "do work", 30))
        self.assertEqual((result.status, result.exit_code, result.timed_out), ("failed", 2, False))
        with patch("agent_runtime.claude.subprocess.Popen", side_effect=FileNotFoundError("claude absent")):
            missing = self.runtime.run(AgentRunRequest(self.workspace, "do work", 30))
        self.assertEqual((missing.status, missing.exit_code), ("failed", -1))
        self.assertEqual(missing.events[0].kind, "error")

    def test_resume_reuses_generic_session_id(self):
        session = AgentSession("claude", "session-1", self.workspace, "completed", "now")
        with patch("agent_runtime.claude.subprocess.Popen", return_value=_Process(stdout=json.dumps({"type": "result"})) ) as popen:
            result = self.runtime.resume(session, AgentRunRequest(self.workspace, "what happened?", 30, artifact_label="phase2"))
        self.assertEqual(result.session_id, "session-1")
        self.assertIn("--resume", popen.call_args.args[0])

    def test_timeout_normalizes_after_process_group_cleanup(self):
        class TimeoutProcess(_Process):
            def __init__(self):
                super().__init__(stdout="partial", stderr="slow", returncode=None)

            def communicate(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("claude-test", timeout)
                return self.stdout, self.stderr

            def poll(self):
                return None

        process = TimeoutProcess()
        with patch("agent_runtime.claude.subprocess.Popen", return_value=process), \
             patch("agent_runtime.claude.os.getpgid", return_value=1234, create=True), \
             patch("agent_runtime.claude.os.killpg", create=True), \
             patch("agent_runtime.claude.signal.SIGKILL", 9, create=True):
            result = self.runtime.run(AgentRunRequest(self.workspace, "do work", 1))
        self.assertEqual((result.status, result.exit_code, result.timed_out), ("timed_out", -1, True))

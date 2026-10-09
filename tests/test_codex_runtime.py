from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_runtime import AgentRunRequest, AgentSession
from agent_runtime.codex import CodexRuntime
from agent_runtime.types import TaskContract


ROOT = Path(__file__).resolve().parents[1]


class _Process:
    pid = 4567

    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode
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
        objective="Keep this task runtime-neutral.",
        editable_files=("solution/", "config.toml"),
        benchmark_command="bash scripts/bench.sh --first 1",
        correctness_contract="oracle available",
        forbidden_changes=("do not edit evaluator",),
        stop_conditions=("stop after a result",),
    )


class CodexRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name)
        self.runtime = CodexRuntime(template_root=ROOT, executable="codex-test")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_capabilities_are_real_cli_capabilities_not_claude_symmetry(self):
        caps = self.runtime.capabilities
        self.assertTrue(caps.supports_resume)
        self.assertTrue(caps.supports_structured_events)
        self.assertTrue(caps.supports_native_project_instructions)
        self.assertTrue(caps.supports_native_sandbox)
        self.assertFalse(caps.supports_native_skills)
        self.assertFalse(caps.supports_hooks)

    def test_prepare_workspace_materializes_codex_contract_and_portable_skills(self):
        created = self.runtime.prepare_workspace(self.workspace, _task())
        task_file = self.workspace / "CODEX_TASK.md"
        agents_file = self.workspace / "AGENTS.md"
        skills_dir = self.workspace / "skills"
        self.assertEqual(created, (task_file, agents_file, skills_dir))

        task_text = task_file.read_text(encoding="utf-8")
        self.assertIn("Keep this task runtime-neutral.", task_text)
        self.assertIn("`solution/`", task_text)
        self.assertNotIn("CLAUDE.md", task_text)

        agents_text = agents_file.read_text(encoding="utf-8")
        self.assertIn("`CODEX_TASK.md`", agents_text)
        self.assertIn("`skills/benchmark/SKILL.md`", agents_text)
        self.assertIn("`skills/triton/SKILL.md`", agents_text)
        self.assertIn("`skills/cuda/SKILL.md`", agents_text)
        self.assertIn("`skills/profiler-nsys/SKILL.md`", agents_text)
        self.assertIn("`skills/profiler-ncu/SKILL.md`", agents_text)
        self.assertTrue((skills_dir / "benchmark" / "SKILL.md").is_file())
        self.assertTrue((skills_dir / "triton" / "SKILL.md").is_file())
        self.assertTrue((skills_dir / "cuda" / "SKILL.md").is_file())
        self.assertTrue((skills_dir / "profiler-nsys" / "SKILL.md").is_file())
        source_skills = ROOT / "templates" / "skills"
        source_files = sorted(path.relative_to(source_skills) for path in source_skills.rglob("*") if path.is_file())
        copied_files = sorted(path.relative_to(skills_dir) for path in skills_dir.rglob("*") if path.is_file())
        self.assertEqual(copied_files, source_files)
        for relative_path in source_files:
            self.assertEqual(
                (skills_dir / relative_path).read_bytes(),
                (source_skills / relative_path).read_bytes(),
            )

    def test_command_construction_keeps_private_flags_out_of_request(self):
        request = AgentRunRequest(self.workspace, "inspect then stop", 30, artifact_label="minimal")
        command = self.runtime.build_command(request)
        self.assertEqual(command[:10], [
            "codex-test", "--ask-for-approval", "never", "--config",
            "sandbox_workspace_write.network_access=true", "--sandbox",
            "workspace-write", "-C", str(self.workspace), "exec",
        ])
        self.assertIn("--json", command)
        self.assertIn("workspace-write", command)
        self.assertIn("--ask-for-approval", command)
        self.assertIn("Read CODEX_TASK.md first", command[-1])
        resumed = self.runtime.build_command(request, session_id="thread-1", resume=True)
        self.assertIn("resume", resumed)
        self.assertIn("thread-1", resumed)
        self.assertIn("-C", resumed)
        self.assertIn(str(self.workspace), resumed)

    def test_event_and_session_parsing(self):
        transcript = "\n".join([
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps({"type": "item.started", "item": {"type": "agent_message", "text": "reading"}}),
            json.dumps({"type": "item.started", "item": {"type": "command_execution"}}),
            json.dumps({"type": "item.completed", "item": {"type": "file_change"}}),
            json.dumps({"type": "turn.completed"}),
        ])
        events = self.runtime.parse_events(transcript)
        self.assertEqual(
            [event.kind for event in events],
            ["session_started", "agent_message", "tool_call", "file_change", "session_completed"],
        )
        self.assertEqual(self.runtime.session_id_from_events(events), "thread-1")

    def test_benchmark_bridge_argument_validation(self):
        from agent_runtime.codex import _FilesystemBenchmarkBridge
        self.assertEqual(
            _FilesystemBenchmarkBridge._validated_args(["--label", "branch result", "--first", "1"]),
            ["--label", "branch result", "--first", "1"],
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _FilesystemBenchmarkBridge._validated_args(["--unknown"])

    def test_benchmark_bridge_runs_only_the_fixed_remote_entrypoint(self):
        from agent_runtime.codex import (
            _BRIDGE_DIR_ENV,
            _BRIDGE_SERVER_ENV,
            _FilesystemBenchmarkBridge,
        )

        bridge = _FilesystemBenchmarkBridge(
            self.workspace,
            {_BRIDGE_DIR_ENV: "worker-bridge", "EXPERIMENTPILOT_BENCHMARK_BUDGET_FILE": "budget.json"},
            timeout=30,
        )
        bridge.start()
        try:
            request = bridge.requests / "request.json.processing"
            request.write_text(
                json.dumps({"protocol_version": 1, "args": ["--label", "branch result"]}),
                encoding="utf-8",
            )
            completed = subprocess.CompletedProcess(["run_remote.py"], 0, "REMOTE PASS", "")
            with patch("agent_runtime.codex.subprocess.run", return_value=completed) as run:
                bridge._serve_request(request)
            response = json.loads((bridge.responses / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(response, {"returncode": 0, "stdout": "REMOTE PASS", "stderr": ""})
            self.assertEqual(
                run.call_args.args[0],
                [__import__("sys").executable, "scripts/run_remote.py", "--label", "branch result"],
            )
            environment = run.call_args.kwargs["env"]
            self.assertNotIn(_BRIDGE_DIR_ENV, environment)
            self.assertEqual(environment[_BRIDGE_SERVER_ENV], "1")
            self.assertEqual(environment["EXPERIMENTPILOT_BENCHMARK_BUDGET_FILE"], "budget.json")
        finally:
            bridge.close()

    def test_run_normalizes_result_and_process_failure(self):
        transcript = "\n".join([
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps({"type": "turn.completed"}),
        ])
        artifact_dir = self.workspace / ".ako"
        artifact_dir.mkdir()
        (artifact_dir / "run-last-message.txt").write_text("KEEP", encoding="utf-8")
        with patch("agent_runtime.codex.subprocess.Popen", return_value=_Process(stdout=transcript)):
            result = self.runtime.run(AgentRunRequest(self.workspace, "do work", 30))
        self.assertEqual((result.session_id, result.status, result.final_message), ("thread-1", "completed", "KEEP"))
        with patch("agent_runtime.codex.subprocess.Popen", side_effect=FileNotFoundError("codex absent")):
            failed = self.runtime.run(AgentRunRequest(self.workspace, "do work", 30))
        self.assertEqual((failed.status, failed.exit_code, failed.events[0].kind), ("failed", -1, "error"))

    def test_resume_and_timeout_are_normalized(self):
        session = AgentSession("codex", "thread-1", self.workspace, "completed", "now")
        with patch("agent_runtime.codex.subprocess.Popen", return_value=_Process(stdout=json.dumps({"type": "turn.completed"}))) as popen:
            resumed = self.runtime.resume(session, AgentRunRequest(self.workspace, "summarize", 30, artifact_label="resume"))
        self.assertEqual(resumed.session_id, "thread-1")
        self.assertIn("resume", popen.call_args.args[0])

        class TimeoutProcess(_Process):
            def __init__(self):
                super().__init__(stdout="partial", stderr="slow", returncode=None)

            def communicate(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("codex-test", timeout)
                return self.stdout, self.stderr

            def poll(self):
                return None

        with patch("agent_runtime.codex.subprocess.Popen", return_value=TimeoutProcess()), \
             patch("agent_runtime.codex.os.getpgid", return_value=4567, create=True), \
             patch("agent_runtime.codex.os.killpg", create=True), \
             patch("agent_runtime.codex.signal.SIGKILL", 9, create=True):
            timed_out = self.runtime.run(AgentRunRequest(self.workspace, "do work", 1))
        self.assertEqual((timed_out.status, timed_out.exit_code, timed_out.timed_out), ("timed_out", -1, True))

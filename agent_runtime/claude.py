"""Claude Code implementation of the Phase 1C AgentRuntime contract."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .base import AgentRuntime
from .registry import RuntimeRegistry
from .types import AgentEvent, AgentRunRequest, AgentRunResult, AgentSession, RuntimeCapabilities, TaskContract


class ClaudeCodeRuntime(AgentRuntime):
    """Preserves the tested Claude Code child layout and CLI semantics."""

    # Claude 专属知识只应出现在这个 Adapter：CLAUDE.md、.claude/、CLI flags、
    # session UUID 和 stream-json 格式都不应泄漏到公共 AgentRuntime 接口。

    name = "claude"

    def __init__(
        self,
        *,
        template_root: Path,
        permissions: Mapping[str, Any] | None = None,
        executable: str = "claude",
    ) -> None:
        self.template_root = Path(template_root)
        self.permissions = dict(permissions or {"allow": []})
        self.executable = executable
        self._active_processes: dict[str, subprocess.Popen[str]] = {}

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_resume=True,
            supports_structured_events=True,
            supports_native_project_instructions=True,
            supports_native_skills=True,
            supports_hooks=True,
            supports_native_sandbox=False,
        )

    def prepare_workspace(self, workspace: Path, task: TaskContract) -> tuple[Path, ...]:
        # 将统一任务合同落成 Claude Code 能自动发现的原生工作区结构。
        workspace = Path(workspace)
        claude_dir = workspace / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []

        instruction_path = workspace / "CLAUDE.md"
        instruction_path.write_text(self.render_task(task), encoding="utf-8")
        created.append(instruction_path)

        scripts_guide = self.template_root / "scripts" / "CLAUDE.md"
        if scripts_guide.is_file():
            destination = workspace / "scripts" / "CLAUDE.md"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(scripts_guide.read_text(encoding="utf-8"), encoding="utf-8")
            created.append(destination)

        for source_name, destination_name in (
            ("hooks", "hooks"),
            ("commands", "commands"),
            ("skills", "skills"),
        ):
            # skills/hooks/commands 属于 Claude 原生项目能力，因此在 Adapter 内复制。
            source = self.template_root / "templates" / ("agent" if source_name != "skills" else "") / source_name
            if source.is_dir():
                destination = claude_dir / destination_name
                self._copy_tree(source, destination)
                if source_name == "hooks":
                    for hook in destination.iterdir():
                        if hook.is_file():
                            hook.chmod(0o755)
                created.append(destination)

        settings = {
            "permissions": self.permissions,
            "hooks": {
                "PostToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{
                        "type": "command",
                        "if": "Bash(*bench.sh*)",
                        "command": ".claude/hooks/advisory-review.sh",
                    }],
                }],
            },
        }
        settings_path = claude_dir / "settings.local.json"
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        created.append(settings_path)
        return tuple(created)

    @staticmethod
    def _copy_tree(source: Path, destination: Path) -> None:
        import shutil

        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

    @staticmethod
    def render_task(task: TaskContract) -> str:
        """Use the Phase 1B rendered body unchanged when the harness provides it."""
        # rendered_instructions 是兼容旧 Claude 模板的通道；没有它时才按统一字段渲染。
        if task.rendered_instructions:
            return task.rendered_instructions
        return (
            f"# Task\n\n{task.objective}\n\n"
            f"## Editable files\n\n" + "\n".join(f"- `{path}`" for path in task.editable_files) +
            f"\n\n## Benchmark\n\n`{task.benchmark_command}`\n\n"
            f"## Correctness\n\n{task.correctness_contract}\n\n"
            f"## Forbidden changes\n\n" + "\n".join(f"- {item}" for item in task.forbidden_changes) +
            f"\n\n## Stop conditions\n\n" + "\n".join(f"- {item}" for item in task.stop_conditions) + "\n"
        )

    def build_command(self, request: AgentRunRequest, *, session_id: str, resume: bool) -> list[str]:
        """Claude-only CLI details stay inside the adapter, not its public request."""
        # run 和 resume 复用同一个构造器，差别只体现在 Claude 私有参数上。
        command = [self.executable]
        if resume:
            command.extend(["--resume", session_id])
        command.extend(["--print", "--verbose", "--output-format", "stream-json"])
        if not resume:
            command.extend(["--session-id", session_id])
        command.append(request.prompt)
        return command

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        # Claude 支持在启动 CLI 前由 Harness 生成会话 UUID。
        return self._execute(request, session_id=str(uuid.uuid4()), resume=False)

    def resume(self, session: AgentSession, request: AgentRunRequest) -> AgentRunResult:
        # Runtime 名称也是 session 的类型标签，防止跨后端误用 session_id。
        if session.runtime != self.name:
            raise ValueError(f"cannot resume {session.runtime!r} session with {self.name!r}")
        return self._execute(request, session_id=session.session_id, resume=True)

    def terminate(self, session: AgentSession) -> None:
        process = self._active_processes.get(session.session_id)
        if process is None or process.poll() is not None:
            return
        try:
            # Popen 使用 start_new_session=True，因此这里杀整个进程组，避免
            # Claude 退出后 benchmark/SSH 等子孙进程继续残留。
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()

    def _execute(self, request: AgentRunRequest, *, session_id: str, resume: bool) -> AgentRunResult:
        # run/resume 的公共执行骨架：准备 artifact -> 启动 CLI -> 等待或超时
        # -> 持久化原始输出 -> 解析事件 -> 返回统一结果。
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        artifact_dir = request.workspace / ".ako"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = artifact_dir / f"{request.artifact_label}-transcript.jsonl"
        stderr_path = artifact_dir / f"{request.artifact_label}-stderr.log"
        environment = os.environ.copy()
        # Claude's WSL process invokes the child benchmark scripts. Preserve
        # normal user configuration, but prevent a Windows/GBK inherited
        # stdout codec from turning a successful benchmark report into a
        # non-zero process after it prints a Unicode advisory marker.
        environment.setdefault("PYTHONIOENCODING", "utf-8")
        environment.update({str(key): str(value) for key, value in request.environment.items()})
        command = self.build_command(request, session_id=session_id, resume=resume)
        try:
            # stdout/stderr 使用 PIPE 是为了同时保留原始审计材料和生成统一结果。
            process = subprocess.Popen(
                command,
                cwd=str(request.workspace),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                start_new_session=True,
            )
        except OSError as exc:
            stderr = str(exc)
            transcript_path.write_text("", encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            return AgentRunResult(
                session_id=session_id,
                status="failed",
                exit_code=-1,
                final_message="",
                events=(AgentEvent(kind="error", message=stderr, data={"command": command}),),
                timed_out=False,
                duration=time.monotonic() - started,
                stderr_tail=stderr[-4096:],
                artifacts={"transcript": transcript_path, "stderr": stderr_path},
                metadata={"runtime": self.name, "started_at": started_at, "resumed": resume},
            )
        self._active_processes[session_id] = process
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=request.timeout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            # 超时不是普通非零退出：先清理进程组，再单独标记 timed_out。
            direct_exit_code = process.poll()
            self.terminate(AgentSession(self.name, session_id, request.workspace, "running", started_at))
            stdout, stderr = process.communicate()
            if direct_exit_code is None:
                exit_code, timed_out = -1, True
            else:
                exit_code = direct_exit_code
        finally:
            self._active_processes.pop(session_id, None)

        artifact_dir.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(stdout or "", encoding="utf-8")
        stderr_path.write_text(stderr or "", encoding="utf-8")
        (artifact_dir / "session-id.txt").write_text(session_id + "\n", encoding="utf-8")
        events = tuple(self.parse_events(stdout or ""))
        # 从这里开始，上层只处理统一事件，不需要理解 Claude 的原生 record。
        status = "timed_out" if timed_out else ("completed" if exit_code == 0 else "failed")
        return AgentRunResult(
            session_id=session_id,
            status=status,
            exit_code=exit_code,
            final_message=self._final_message(events),
            events=events,
            timed_out=timed_out,
            duration=time.monotonic() - started,
            stderr_tail=(stderr or "")[-4096:],
            artifacts={"transcript": transcript_path, "stderr": stderr_path},
            metadata={"runtime": self.name, "started_at": started_at, "resumed": resume},
        )

    @staticmethod
    def parse_events(transcript: str) -> list[AgentEvent]:
        """Conservatively classify stream-json; unrecognized records remain raw."""
        # 保守归一化：只映射确定的公共类别；未知格式保留为 raw，避免信息丢失。
        events: list[AgentEvent] = []
        for line in transcript.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                events.append(AgentEvent(kind="raw", message=line, data={"raw": line}))
                continue
            record_type = str(record.get("type", ""))
            payload = record.get("message", record)
            content = payload.get("content") if isinstance(payload, dict) else None
            kind = "raw"
            content_types = {
                item.get("type") for item in content if isinstance(item, dict)
            } if isinstance(content, list) else set()
            if record_type in {"system", "init"}:
                kind = "session_started"
            elif record_type in {"file_change", "file_changed"}:
                kind = "file_change"
            elif record_type in {"tool_use", "tool_call"} or "tool_use" in content_types:
                kind = "tool_call"
            elif record_type == "tool_result" or "tool_result" in content_types:
                kind = "tool_result"
            elif record_type in {"result", "completed"}:
                kind = "session_completed"
            elif record_type in {"assistant", "message"}:
                kind = "agent_message"
            events.append(AgentEvent(kind=kind, timestamp=record.get("timestamp"), data=record))
        return events

    @staticmethod
    def _final_message(events: tuple[AgentEvent, ...]) -> str:
        for event in reversed(events):
            data = event.data
            if event.kind == "session_completed":
                return str(data.get("result") or data.get("message") or "")
        return ""


# Phase 1C deliberately exposes only the verified runtime. Phase 1D adds
# Codex registration only after it has a separately validated implementation.
RuntimeRegistry.register("claude", ClaudeCodeRuntime)

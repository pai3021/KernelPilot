"""Codex CLI implementation of the frozen AgentRuntime contract."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .base import AgentRuntime
from .registry import RuntimeRegistry
from .types import AgentEvent, AgentRunRequest, AgentRunResult, AgentSession, RuntimeCapabilities, TaskContract


_BRIDGE_DIR_ENV = "EXPERIMENTPILOT_BENCHMARK_BRIDGE_DIR"
_BRIDGE_SERVER_ENV = "EXPERIMENTPILOT_BENCHMARK_BRIDGE_SERVER"
_BRIDGE_TIMEOUT_ENV = "EXPERIMENTPILOT_BENCHMARK_BRIDGE_TIMEOUT"
_BRIDGE_VALUE_FLAGS = frozenset({
    "--label", "--first", "--group", "--exclude-group", "--index",
    "--variance-check", "--ab-compare",
})
_BRIDGE_SWITCH_FLAGS = frozenset({"--force-baseline", "-q", "--quiet", "--smoke", "--capture-logs"})


class _FilesystemBenchmarkBridge:
    """Serve a worker's fixed remote benchmark through workspace files only."""

    # Codex 的 workspace-write sandbox 可能禁止新建网络 socket。Worker 因此只
    # 写请求文件，Harness 侧线程校验参数后代跑固定 run_remote.py 并写回响应。
    # 这是传输适配，不会改变 benchmark 的正确性、计分或预算语义。

    def __init__(self, workspace: Path, environment: Mapping[str, str], *, timeout: int) -> None:
        self.workspace = workspace
        self.root = workspace / ".ako" / "benchmark-bridge"
        self.requests = self.root / "requests"
        self.responses = self.root / "responses"
        self.environment = dict(environment)
        self.timeout = timeout
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="codex-benchmark-bridge", daemon=True)

    def start(self) -> None:
        self.requests.mkdir(parents=True, exist_ok=True)
        self.responses.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @staticmethod
    def _validated_args(raw: object) -> list[str]:
        # 严格白名单避免 Worker 借 bridge 执行任意父进程命令。
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("Benchmark bridge request has invalid arguments")
        args = list(raw)
        position = 0
        while position < len(args):
            item = args[position]
            if item in _BRIDGE_SWITCH_FLAGS:
                position += 1
                continue
            if item in _BRIDGE_VALUE_FLAGS:
                if position + 1 >= len(args) or args[position + 1].startswith("-"):
                    raise ValueError(f"Benchmark bridge option {item} requires a value")
                position += 2
                continue
            if any(item.startswith(option + "=") for option in _BRIDGE_VALUE_FLAGS if option.startswith("--")):
                position += 1
                continue
            raise ValueError(f"Benchmark bridge rejected unsupported argument: {item}")
        return args

    def _respond(self, path: Path, value: dict[str, object]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value), encoding="utf-8")
        temporary.replace(path)

    def _serve_request(self, request_path: Path) -> None:
        response_path = self.responses / request_path.name.removesuffix(".processing")
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            if request.get("protocol_version") != 1:
                raise ValueError("Benchmark bridge protocol mismatch")
            args = self._validated_args(request.get("args"))
            environment = dict(self.environment)
            environment.pop(_BRIDGE_DIR_ENV, None)
            environment[_BRIDGE_SERVER_ENV] = "1"
            completed = subprocess.run(
                [sys.executable, "scripts/run_remote.py", *args],
                cwd=self.workspace,
                text=True,
                capture_output=True,
                timeout=self.timeout + 30,
                check=False,
                env=environment,
            )
            response = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
        except Exception as exc:  # Normalize bridge faults into ordinary command output.
            response = {"returncode": 1, "stdout": "", "stderr": f"Benchmark bridge error: {type(exc).__name__}: {exc}\n"}
        self._respond(response_path, response)

    def _serve(self) -> None:
        while not self._stop.is_set():
            for request_path in sorted(self.requests.glob("*.json")):
                claimed = request_path.with_name(request_path.name + ".processing")
                try:
                    request_path.replace(claimed)
                except FileNotFoundError:
                    continue
                self._serve_request(claimed)
            self._stop.wait(0.05)


class CodexRuntime(AgentRuntime):
    """Run Codex CLI non-interactively inside one already-materialized child."""

    # Codex 专属知识集中在这里：AGENTS.md/CODEX_TASK.md、exec CLI、sandbox、
    # thread_id 和 JSONL schema；上层 Campaign 只看统一接口。

    name = "codex"

    def __init__(
        self,
        *,
        template_root: Path,
        executable: str | None = None,
        permissions: Mapping[str, Any] | None = None,
    ) -> None:
        self.template_root = Path(template_root)
        self.executable = executable or self.discover_executable()
        # Accepted only so RuntimeRegistry can construct both runtimes from the
        # same spawn selector. Codex permissions are CLI sandbox policy, not a
        # copied Claude allow-list.
        self.permissions = dict(permissions or {})
        self._active_processes: dict[str, subprocess.Popen[str]] = {}

    @staticmethod
    def discover_executable() -> str:
        """Locate the native CLI used by the WSL/Linux control plane."""
        # 查找顺序：显式环境变量 -> 当前 PATH -> 用户已有 NVM 安装 -> 交给
        # Popen 报标准“命令不存在”错误。这里不会回退到 Windows codex.exe。
        override = os.environ.get("CODEX_EXECUTABLE")
        if override:
            return override
        native = shutil.which("codex")
        if native:
            return native
        nvm_candidates = sorted(Path.home().glob(".nvm/versions/node/*/bin/codex"))
        if nvm_candidates:
            return str(nvm_candidates[-1])
        return "codex"

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_resume=True,
            supports_structured_events=True,
            supports_native_project_instructions=True,
            supports_native_skills=False,
            supports_hooks=False,
            supports_native_sandbox=True,
        )

    def prepare_workspace(self, workspace: Path, task: TaskContract) -> tuple[Path, ...]:
        # 将统一 TaskContract 渲染成 Codex 工作区文件；不会创建 .claude 目录。
        workspace = Path(workspace)
        task_path = workspace / "CODEX_TASK.md"
        task_path.write_text(self.render_task(task), encoding="utf-8")

        agents_path = workspace / "AGENTS.md"
        agents_path.write_text(self.render_agents(), encoding="utf-8")

        created = [task_path, agents_path]
        skills_source = self.template_root / "templates" / "skills"
        if skills_source.is_dir():
            skills_path = workspace / "skills"
            self._copy_tree(skills_source, skills_path)
            created.append(skills_path)
        return tuple(created)

    @staticmethod
    def _copy_tree(source: Path, destination: Path) -> None:
        """Copy portable Markdown guidance without enabling a Codex skill API."""
        # 当前只是把 Kernel Skills 当普通 Markdown 指南使用，而非宣称原生能力。
        shutil.copytree(source, destination, dirs_exist_ok=True)

    @staticmethod
    def render_agents() -> str:
        return (
            "# KernelPilot Codex Worker\n\n"
            "You are a GPU kernel optimization worker. Read `CODEX_TASK.md` first "
            "for this run's TaskContract.\n\n"
            "Correctness is a hard gate: compare performance only after correctness "
            "PASS. Strictly follow the editable files and forbidden changes. Do not "
            "modify the benchmark evaluator or reference implementation, or bypass the "
            "fixed evaluator. Validate every candidate with the existing benchmark command; "
            "do not claim an optimization succeeded from reasoning alone.\n\n"
            "Read the following Kernel Skills when the implementation strategy calls for them:\n\n"
            "- benchmark / evaluation: `skills/benchmark/SKILL.md`\n"
            "- Triton candidate: `skills/triton/SKILL.md`\n"
            "- CUDA candidate: `skills/cuda/SKILL.md`\n"
            "- C++ candidate: `skills/cpp/SKILL.md`\n"
            "- CuTe DSL: `skills/cute-dsl/SKILL.md`\n"
            "- TileLang: `skills/tilelang/SKILL.md`\n"
            "- timeline performance diagnosis: `skills/profiler-nsys/SKILL.md` only when "
            "the corresponding profiler command is available\n"
            "- deep hardware-counter diagnosis: `skills/profiler-ncu/SKILL.md` only when "
            "the corresponding profiler command and counter permissions are available\n"
            "- correctness diagnosis: `skills/sanitizer/SKILL.md` only when the "
            "corresponding sanitizer command is available\n\n"
            "If a Skill conflicts with `CODEX_TASK.md`, `CODEX_TASK.md` takes precedence.\n"
        )

    @staticmethod
    def render_task(task: TaskContract) -> str:
        return (
            "# KernelPilot Codex Task\n\n"
            f"## Objective\n\n{task.objective}\n\n"
            "## Editable files\n\n" + "\n".join(f"- `{path}`" for path in task.editable_files) +
            f"\n\n## Benchmark command\n\n`{task.benchmark_command}`\n\n"
            f"## Correctness contract\n\n{task.correctness_contract}\n\n"
            "## Forbidden changes\n\n" + "\n".join(f"- {item}" for item in task.forbidden_changes) +
            "\n\n## Stop conditions\n\n" + "\n".join(f"- {item}" for item in task.stop_conditions) +
            "\n"
        )

    @staticmethod
    def _runtime_prompt(request: AgentRunRequest) -> str:
        return (
            "Read CODEX_TASK.md first; it is the runtime-neutral task contract. "
            "Then follow this run-specific request:\n\n" + request.prompt
        )

    def build_command(
        self, request: AgentRunRequest, *, session_id: str | None = None, resume: bool = False
    ) -> list[str]:
        # Codex 私有参数停留在 Adapter 内；AgentRunRequest 不包含 exec、--json、
        # --output-last-message 或 workspace-write 等实现细节。
        artifact_dir = request.workspace / ".ako"
        final_message = artifact_dir / f"{request.artifact_label}-last-message.txt"
        common = [
            self.executable,
            "--ask-for-approval", "never",
            "--config", "sandbox_workspace_write.network_access=true",
            "--sandbox", "workspace-write",
            "-C", str(request.workspace),
            "exec",
            "--json",
            "--output-last-message", str(final_message),
        ]
        if request.model:
            common.extend(["--model", request.model])
        reasoning_effort = request.environment.get("AGENT_RUNTIME_REASONING_EFFORT")
        if reasoning_effort:
            common.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
        if resume:
            # 将公共 resume(session, request) 翻译为 Codex 的 exec resume 协议。
            if not session_id:
                raise ValueError("Codex resume requires a session_id from a real prior event")
            return common[:10] + ["resume", "--json", "--output-last-message", str(final_message), session_id, request.prompt]
        return common + [self._runtime_prompt(request)]

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        # 新 Codex 会话的真实 thread_id 尚未知，要等 JSONL 首批事件返回后提取。
        return self._execute(request, session_id=None, resume=False)

    def resume(self, session: AgentSession, request: AgentRunRequest) -> AgentRunResult:
        # 只能使用此前从真实 Codex 事件中得到的非空 thread_id。
        if session.runtime != self.name:
            raise ValueError(f"cannot resume {session.runtime!r} session with {self.name!r}")
        if not session.session_id:
            raise ValueError("Codex cannot resume because the prior CLI stream lacked a session id")
        return self._execute(request, session_id=session.session_id, resume=True)

    def terminate(self, session: AgentSession) -> None:
        process = self._active_processes.get(session.session_id)
        if process is None or process.poll() is not None:
            return
        try:
            # 与 Claude Adapter 一样清理整个 POSIX 进程组，防止孙进程泄漏。
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError):
            process.kill()

    def _execute(self, request: AgentRunRequest, *, session_id: str | None, resume: bool) -> AgentRunResult:
        # run/resume 共用执行骨架；二者只在是否已有 session_id 和命令构造上不同。
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        artifact_dir = request.workspace / ".ako"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = artifact_dir / f"{request.artifact_label}-transcript.jsonl"
        stderr_path = artifact_dir / f"{request.artifact_label}-stderr.log"
        environment = os.environ.copy()
        environment.update({str(key): str(value) for key, value in request.environment.items()})
        bridge = _FilesystemBenchmarkBridge(request.workspace, environment, timeout=request.timeout)
        # bridge 生命周期严格包住 Codex 子进程，并在 finally 中保证关闭。
        bridge.start()
        environment[_BRIDGE_DIR_ENV] = str(bridge.root)
        environment[_BRIDGE_TIMEOUT_ENV] = str(request.timeout + 30)
        executable_path = Path(self.executable)
        if ".nvm/versions/node/" in str(executable_path):
            node_bin = str(executable_path.parent)
            environment["PATH"] = node_bin + os.pathsep + environment.get("PATH", "")
        command = self.build_command(request, session_id=session_id, resume=resume)
        try:
            # stdin=DEVNULL 防止非交互任务在最终回复后继续等待终端输入。
            process = subprocess.Popen(
                command, cwd=str(request.workspace), text=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
                start_new_session=True,
            )
        except OSError as exc:
            bridge.close()
            stderr = str(exc)
            transcript_path.write_text("", encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            return AgentRunResult(
                session_id=session_id or "", status="failed", exit_code=-1,
                final_message="", events=(AgentEvent(kind="error", message=stderr),),
                timed_out=False, duration=time.monotonic() - started, stderr_tail=stderr[-4096:],
                artifacts={"transcript": transcript_path, "stderr": stderr_path},
                metadata={"runtime": self.name, "started_at": started_at, "resumed": resume},
            )

        process_key = session_id or f"pending-{process.pid}"
        self._active_processes[process_key] = process
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=request.timeout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            # 超时后清理进程组；timed_out 与普通 CLI 失败分开表达。
            direct_exit_code = process.poll()
            self.terminate(AgentSession(self.name, process_key, request.workspace, "running", started_at))
            stdout, stderr = process.communicate()
            if direct_exit_code is None:
                exit_code, timed_out = -1, True
            else:
                exit_code = direct_exit_code
        finally:
            self._active_processes.pop(process_key, None)
            bridge.close()

        artifact_dir.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(stdout or "", encoding="utf-8")
        stderr_path.write_text(stderr or "", encoding="utf-8")
        events = tuple(self.parse_events(stdout or ""))
        # 新会话从 thread.started 提取 ID；resume 流未重复报告时沿用旧 ID。
        observed_session_id = self.session_id_from_events(events) or session_id or ""
        final_message_path = artifact_dir / f"{request.artifact_label}-last-message.txt"
        final_message = (
            final_message_path.read_text(encoding="utf-8", errors="replace")
            if final_message_path.is_file() else self._final_message(events)
        )
        status = "timed_out" if timed_out else ("completed" if exit_code == 0 else "failed")
        return AgentRunResult(
            session_id=observed_session_id, status=status, exit_code=exit_code,
            final_message=final_message, events=events, timed_out=timed_out,
            duration=time.monotonic() - started, stderr_tail=(stderr or "")[-4096:],
            artifacts={"transcript": transcript_path, "stderr": stderr_path, "final_message": final_message_path},
            metadata={
                "runtime": self.name, "started_at": started_at, "resumed": resume,
                "session_id_available": bool(observed_session_id),
                "executable": self.executable,
            },
        )

    @staticmethod
    def parse_events(transcript: str) -> list[AgentEvent]:
        """Map Codex JSONL conservatively; unknown schemas remain raw."""
        # 只把稳定 schema 映射为公共 kind，其余完整保留为 raw 供诊断。
        events: list[AgentEvent] = []
        for line in transcript.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                events.append(AgentEvent(kind="raw", message=line, data={"raw": line}))
                continue
            record_type = str(record.get("type", ""))
            item = record.get("item", {}) if isinstance(record.get("item"), dict) else {}
            item_type = str(item.get("type", ""))
            kind = "raw"
            if record_type in {"thread.started", "turn.started"}:
                kind = "session_started"
            elif record_type in {"turn.completed", "turn.failed"}:
                kind = "session_completed" if record_type == "turn.completed" else "error"
            elif record_type in {"item.started", "item.updated", "item.completed"}:
                if item_type in {"command_execution", "mcp_tool_call"}:
                    kind = "tool_call" if record_type != "item.completed" else "tool_result"
                elif item_type in {"file_change", "file_change_request"}:
                    kind = "file_change"
                elif item_type in {"agent_message", "reasoning"}:
                    kind = "agent_message"
                elif item_type == "error":
                    kind = "error"
            elif record_type in {"error", "thread.error"}:
                kind = "error"
            events.append(AgentEvent(kind=kind, timestamp=record.get("timestamp"), data=record))
        return events

    @staticmethod
    def session_id_from_events(events: tuple[AgentEvent, ...] | list[AgentEvent]) -> str | None:
        # 兼容已观察到的多个字段名，但只接受真实事件中出现的非空字符串。
        for event in events:
            record = event.data
            for key in ("thread_id", "session_id", "id"):
                value = record.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _final_message(events: tuple[AgentEvent, ...]) -> str:
        for event in reversed(events):
            item = event.data.get("item", {})
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    return text
        return ""


RuntimeRegistry.register("codex", CodexRuntime)

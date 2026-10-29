"""Native Codex launch with a minimal benchmark endpoint and no task injection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Mapping

from scripts.prepare_codex_baseline import MinimalBenchmarkEndpoint


def native_command(executable: str, workspace: Path, model: str, reasoning_effort: str) -> list[str]:
    return [executable, "--ask-for-approval", "never", "--config", "sandbox_workspace_write.network_access=true",
            "--sandbox", "workspace-write", "-C", str(workspace), "exec", "--json", "--model", model,
            "--config", f'model_reasoning_effort="{reasoning_effort}"',
            "--output-last-message", str(workspace / ".native-last-message.txt"),
            "Read PROBLEM.md, implement the task, and leave your chosen final candidate in solution/kernel.py."]


def _kill(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        process.kill()


def run_native(workspace: Path, observer_root: Path, *, executable: str, model: str,
               reasoning_effort: str, evaluate: Callable[[str], Mapping[str, Any]],
               timeout: float = 1800, endpoint_join_timeout: float = 1200, search_limit: int = 6) -> dict[str, Any]:
    """The external callback owns GPU access; this launcher owns only its CLI group."""
    workspace, observer_root = Path(workspace).resolve(), Path(observer_root).resolve()
    if timeout <= 0 or endpoint_join_timeout <= 0:
        raise ValueError("Timeouts must be positive")
    if (observer_root / "native-request.json").exists():
        raise FileExistsError("This native run already started; never resume it implicitly")
    endpoint = MinimalBenchmarkEndpoint(workspace, observer_root / "endpoint", evaluate, search_limit=search_limit)
    stopped, faults = threading.Event(), []
    def serve():
        try:
            while not stopped.is_set():
                endpoint.serve_once()
                stopped.wait(0.05)
        except BaseException as exc:
            faults.append(repr(exc))
    thread = threading.Thread(target=serve, name="minimal-fixed-evaluator", daemon=True)
    command = native_command(executable, workspace, model, reasoning_effort)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("EXPERIMENTPILOT_", "BUNDLE_"))
                   and key != "AGENT_RUNTIME_REASONING_EFFORT"}
    environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    observer_root.mkdir(parents=True, exist_ok=True)
    request = {"command": command, "workspace": str(workspace), "model": model,
               "reasoning_effort": reasoning_effort, "timeout": timeout,
               "context": "PROBLEM.md only; no KernelPilot task or runtime prompt"}
    (observer_root / "native-request.json").write_text(json.dumps(request, indent=2) + "\n")
    stdout, stderr = observer_root / "native-transcript.jsonl", observer_root / "native-stderr.log"
    started = time.monotonic()
    timed_out = blocked = False
    thread.start()
    try:
        with stdout.open("w") as out, stderr.open("w") as err:
            process = subprocess.Popen(command, cwd=workspace, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, start_new_session=True)
            while process.poll() is None:
                state = json.loads(endpoint.ledger.read_text()) if endpoint.ledger.exists() else {}
                if state.get("incomplete") or faults:
                    blocked = True
                    _kill(process)
                    break
                if time.monotonic() - started >= timeout:
                    timed_out = True
                    _kill(process)
                    break
                time.sleep(0.1)
            process.wait()
            exit_code = process.returncode
    finally:
        stopped.set()
        thread.join(timeout=endpoint_join_timeout)
    if thread.is_alive():
        # A later terminal comparison must never overlap an in-flight search.
        raise RuntimeError("Search evaluator is still in flight; stop this cohort and inspect it")
    state = json.loads(endpoint.ledger.read_text()) if endpoint.ledger.exists() else {}
    blocked = blocked or bool(state.get("incomplete")) or bool(faults)
    usage: dict[str, int] = {}
    session_id = ""
    for line in stdout.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("type") == "thread.started":
            session_id = record.get("thread_id", "")
        if record.get("type") == "turn.completed":
            for key, value in record.get("usage", {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
    result = {"status": "blocked" if blocked else "timed_out" if timed_out else
                         "completed" if exit_code == 0 else "failed",
              "exit_code": exit_code, "session_id": session_id, "timed_out": timed_out,
              "duration_seconds": time.monotonic() - started, "usage": usage or None,
              "submission_attempts": state.get("submission_attempts", 0), "endpoint_faults": faults,
              "artifacts": {"transcript": str(stdout), "stderr": str(stderr)}}
    (observer_root / "native-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

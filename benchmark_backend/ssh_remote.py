"""SSH transport for the fixed KernelBench evaluator protocol."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
import uuid
from typing import Any, Mapping, Sequence

from .base import BenchmarkBackend, BenchmarkExecutionResult, ExecutionStatus, failure_results


_ALLOWED_GPUS = frozenset({0, 1, 2, 3, 5})
_TRANSFER_TIMEOUT_SECONDS = 90
_TRANSFER_ATTEMPTS = 2
_PREFLIGHT_ATTEMPTS = 2
_PREFLIGHT_RETRY_DELAY_SECONDS = 2
_TRANSIENT_PREFLIGHT_MARKERS = frozenset({
    "missing remote_python",
    "missing KernelBench checkout",
    "remote_runs_dir not writable",
})
_DEFAULT_REMOTE_COMPUTE_SANITIZER = ""
_DEFAULT_REMOTE_NSYS = ""


@dataclass(frozen=True, slots=True)
class RemoteSSHConfig:
    """Explicit, key-based SSH configuration; no passwords or shell profiles."""

    remote_host: str
    remote_repo: str
    remote_python: str
    remote_runs_dir: str
    remote_kernelbench: str
    remote_compute_sanitizer: str = _DEFAULT_REMOTE_COMPUTE_SANITIZER
    remote_nsys: str = _DEFAULT_REMOTE_NSYS
    preferred_gpu: int | None = None
    timeout_seconds: int = 300
    max_gpu_memory_mb: int = 1024
    ssh_binary: str = "ssh"
    scp_binary: str = "scp"
    ssh_config_file: str = ""
    identity_file: str = ""
    known_hosts_file: str = ""
    ssh_control_path: str = ""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RemoteSSHConfig":
        def required(name: str) -> str:
            value = values.get(name, "")
            if not value:
                raise ValueError(f"Remote benchmark config requires {name!r}")
            return str(value)

        raw_gpu = values.get("preferred_gpu", values.get("gpu"))
        gpu = None if raw_gpu in (None, "") else int(raw_gpu)
        if gpu == 4:
            raise ValueError("Physical GPU 4 is reserved and cannot be configured")
        if gpu is not None and gpu not in _ALLOWED_GPUS:
            raise ValueError(f"preferred_gpu must be one of {sorted(_ALLOWED_GPUS)}, got {gpu}")
        timeout = int(values.get("evaluation_timeout", values.get("timeout_seconds", 300)))
        if timeout < 1:
            raise ValueError("evaluation_timeout must be positive")
        return cls(
            remote_host=required("remote_host"),
            remote_repo=required("remote_repo"),
            remote_python=required("remote_python"),
            remote_runs_dir=required("remote_runs_dir"),
            remote_kernelbench=required("remote_kernelbench"),
            remote_compute_sanitizer=str(
                values.get("remote_compute_sanitizer", _DEFAULT_REMOTE_COMPUTE_SANITIZER)
            ),
            remote_nsys=str(values.get("remote_nsys", _DEFAULT_REMOTE_NSYS)),
            preferred_gpu=gpu,
            timeout_seconds=timeout,
            max_gpu_memory_mb=int(values.get("max_gpu_memory_mb", 1024)),
            ssh_binary=str(values.get("ssh_binary", "ssh")),
            scp_binary=str(values.get("scp_binary", "scp")),
            ssh_config_file=str(values.get("ssh_config_file", "")),
            identity_file=str(values.get("identity_file", "")),
            known_hosts_file=str(values.get("known_hosts_file", "")),
            ssh_control_path=str(values.get("ssh_control_path", "")),
        )


class RemoteSSHBenchmarkBackend(BenchmarkBackend):
    """Transfer one packed candidate to an isolated remote run workspace."""

    def __init__(self, config: RemoteSSHConfig | Mapping[str, Any]):
        self.config = config if isinstance(config, RemoteSSHConfig) else RemoteSSHConfig.from_mapping(config)

    def _run(self, args: list[str], *, timeout: int, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, text=True, input=input_text, capture_output=True, timeout=timeout, check=False)

    def _ssh(self, command: str, *, timeout: int) -> subprocess.CompletedProcess[str]:
        return self._run([self.config.ssh_binary, *self._ssh_config_args(), self.config.remote_host, command], timeout=timeout)

    def _scp_to(self, local: Path, remote: str, *, timeout: int) -> subprocess.CompletedProcess[str]:
        return self._run([
            self.config.scp_binary, *self._ssh_config_args(), self._scp_local_path(local),
            f"{self.config.remote_host}:{remote}",
        ], timeout=timeout)

    def _scp_from(self, remote: str, local: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
        return self._run([
            self.config.scp_binary, *self._ssh_config_args(), f"{self.config.remote_host}:{remote}",
            self._scp_local_path(local),
        ], timeout=timeout)

    @staticmethod
    def _retry_transfer(operation: Any) -> tuple[subprocess.CompletedProcess[str], int]:
        """Retry one transient SCP timeout without retrying a failed evaluation.

        The request and evaluator helper are small control-plane files.  A
        transport timeout before the remote evaluator starts is safe to retry;
        non-zero SCP exits are returned to the caller unchanged so permission,
        path, and protocol errors remain fail-fast.
        """
        for attempt in range(1, _TRANSFER_ATTEMPTS + 1):
            try:
                return operation(timeout=_TRANSFER_TIMEOUT_SECONDS), attempt
            except subprocess.TimeoutExpired:
                if attempt == _TRANSFER_ATTEMPTS:
                    raise
        raise AssertionError("unreachable")

    def _scp_local_path(self, path: Path) -> str:
        """Translate a WSL-mounted path only when using Windows OpenSSH.

        A Windows control-plane process already supplies native paths.  This
        small compatibility branch lets the same backend be integration-tested
        from WSL through ``scp.exe`` without changing the protocol.
        """
        value = str(path)
        if not self.config.scp_binary.lower().endswith(".exe") or not value.startswith("/mnt/"):
            return value
        pieces = Path(value).parts
        if len(pieces) < 4 or len(pieces[2]) != 1 or not pieces[2].isalpha():
            return value
        return pieces[2].upper() + ":\\" + "\\".join(pieces[3:])

    def _ssh_config_args(self) -> list[str]:
        args: list[str] = []
        if self.config.ssh_config_file:
            args.extend(["-F", self.config.ssh_config_file])
        if self.config.identity_file:
            args.extend(["-i", self.config.identity_file])
        if self.config.known_hosts_file:
            args.extend(["-o", f"UserKnownHostsFile={self.config.known_hosts_file}"])
        if self.config.ssh_control_path:
            # A WSL parent opens this master before Codex starts.  Codex's
            # workspace-write sandbox may deny new AF_INET sockets, while a
            # connection to this child-local AF_UNIX control socket remains
            # within the worker's allowed workspace boundary.
            args.extend([
                "-o", "ControlMaster=no",
                "-o", f"ControlPath={self.config.ssh_control_path}",
            ])
        return args

    def _select_gpu(self) -> int:
        query = "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
        completed = self._ssh(query, timeout=min(20, self.config.timeout_seconds))
        if completed.returncode != 0:
            raise RuntimeError(f"GPU query failed: {completed.stderr.strip() or completed.stdout.strip()}")
        rows: list[tuple[int, int, int]] = []
        for line in completed.stdout.splitlines():
            try:
                gpu, memory, utilization = (int(part.strip()) for part in line.split(","))
                rows.append((gpu, memory, utilization))
            except ValueError:
                continue
        by_id = {gpu: (memory, utilization) for gpu, memory, utilization in rows}
        candidates = [self.config.preferred_gpu] if self.config.preferred_gpu is not None else sorted(_ALLOWED_GPUS)
        for gpu in candidates:
            if gpu not in _ALLOWED_GPUS:
                continue
            state = by_id.get(gpu)
            if state and state[0] <= self.config.max_gpu_memory_mb and state[1] == 0:
                return gpu
        requested = self.config.preferred_gpu
        if requested is not None:
            state = by_id.get(requested)
            detail = "not reported" if state is None else f"memory={state[0]} MiB util={state[1]}%"
            raise RuntimeError(f"Preferred GPU {requested} is unavailable ({detail})")
        raise RuntimeError(f"No idle allowed GPU found in {sorted(_ALLOWED_GPUS)}; GPU 4 is reserved")

    def _preflight(self, physical_gpu: int) -> dict[str, Any]:
        """Check only the execution prerequisites before candidate transfer."""
        python = shlex.quote(self.config.remote_python)
        kernelbench = shlex.quote(self.config.remote_kernelbench)
        runs_dir = shlex.quote(self.config.remote_runs_dir)
        witness = shlex.quote(
            "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
            "print('PREFLIGHT torch=' + torch.__version__ + ' cuda=' + str(torch.version.cuda) + "
            "' device=' + torch.cuda.get_device_name(0))"
        )
        command = (
            f"test -x {python} || {{ echo missing remote_python; exit 20; }}; "
            f"test -d {kernelbench} || {{ echo missing KernelBench checkout; exit 21; }}; "
            f"mkdir -p -- {runs_dir} && test -w {runs_dir} || {{ echo remote_runs_dir not writable; exit 22; }}; "
            f"CUDA_VISIBLE_DEVICES={physical_gpu} {python} -c {witness}"
        )
        for attempt in range(1, _PREFLIGHT_ATTEMPTS + 1):
            completed = self._ssh(command, timeout=min(30, self.config.timeout_seconds))
            if completed.returncode == 0:
                return {
                    "preflight": "PASS",
                    "preflight_log": completed.stdout.strip(),
                    "preflight_attempts": attempt,
                }
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
            transient_environment = any(marker in detail for marker in _TRANSIENT_PREFLIGHT_MARKERS)
            if attempt < _PREFLIGHT_ATTEMPTS and transient_environment:
                # No candidate has been uploaded or evaluated yet. A bounded
                # retry is therefore safe and does not consume a formal GPU
                # evaluation or the branch's benchmark budget.
                time.sleep(_PREFLIGHT_RETRY_DELAY_SECONDS)
                continue
            raise RuntimeError(f"Remote preflight failed: {detail}")
        raise AssertionError("unreachable")

    def run_sanitizer(
        self,
        candidate_blob: str,
        task_id: str,
        *,
        tool: str = "memcheck",
        timeout_seconds: int = 300,
    ) -> dict[str, Any]:
        """Run one diagnostic-only sanitizer workload outside the evaluator path.

        This is deliberately not part of ``evaluate_blob``: its result is
        evidence for debugging, never a KernelBench correctness or latency
        verdict.
        """
        if tool not in {"memcheck", "racecheck", "initcheck", "synccheck"}:
            raise ValueError(f"Unsupported compute-sanitizer tool: {tool}")
        if timeout_seconds < 1:
            raise ValueError("Sanitizer timeout must be positive")
        started = time.monotonic()
        result: dict[str, Any] = {
            "protocol_version": 1,
            "diagnostic": "compute-sanitizer",
            "task_id": task_id,
            "tool": tool,
            "status": "INFRA_FAILURE",
            "command": "",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "sanitizer_error": False,
            "formal_correctness_verdict": "NOT_RUN",
        }
        try:
            physical_gpu = self._select_gpu()
            remote_run = self._remote_join(
                self.config.remote_runs_dir, f"ep-diagnostic-{int(time.time())}-{uuid.uuid4().hex[:12]}"
            )
            result.update({"physical_gpu_id": physical_gpu, "visible_device": "cuda:0", "remote_workspace": remote_run})
            tool_path = shlex.quote(self.config.remote_compute_sanitizer)
            preflight = self._ssh(
                f"test -x {tool_path} || {{ echo missing remote_compute_sanitizer; exit 23; }}; "
                f"test -d {shlex.quote(self.config.remote_kernelbench)} || "
                "{ echo missing KernelBench checkout; exit 21; }",
                timeout=min(30, timeout_seconds),
            )
            if preflight.returncode != 0:
                result.update({
                    "status": "UNAVAILABLE",
                    "exit_code": preflight.returncode,
                    "stderr": preflight.stderr or preflight.stdout,
                })
                return result

            request = {
                "candidate_blob": candidate_blob,
                "task_id": task_id,
                "kernelbench_path": self.config.remote_kernelbench,
            }
            helper = Path(__file__).resolve().parent.parent / "scripts" / "remote_diagnostic.py"
            staging_root = Path.cwd() if self.config.scp_binary.lower().endswith(".exe") else None
            with tempfile.TemporaryDirectory(prefix="experimentpilot-diagnostic-", dir=staging_root) as temp_dir:
                local_dir = Path(temp_dir)
                request_path = local_dir / "request.json"
                request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
                mkdir = self._ssh(f"mkdir -p -- {shlex.quote(remote_run)}", timeout=30)
                if mkdir.returncode != 0:
                    raise RuntimeError(mkdir.stderr.strip() or mkdir.stdout.strip() or "remote mkdir failed")
                for source, destination in ((request_path, "request.json"), (helper, "remote_diagnostic.py")):
                    uploaded, _ = self._retry_transfer(
                        lambda *, timeout: self._scp_to(
                            source, self._remote_join(remote_run, destination), timeout=timeout
                        )
                    )
                    if uploaded.returncode != 0:
                        raise RuntimeError(uploaded.stderr.strip() or uploaded.stdout.strip() or f"upload failed: {source.name}")
                command = " ".join([
                    f"cd {shlex.quote(self.config.remote_repo)} &&",
                    f"CUDA_VISIBLE_DEVICES={physical_gpu}",
                    f"PYTHONPATH={shlex.quote(self.config.remote_repo)}",
                    "timeout", "--signal=TERM", str(timeout_seconds),
                    tool_path, "--tool", shlex.quote(tool), "--error-exitcode=42",
                    shlex.quote(self.config.remote_python),
                    shlex.quote(self._remote_join(remote_run, "remote_diagnostic.py")),
                    "--request", shlex.quote(self._remote_join(remote_run, "request.json")),
                    ">", shlex.quote(self._remote_join(remote_run, "stdout.log")),
                    "2>", shlex.quote(self._remote_join(remote_run, "stderr.log")),
                ])
                result["command"] = command
                completed = self._ssh(command, timeout=timeout_seconds + 20)
                result["exit_code"] = completed.returncode
                for local_name in ("stdout.log", "stderr.log"):
                    local_path = local_dir / local_name
                    downloaded, _ = self._retry_transfer(
                        lambda *, timeout: self._scp_from(
                            self._remote_join(remote_run, local_name), local_path, timeout=timeout
                        )
                    )
                    if downloaded.returncode == 0 and local_path.is_file():
                        result[local_name.removesuffix(".log")] = local_path.read_text(encoding="utf-8", errors="replace")
                    elif downloaded.returncode != 0:
                        result["stderr"] += (
                            "\nfailed to download " + local_name + ": "
                            + (downloaded.stderr.strip() or downloaded.stdout.strip())
                        )
                diagnostic_text = result["stdout"] + "\n" + result["stderr"]
                result["sanitizer_error"] = (
                    completed.returncode == 42
                    or any(
                        int(count) > 0
                        for count in re.findall(r"ERROR SUMMARY:\s*(\d+) errors", diagnostic_text)
                    )
                )
                if completed.returncode == 0:
                    result["status"] = "PASSED"
                elif completed.returncode == 124:
                    result["status"] = "TIMEOUT"
                elif result["sanitizer_error"]:
                    result["status"] = "SANITIZER_ERROR"
                else:
                    result["status"] = "CANDIDATE_FAILURE"
        except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
            result.update({"status": "TIMEOUT" if isinstance(exc, subprocess.TimeoutExpired) else "INFRA_FAILURE", "stderr": f"{type(exc).__name__}: {exc}"})
        finally:
            result["duration_seconds"] = round(time.monotonic() - started, 6)
        return result

    def run_profiler(
        self,
        candidate_blob: str,
        task_id: str,
        *,
        artifact_dir: Path,
        timeout_seconds: int = 300,
    ) -> dict[str, Any]:
        """Collect an on-demand Nsight Systems CUDA timeline outside benchmarking.

        The report and summaries are diagnostic evidence only.  This method is
        deliberately separate from ``evaluate_blob`` so it cannot affect the
        KernelBench correctness, latency, or speedup contract.
        """
        if timeout_seconds < 1:
            raise ValueError("Profiler timeout must be positive")
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        report_path = artifact_dir / "timeline.nsys-rep"
        stats_path = artifact_dir / "stats.csv"
        stats_stderr_path = artifact_dir / "stats.stderr.log"
        started = time.monotonic()
        result: dict[str, Any] = {
            "protocol_version": 1,
            "tool": "nsight-systems",
            "mode": "cuda_timeline",
            "status": "ERROR",
            "command": "",
            "stats_command": "",
            "exit_code": None,
            "stats_exit_code": None,
            "stdout": "",
            "stderr": "",
            "task_id": task_id,
            "report_path": str(report_path),
            "stats_path": str(stats_path),
            "diagnostic_only": True,
            "formal_correctness_verdict": "NOT_RUN",
            "formal_performance_verdict": "NOT_RUN",
        }
        try:
            physical_gpu = self._select_gpu()
            remote_run = self._remote_join(
                self.config.remote_runs_dir, f"ep-profile-{int(time.time())}-{uuid.uuid4().hex[:12]}"
            )
            result.update({
                "physical_gpu_id": physical_gpu, "selected_gpu": physical_gpu,
                "visible_device": "cuda:0", "device": "cuda:0", "remote_workspace": remote_run,
            })
            nsys = shlex.quote(self.config.remote_nsys)
            preflight = self._ssh(
                f"test -x {nsys} || {{ echo missing remote_nsys; exit 24; }}; "
                f"test -d {shlex.quote(self.config.remote_kernelbench)} || "
                "{ echo missing KernelBench checkout; exit 21; }",
                timeout=min(30, timeout_seconds),
            )
            if preflight.returncode != 0:
                result.update({
                    "status": "UNAVAILABLE",
                    "exit_code": preflight.returncode,
                    "stderr": preflight.stderr or preflight.stdout,
                })
                return result

            request = {
                "candidate_blob": candidate_blob,
                "task_id": task_id,
                "kernelbench_path": self.config.remote_kernelbench,
            }
            helper = Path(__file__).resolve().parent.parent / "scripts" / "remote_diagnostic.py"
            staging_root = Path.cwd() if self.config.scp_binary.lower().endswith(".exe") else None
            with tempfile.TemporaryDirectory(prefix="experimentpilot-profile-", dir=staging_root) as temp_dir:
                local_dir = Path(temp_dir)
                request_path = local_dir / "request.json"
                request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
                mkdir = self._ssh(f"mkdir -p -- {shlex.quote(remote_run)}", timeout=30)
                if mkdir.returncode != 0:
                    raise RuntimeError(mkdir.stderr.strip() or mkdir.stdout.strip() or "remote mkdir failed")
                for source, destination in ((request_path, "request.json"), (helper, "remote_diagnostic.py")):
                    uploaded, _ = self._retry_transfer(
                        lambda *, timeout: self._scp_to(
                            source, self._remote_join(remote_run, destination), timeout=timeout
                        )
                    )
                    if uploaded.returncode != 0:
                        raise RuntimeError(uploaded.stderr.strip() or uploaded.stdout.strip() or f"upload failed: {source.name}")
                remote_report_prefix = self._remote_join(remote_run, "timeline")
                command = " ".join([
                    f"cd {shlex.quote(self.config.remote_repo)} &&",
                    f"CUDA_VISIBLE_DEVICES={physical_gpu}",
                    f"PYTHONPATH={shlex.quote(self.config.remote_repo)}",
                    "timeout", "--signal=TERM", str(timeout_seconds),
                    nsys, "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
                    "--force-overwrite=true", f"--output={shlex.quote(remote_report_prefix)}",
                    shlex.quote(self.config.remote_python),
                    shlex.quote(self._remote_join(remote_run, "remote_diagnostic.py")),
                    "--request", shlex.quote(self._remote_join(remote_run, "request.json")),
                    ">", shlex.quote(self._remote_join(remote_run, "profile.stdout.log")),
                    "2>", shlex.quote(self._remote_join(remote_run, "profile.stderr.log")),
                ])
                result["command"] = command
                completed = self._ssh(command, timeout=timeout_seconds + 30)
                result["exit_code"] = completed.returncode

                for remote_name, local_path, field in (
                    ("profile.stdout.log", local_dir / "profile.stdout.log", "stdout"),
                    ("profile.stderr.log", local_dir / "profile.stderr.log", "stderr"),
                ):
                    downloaded, _ = self._retry_transfer(
                        lambda *, timeout: self._scp_from(
                            self._remote_join(remote_run, remote_name), local_path, timeout=timeout
                        )
                    )
                    if downloaded.returncode == 0 and local_path.is_file():
                        result[field] = local_path.read_text(encoding="utf-8", errors="replace")
                if completed.returncode != 0:
                    result["status"] = "ERROR"
                    return result

                downloaded, _ = self._retry_transfer(
                    lambda *, timeout: self._scp_from(
                        self._remote_join(remote_run, "timeline.nsys-rep"), report_path, timeout=timeout
                    )
                )
                if downloaded.returncode != 0 or not report_path.is_file():
                    raise RuntimeError(downloaded.stderr.strip() or downloaded.stdout.strip() or "profile report was not produced")

                stats_command = " ".join([
                    f"cd {shlex.quote(self.config.remote_repo)} &&",
                    nsys, "stats", "--format", "csv",
                    "--report", "cuda_gpu_kern_sum", "--report", "cuda_api_sum",
                    "--report", "cuda_gpu_mem_time_sum",
                    shlex.quote(self._remote_join(remote_run, "timeline.nsys-rep")),
                    ">", shlex.quote(self._remote_join(remote_run, "stats.csv")),
                    "2>", shlex.quote(self._remote_join(remote_run, "stats.stderr.log")),
                ])
                result["stats_command"] = stats_command
                stats_completed = self._ssh(stats_command, timeout=timeout_seconds + 30)
                result["stats_exit_code"] = stats_completed.returncode
                for remote_name, local_path in (
                    ("stats.csv", stats_path),
                    ("stats.stderr.log", stats_stderr_path),
                ):
                    downloaded, _ = self._retry_transfer(
                        lambda *, timeout: self._scp_from(
                            self._remote_join(remote_run, remote_name), local_path, timeout=timeout
                        )
                    )
                    if downloaded.returncode != 0 and remote_name == "stats.csv":
                        raise RuntimeError(downloaded.stderr.strip() or downloaded.stdout.strip() or "profile stats were not produced")
                result["status"] = "PASS" if stats_completed.returncode == 0 else "ERROR"
        except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
            result.update({"status": "TIMEOUT" if isinstance(exc, subprocess.TimeoutExpired) else "ERROR", "stderr": f"{type(exc).__name__}: {exc}"})
        finally:
            result["duration_seconds"] = round(time.monotonic() - started, 6)
        return result

    @staticmethod
    def _remote_join(root: str, leaf: str) -> str:
        return root.rstrip("/") + "/" + leaf

    @staticmethod
    def _add_metadata(results: dict, metadata: Mapping[str, Any]) -> None:
        for traces in results.values():
            for item in traces.values():
                item_metadata = item.setdefault("metadata", {})
                if isinstance(item_metadata, dict):
                    item_metadata.update(metadata)

    def _failure(
        self, candidate_blob: str, task_ids: Sequence[str], status: ExecutionStatus, message: str,
        metadata: Mapping[str, Any],
    ) -> BenchmarkExecutionResult:
        results = failure_results(candidate_blob, task_ids, status=status, error_log=message, metadata=metadata)
        return BenchmarkExecutionResult(status, None, error_log=message, metadata=dict(metadata), results=results)

    def evaluate_blob(
        self, candidate_blob: str, task_ids: Sequence[str], config: Mapping[str, Any], *, capture_logs: bool = False,
    ) -> BenchmarkExecutionResult:
        # The guard is a no-op outside a Phase 2 branch.  Keeping it at this
        # evaluator-facing seam also prevents direct ``run_remote.py`` or
        # Python backend use from silently bypassing scripts/bench.sh.
        from .budget import ensure_evaluation_authorized
        ensure_evaluation_authorized()
        run_id = f"ep-{int(time.time())}-{uuid.uuid4().hex[:12]}"
        remote_run = self._remote_join(self.config.remote_runs_dir, run_id)
        metadata: dict[str, Any] = {
            "backend": "ssh", "remote_host": self.config.remote_host,
            "remote_workspace": remote_run, "run_id": run_id,
        }
        started = time.monotonic()
        try:
            physical_gpu = self._select_gpu()
            metadata.update({"physical_gpu_id": physical_gpu, "visible_device": "cuda:0"})
            metadata.update(self._preflight(physical_gpu))
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            metadata["duration_seconds"] = round(time.monotonic() - started, 6)
            return self._failure(candidate_blob, task_ids, ExecutionStatus.INFRA_FAILURE,
                                 f"Remote preflight/GPU selection failed: {type(exc).__name__}: {exc}", metadata)

        request = {
            "candidate_blob": candidate_blob, "task_ids": list(task_ids), "params": dict(config),
            "capture_logs": capture_logs, "kernelbench_path": self.config.remote_kernelbench,
            "metadata": metadata,
        }
        helper = Path(__file__).resolve().parent.parent / "scripts" / "remote_evaluator.py"
        # Keep transfer staging under the workspace when a WSL process invokes
        # Windows OpenSSH: ``/tmp`` is not a Windows-accessible local path.
        staging_root = Path.cwd() if self.config.scp_binary.lower().endswith(".exe") else None
        with tempfile.TemporaryDirectory(prefix="experimentpilot-remote-", dir=staging_root) as temp_dir:
            local_dir = Path(temp_dir)
            request_path = local_dir / "request.json"
            request_path.write_text(json.dumps(request, indent=2))
            result_path = local_dir / "result.json"
            try:
                mkdir = self._ssh(f"mkdir -p -- {shlex.quote(remote_run)}", timeout=30)
                if mkdir.returncode != 0:
                    raise RuntimeError(mkdir.stderr.strip() or mkdir.stdout.strip() or "remote mkdir failed")
                for source, destination in ((request_path, "request.json"), (helper, "remote_evaluator.py")):
                    sent, transfer_attempt = self._retry_transfer(
                        lambda *, timeout: self._scp_to(source, self._remote_join(remote_run, destination), timeout=timeout)
                    )
                    metadata[f"upload_{destination}_attempts"] = transfer_attempt
                    if sent.returncode != 0:
                        raise RuntimeError(sent.stderr.strip() or sent.stdout.strip() or f"upload failed: {source.name}")
                command = " ".join([
                    f"cd {shlex.quote(self.config.remote_repo)} &&",
                    f"CUDA_VISIBLE_DEVICES={physical_gpu}",
                    f"PYTHONPATH={shlex.quote(self.config.remote_repo)}",
                    "timeout", "--signal=TERM", str(self.config.timeout_seconds),
                    shlex.quote(self.config.remote_python),
                    shlex.quote(self._remote_join(remote_run, "remote_evaluator.py")),
                    "--request", shlex.quote(self._remote_join(remote_run, "request.json")),
                    "--result", shlex.quote(self._remote_join(remote_run, "result.json")),
                    ">", shlex.quote(self._remote_join(remote_run, "stdout.log")),
                    "2>", shlex.quote(self._remote_join(remote_run, "stderr.log")),
                ])
                completed = self._ssh(command, timeout=self.config.timeout_seconds + 20)
                if completed.returncode == 124:
                    raise TimeoutError(f"Remote evaluator exceeded {self.config.timeout_seconds}s")
                fetched, transfer_attempt = self._retry_transfer(
                    lambda *, timeout: self._scp_from(self._remote_join(remote_run, "result.json"), result_path, timeout=timeout)
                )
                metadata["download_result.json_attempts"] = transfer_attempt
                if fetched.returncode != 0:
                    detail = fetched.stderr.strip() or completed.stderr.strip() or "result.json was not produced"
                    raise RuntimeError(detail)
                envelope = json.loads(result_path.read_text())
                if envelope.get("protocol_version") != 1 or not isinstance(envelope.get("results"), dict):
                    raise ValueError("Malformed remote evaluator JSON envelope")
                remote_metadata = envelope.get("metadata") if isinstance(envelope.get("metadata"), dict) else {}
                metadata.update(remote_metadata)
                metadata["duration_seconds"] = round(time.monotonic() - started, 6)
                results = envelope["results"]
                self._add_metadata(results, metadata)
                if envelope.get("status") != ExecutionStatus.SUCCESS.value:
                    message = str(envelope.get("error_log") or "Remote evaluator reported infrastructure failure")
                    return BenchmarkExecutionResult(ExecutionStatus.INFRA_FAILURE, None, error_log=message,
                                                    metadata=metadata, results=results)
                return BenchmarkExecutionResult.from_results(results, metadata=metadata)
            except (OSError, subprocess.TimeoutExpired, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                status = ExecutionStatus.TIMEOUT if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else ExecutionStatus.INFRA_FAILURE
                metadata["duration_seconds"] = round(time.monotonic() - started, 6)
                return self._failure(candidate_blob, task_ids, status,
                                     f"Remote evaluation failed: {type(exc).__name__}: {exc}", metadata)

    def cleanup(self, run_id: str) -> None:
        """Delete exactly one prior run directory; callers choose retention policy."""
        if not run_id.startswith("ep-") or "/" in run_id or "\\" in run_id:
            raise ValueError("Invalid ExperimentPilot run ID")
        path = self._remote_join(self.config.remote_runs_dir, run_id)
        completed = self._ssh(f"rm -rf -- {shlex.quote(path)}", timeout=30)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "remote cleanup failed")

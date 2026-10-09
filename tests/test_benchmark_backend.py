import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from benchmark_backend import (
    ExecutionStatus,
    LocalBenchmarkBackend,
    RemoteSSHBenchmarkBackend,
    RemoteSSHConfig,
)


def blob():
    return json.dumps({
        "format": "ako4x-kernelbench-v1", "name": "candidate", "definition": "square",
        "author": "test", "build": {"language": "triton", "entry_point": "kernel.py::run"},
        "sources": {"kernel.py": "class ModelNew: pass"},
    })


def remote_config(**overrides):
    values = {
        "remote_host": "lab-gpu", "remote_repo": "/srv/AKO4X",
        "remote_python": "/srv/python", "remote_runs_dir": "/srv/runs",
        "remote_kernelbench": "/srv/KernelBench", "preferred_gpu": 1,
        "evaluation_timeout": 30,
    }
    values.update(overrides)
    return values


class BackendContractTest(unittest.TestCase):
    def test_local_backend_preserves_adapter_result_shape(self):
        expected = {"square": {"level1/demo": {"status": "PASSED", "latency_ms": 2.0,
                                            "reference_latency_ms": 4.0, "speedup_factor": 2.0}}}
        with patch("scripts.benchmark_adapter.run", return_value=expected):
            result = LocalBenchmarkBackend("/dataset").evaluate_blob(blob(), ["level1/demo"], {})
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertTrue(result.correct)
        self.assertEqual(result.results, expected)

    def test_gpu_four_is_rejected_at_config_time(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            RemoteSSHConfig.from_mapping(remote_config(preferred_gpu=4))

    def test_remote_compute_sanitizer_path_is_configurable(self):
        default = RemoteSSHConfig.from_mapping(remote_config())
        self.assertEqual(default.remote_compute_sanitizer, "")
        configured = RemoteSSHConfig.from_mapping(
            remote_config(remote_compute_sanitizer="/srv/tools/compute-sanitizer")
        )
        self.assertEqual(configured.remote_compute_sanitizer, "/srv/tools/compute-sanitizer")

    def test_remote_nsys_path_is_configurable(self):
        configured = RemoteSSHConfig.from_mapping(remote_config(remote_nsys="/srv/tools/nsys"))
        self.assertEqual(configured.remote_nsys, "/srv/tools/nsys")


class RemoteBackendTest(unittest.TestCase):
    def setUp(self):
        self.backend = RemoteSSHBenchmarkBackend(remote_config())

    @staticmethod
    def _ok() -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess([], 0, "", "")

    def _install_success_transport(self):
        def ssh(command, *, timeout):
            if command.startswith("nvidia-smi"):
                return subprocess.CompletedProcess([], 0, "0, 0, 0\n1, 0, 0\n4, 0, 0\n", "")
            return self._ok()

        def scp_to(local, remote, *, timeout):
            self.assertIn("/srv/runs/ep-", remote)
            return self._ok()

        def scp_from(remote, local, *, timeout):
            local.write_text(json.dumps({
                "protocol_version": 1, "status": "SUCCESS", "error_log": "",
                "metadata": {"remote_evaluator": "fixed-v1"},
                "results": {"square": {"level1/demo": {
                    "status": "PASSED", "latency_ms": 2.5,
                    "reference_latency_ms": 2.0, "speedup_factor": 0.8,
                }}},
            }))
            return self._ok()

        self.backend._ssh = ssh
        self.backend._scp_to = scp_to
        self.backend._scp_from = scp_from

    def test_round_trip_records_physical_and_visible_gpu(self):
        self._install_success_transport()
        result = self.backend.evaluate_blob(blob(), ["level1/demo"], {"device": "cuda:0"})
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertEqual(result.metadata["physical_gpu_id"], 1)
        self.assertEqual(result.metadata["visible_device"], "cuda:0")
        item = result.results["square"]["level1/demo"]
        self.assertEqual(item["metadata"]["physical_gpu_id"], 1)

    def test_windows_scp_path_is_translated_only_for_wsl_mounts(self):
        backend = RemoteSSHBenchmarkBackend(remote_config(scp_binary="scp.exe"))
        self.assertEqual(
            backend._scp_local_path(Path("/mnt/d/mycode/AKO4X/request.json")),
            r"D:\mycode\AKO4X\request.json",
        )
        self.assertEqual(backend._scp_local_path(Path("/tmp/request.json")), "/tmp/request.json")

    def test_control_path_is_applied_to_ssh_and_scp_transport(self):
        backend = RemoteSSHBenchmarkBackend(remote_config(ssh_control_path="/tmp/child/.ako/lab-gpu.sock"))
        args = backend._ssh_config_args()
        self.assertEqual(
            args[-4:],
            ["-o", "ControlMaster=no", "-o", "ControlPath=/tmp/child/.ako/lab-gpu.sock"],
        )

    def test_gpu_query_failure_is_normalized_as_infra_failure(self):
        self.backend._ssh = lambda command, *, timeout: subprocess.CompletedProcess([], 255, "", "connection refused")
        result = self.backend.evaluate_blob(blob(), ["level1/demo"], {})
        self.assertEqual(result.status, ExecutionStatus.INFRA_FAILURE)
        self.assertEqual(result.results["square"]["level1/demo"]["status"], "INFRA_FAILURE")

    def test_preflight_retries_one_transient_environment_witness(self):
        preflight_calls = []

        def ssh(command, *, timeout):
            preflight_calls.append(command)
            if len(preflight_calls) == 1:
                return subprocess.CompletedProcess([], 20, "missing remote_python\n", "")
            return self._ok()

        self.backend._ssh = ssh
        with patch("benchmark_backend.ssh_remote.time.sleep") as sleep:
            result = self.backend._preflight(1)
        self.assertEqual(result["preflight_attempts"], 2)
        self.assertEqual(len(preflight_calls), 2)
        sleep.assert_called_once_with(2)

    def test_remote_timeout_is_normalized_without_raising(self):
        def ssh(command, *, timeout):
            if command.startswith("nvidia-smi"):
                return subprocess.CompletedProcess([], 0, "1, 0, 0\n", "")
            if command.startswith("mkdir"):
                return self._ok()
            if "timeout --signal=TERM" in command:
                return subprocess.CompletedProcess([], 124, "", "")
            return self._ok()

        self.backend._ssh = ssh
        self.backend._scp_to = lambda *args, **kwargs: self._ok()
        result = self.backend.evaluate_blob(blob(), ["level1/demo"], {})
        self.assertEqual(result.status, ExecutionStatus.TIMEOUT)
        self.assertEqual(result.results["square"]["level1/demo"]["status"], "TIMEOUT")

    def test_malformed_result_json_is_normalized_without_raising(self):
        self._install_success_transport()
        def malformed_scp(remote, local, *, timeout):
            local.write_text("not-json")
            return self._ok()
        self.backend._scp_from = malformed_scp
        result = self.backend.evaluate_blob(blob(), ["level1/demo"], {})
        self.assertEqual(result.status, ExecutionStatus.INFRA_FAILURE)
        self.assertIn("JSONDecodeError", result.error_log)

    def test_transfer_timeout_retries_once(self):
        calls = []

        def operation(*, timeout):
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(["scp"], timeout)
            return self._ok()

        result, attempts = self.backend._retry_transfer(operation)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(attempts, 2)
        self.assertEqual(calls, [90, 90])

    def test_remote_sanitizer_uses_configured_tool_and_stays_outside_benchmark(self):
        backend = RemoteSSHBenchmarkBackend(
            remote_config(remote_compute_sanitizer="/srv/tools/compute-sanitizer")
        )
        commands = []
        backend._select_gpu = lambda: 1

        def ssh(command, *, timeout):
            commands.append(command)
            return self._ok()

        def scp_to(local, remote, *, timeout):
            self.assertIn("ep-diagnostic-", remote)
            return self._ok()

        def scp_from(remote, local, *, timeout):
            if remote.endswith("stdout.log"):
                local.write_text("DIAGNOSTIC_CANDIDATE_WORKLOAD_COMPLETED\n")
            else:
                local.write_text("========= ERROR SUMMARY: 0 errors\n")
            return self._ok()

        backend._ssh = ssh
        backend._scp_to = scp_to
        backend._scp_from = scp_from
        result = backend.run_sanitizer(blob(), "level1/demo", timeout_seconds=30)
        self.assertEqual(result["status"], "PASSED")
        self.assertFalse(result["sanitizer_error"])
        self.assertEqual(result["formal_correctness_verdict"], "NOT_RUN")
        command = result["command"]
        self.assertIn("/srv/tools/compute-sanitizer --tool memcheck --error-exitcode=42", command)
        self.assertIn("/srv/runs/ep-diagnostic-", command)
        self.assertFalse(any("remote_evaluator.py" in item for item in commands))

    def test_remote_sanitizer_reports_tool_errors_without_a_benchmark_verdict(self):
        backend = RemoteSSHBenchmarkBackend(
            remote_config(remote_compute_sanitizer="/srv/tools/compute-sanitizer")
        )
        backend._select_gpu = lambda: 1

        def ssh(command, *, timeout):
            if "compute-sanitizer" in command and "--error-exitcode=42" in command:
                return subprocess.CompletedProcess([], 42, "", "")
            return self._ok()

        def scp_from(remote, local, *, timeout):
            local.write_text("========= ERROR SUMMARY: 2 errors\n")
            return self._ok()

        backend._ssh = ssh
        backend._scp_to = lambda *args, **kwargs: self._ok()
        backend._scp_from = scp_from
        result = backend.run_sanitizer(blob(), "level1/demo", timeout_seconds=30)
        self.assertEqual(result["status"], "SANITIZER_ERROR")
        self.assertTrue(result["sanitizer_error"])
        self.assertEqual(result["formal_correctness_verdict"], "NOT_RUN")

    def test_remote_profiler_uses_lightweight_nsys_and_writes_diagnostic_artifacts(self):
        backend = RemoteSSHBenchmarkBackend(remote_config(remote_nsys="/srv/tools/nsys"))
        commands = []
        backend._select_gpu = lambda: 1

        def ssh(command, *, timeout):
            commands.append(command)
            return self._ok()

        def scp_from(remote, local, *, timeout):
            if remote.endswith("timeline.nsys-rep"):
                local.write_bytes(b"nsys report")
            elif remote.endswith("stats.csv"):
                local.write_text(
                    "Time (%),Total Time (ns),Instances,Avg (ns),Name\n"
                    "100.0,1000,1,1000.0,kernel\n"
                )
            else:
                local.write_text("")
            return self._ok()

        backend._ssh = ssh
        backend._scp_to = lambda *args, **kwargs: self._ok()
        backend._scp_from = scp_from
        with tempfile.TemporaryDirectory() as temporary:
            artifact_dir = Path(temporary) / "profile"
            result = backend.run_profiler(blob(), "level1/demo", artifact_dir=artifact_dir, timeout_seconds=30)
            self.assertTrue((artifact_dir / "timeline.nsys-rep").is_file())
            self.assertTrue((artifact_dir / "stats.csv").is_file())
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(result["formal_correctness_verdict"], "NOT_RUN")
        self.assertEqual(result["formal_performance_verdict"], "NOT_RUN")
        self.assertEqual((result["selected_gpu"], result["device"]), (1, "cuda:0"))
        self.assertEqual(result["report_path"], str(artifact_dir / "timeline.nsys-rep"))
        self.assertEqual(result["stats_path"], str(artifact_dir / "stats.csv"))
        self.assertIn("/srv/tools/nsys profile --trace=cuda,nvtx,osrt --sample=none", result["command"])
        self.assertNotIn("--gpu-metrics-device", result["command"])
        self.assertNotIn("--gpu-metrics-devices", result["command"])
        self.assertTrue(any("cuda_gpu_kern_sum" in command for command in commands))

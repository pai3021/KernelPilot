from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from spawn import load_remote_config, populate_child
from scripts.run_remote_profile import _parse_nsys_stats


class RemoteDiagnosticSpawnTest(unittest.TestCase):
    def test_ssh_child_uses_explicit_remote_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definition = root / "definition.json"
            workloads = root / "workloads.jsonl"
            definition.write_text(json.dumps({
                "name": "demo", "reference": "class Model:\n    pass\n", "axes": {},
            }), encoding="utf-8")
            workloads.write_text(json.dumps({
                "workload": {"uuid": "level1/demo", "axes": {"level": "level1"}},
            }) + "\n", encoding="utf-8")
            child = root / "child"
            populate_child(
                child, operator="demo", op_type="level1", gpu="cuda:0", backend="ssh",
                kernel_path=None, definition_path=definition, workloads_path=workloads,
                dataset_path=root,
                remote_config={
                    "host": "example-gpu", "remote_repo": "/srv/kernelpilot",
                    "python": "/srv/venv/bin/python", "kernelbench": "/srv/KernelBench",
                    "runs_dir": "/srv/kernelpilot/runs",
                    "compute_sanitizer": "/srv/tools/compute-sanitizer",
                    "nsys": "/srv/tools/nsys",
                },
            )

            profile = (child / "scripts" / "profile.sh").read_text(encoding="utf-8")
            sanitize = (child / "scripts" / "sanitize.sh").read_text(encoding="utf-8")
            config = (child / "config.toml").read_text(encoding="utf-8", errors="replace")
            self.assertIn("python3 scripts/run_remote_profile.py", profile)
            self.assertIn("python3 scripts/run_remote_sanitize.py", sanitize)
            self.assertNotIn("exit 2", profile)
            self.assertNotIn("exit 2", sanitize)
            self.assertNotIn(b"\r\n", (child / "scripts" / "sanitize.sh").read_bytes())
            self.assertTrue((child / "scripts" / "run_remote_profile.py").is_file())
            self.assertTrue((child / "scripts" / "run_remote_sanitize.py").is_file())
            self.assertIn(
                'compute_sanitizer = "/srv/tools/compute-sanitizer"',
                config,
            )
            self.assertIn('nsys = "/srv/tools/nsys"', config)
            self.assertIn('host = "example-gpu"', config)
            self.assertNotIn("lab-gpu", config)
            parsed = tomllib.loads(config)
            self.assertEqual(parsed["benchmark"]["remote"]["host"], "example-gpu")

    def test_remote_config_requires_complete_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "remote.toml"
            path.write_text('[benchmark.remote]\nhost = "example-gpu"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "remote_repo"):
                load_remote_config(path)
            path.write_text(
                '[benchmark.remote]\nhost = "example-gpu"\n'
                'remote_repo = "/srv/kernelpilot"\npython = "/srv/python"\n'
                'kernelbench = "/srv/KernelBench"\nruns_dir = "/srv/runs"\n',
                encoding="utf-8",
            )
            self.assertEqual(load_remote_config(path)["host"], "example-gpu")

    def test_shipped_remote_example_has_required_fields(self):
        example = Path(__file__).resolve().parents[1] / "configs" / "remote.example.toml"
        self.assertEqual(load_remote_config(example)["host"], "gpu-host")

    def test_nsys_csv_summary_keeps_raw_unknowns_out_of_invented_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            stats = Path(temporary) / "stats.csv"
            stats.write_text(
                "Time (%),Total Time (ns),Instances,Avg (ns),Name\n"
                "80.0,8000,4,2000.0,kernel_a\n"
                "20.0,2000,2,1000.0,kernel_b\n\n"
                "Time (%),Total Time (ns),Num Calls,Avg (ns),Name\n"
                "100.0,3000,3,1000.0,cudaLaunchKernel\n",
                encoding="utf-8",
            )
            summary = _parse_nsys_stats(stats)
        self.assertEqual(summary["kernel_launch_count"], 6)
        self.assertEqual(summary["top_cuda_kernels"][0]["name"], "kernel_a")
        self.assertEqual(summary["top_cuda_api_calls"][0]["calls"], 3)
        self.assertIsNone(summary["cuda_memcpy_count"])
        self.assertEqual(summary["parse_status"]["kernels"], "parsed")

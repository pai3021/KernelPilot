import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import benchmark_adapter as adapter


TASK_SOURCE = """
class Model:
    pass
def get_inputs():
    return []
def get_init_inputs():
    return []
"""


class FakeDevice:
    type = "cuda"
    index = 0
    def __str__(self):
        return "cuda:0"


class FakeResult:
    def __init__(self, compiled=True, correctness=True, runtime=2.0, ref_runtime=4.0, metadata=None):
        self.compiled = compiled
        self.correctness = correctness
        self.runtime = runtime
        self.ref_runtime = ref_runtime
        self.metadata = metadata or {}


class AdapterContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.task_root = root / "KernelBench" / "level1"
        self.task_root.mkdir(parents=True)
        (self.task_root / "demo.py").write_text(TASK_SOURCE)
        source = root / "solution"
        source.mkdir()
        (source / "kernel.py").write_text("class ModelNew:\n    pass\n")
        self.blob = adapter.pack(
            source, {"language": "python", "entry_point": "kernel.py::run"},
            name="candidate", definition="demo", author="test",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _run_with(self, result):
        torch = types.SimpleNamespace(
            float32=object(), device=lambda raw: FakeDevice(),
        )
        eval_module = types.ModuleType("kernelbench.eval")
        eval_module.eval_kernel_against_ref = lambda *args, **kwargs: result
        package = types.ModuleType("kernelbench")
        package.eval = eval_module
        with patch.dict(sys.modules, {"torch": torch, "kernelbench": package,
                                      "kernelbench.eval": eval_module}):
            return adapter.run(
                self.blob, ["level1/demo"],
                {"correct_trials": 3, "perf_trials": 20, "device": "cuda:0"},
                dataset_path=self.tmp.name,
            )

    def test_pack_and_meta_are_plain_data(self):
        self.assertEqual(adapter.solution_meta(self.blob)["definition"], "demo")
        self.assertEqual(json.loads(self.blob)["format"], "ako4x-kernelbench-v1")

    def test_pass_normalizes_correctness_latency_and_speedup(self):
        item = self._run_with(FakeResult())["demo"]["level1/demo"]
        self.assertEqual(item["status"], adapter.STATUS_PASSED)
        self.assertEqual(item["latency_ms"], 2.0)
        self.assertEqual(item["reference_latency_ms"], 4.0)
        self.assertEqual(item["speedup_factor"], 2.0)
        self.assertEqual(item["num_correct_trials"], 3)
        self.assertEqual(item["num_perf_trials"], 20)

    def test_compile_failure_is_not_scored(self):
        item = self._run_with(FakeResult(compiled=False))["demo"]["level1/demo"]
        self.assertEqual(item["status"], adapter.STATUS_COMPILE_ERROR)
        self.assertNotIn("latency_ms", item)

    def test_correctness_failure_is_not_scored(self):
        item = self._run_with(FakeResult(correctness=False))["demo"]["level1/demo"]
        self.assertEqual(item["status"], adapter.STATUS_INCORRECT_NUMERICAL)
        self.assertNotIn("speedup_factor", item)

    def test_runtime_failure_is_not_scored(self):
        item = self._run_with(FakeResult(correctness=False, metadata={"runtime_error": "boom"}))["demo"]["level1/demo"]
        self.assertEqual(item["status"], adapter.STATUS_RUNTIME_ERROR)
        self.assertNotIn("speedup_factor", item)

    def test_unknown_task_is_never_silently_skipped(self):
        with self.assertRaises(FileNotFoundError):
            adapter.run(self.blob, ["level1/missing"], {}, dataset_path=self.tmp.name)


if __name__ == "__main__":
    unittest.main()

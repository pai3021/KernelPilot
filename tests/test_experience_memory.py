from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from agent_runtime import TaskContract
from benchmark_backend.provenance import capture_evaluated_candidate
from campaign.controller import TwoBranchController
from campaign.models import BranchSpec
from experience_memory import (
    ExperienceContextBuilder,
    ExperienceRecord,
    ExperienceRetriever,
    ExperienceStore,
    TaskSignature,
    extract_round,
)
from campaign.models import BranchResult, RoundResult, RoundState


SOURCE = TaskSignature(
    "KernelBench", "source-square", "level1", "matrix_multiplication", "gemm",
    "A[N,N],B[N,N]", "square", "float32", "RTX4090", "cuda_event_correctness_latency",
)
TARGET = TaskSignature(
    "KernelBench", "target-rectangular", "level1", "matrix_multiplication", "gemm",
    "A[M,K],B[K,N]", "rectangular", "float32", "RTX4090", "cuda_event_correctness_latency",
)


def record(experience_id: str, outcome: str, *, family: str = "native_dispatch") -> ExperienceRecord:
    return ExperienceRecord(
        experience_id, SOURCE, "campaign", "round", experience_id, "parent",
        f"hypothesis {experience_id}", family, "small, bounded implementation attempt",
        {"status": "PASSED", "speedup_factor": 1.02}, outcome,
        None if outcome in {"PROMOTED", "VALID_NOT_PROMOTED"} else "COMPILE_ERROR",
        None, f"lesson {experience_id}", hashlib.sha256(experience_id.encode()).hexdigest(),
        {"evaluated_candidate_snapshot": f"/artifact/{experience_id}.py"},
        datetime.now(timezone.utc).isoformat(),
    )


class ExperienceMemoryTest(unittest.TestCase):
    def test_record_round_trip_and_append_get_dedupe(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperienceStore(Path(directory) / "experiences.jsonl")
            item = record("EXP-1", "PROMOTED")
            self.assertEqual(ExperienceRecord.from_dict(item.to_dict()), item)
            self.assertTrue(store.append(item))
            self.assertFalse(store.append(item))
            self.assertEqual(store.get("EXP-1"), item)
            self.assertEqual(store.query(lambda value: value.outcome == "PROMOTED"), [item])

    def test_metadata_ranking_is_deterministic_and_mixes_success_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperienceStore(Path(directory) / "experiences.jsonl")
            for item in (record("EXP-z", "PROMOTED"), record("EXP-a", "VALID_NOT_PROMOTED"), record("EXP-f", "COMPILE_FAILURE", family="custom_triton")):
                store.append(item)
            hits = ExperienceRetriever(store).query(TARGET)
            self.assertEqual([hit.record.experience_id for hit in hits], ["EXP-a", "EXP-z", "EXP-f"])
            self.assertTrue(all(hit.score == 9 for hit in hits))
            self.assertIn("COMPILE_FAILURE", [hit.record.outcome for hit in hits])

    def test_infra_failure_is_not_returned_as_a_strategy_lesson(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperienceStore(Path(directory) / "experiences.jsonl")
            store.append(record("EXP-infra", "INFRA_FAILURE"))
            self.assertEqual(ExperienceRetriever(store).query(TARGET), [])

    def test_cross_operator_records_are_excluded_before_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperienceStore(Path(directory) / "experiences.jsonl")
            store.append(record("EXP-gemm", "PROMOTED"))
            unrelated = TaskSignature(
                "KernelBench", "relu", "level1", "activation", "relu",
                "x", "large", "float32", "RTX4090", "cuda_event_correctness_latency",
            )
            self.assertEqual(ExperienceRetriever(store).query(unrelated), [])

    def test_progressive_context_is_bounded_and_advisory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperienceStore(Path(directory) / "experiences.jsonl")
            for item in (record("EXP-success", "PROMOTED"), record("EXP-failure", "COMPILE_FAILURE", family="custom_triton")):
                store.append(item)
            hits = ExperienceRetriever(store).query(TARGET)
            text, loaded = ExperienceContextBuilder(max_full_records=2, max_characters=400).build(hits)
            self.assertEqual(set(loaded), {"EXP-success", "EXP-failure"})
            self.assertLessEqual(len(text), 400)
            self.assertIn("advisory priors", text)
            self.assertIn("evaluate the current task independently", text)

    def test_memory_disabled_path_is_valid(self):
        text, loaded = ExperienceContextBuilder().build([])
        self.assertEqual(loaded, [])
        self.assertIn("Cross-task experience memory", text)

    def test_evaluated_snapshot_remains_distinct_after_later_workspace_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            kernel = workspace / "solution" / "kernel.py"
            kernel.parent.mkdir(parents=True)
            kernel.write_text("evaluated version\n")
            witness = capture_evaluated_candidate(workspace)
            kernel.write_text("later unbenchmarked edit\n")
            snapshot = Path(witness["artifact_path"])
            self.assertEqual(snapshot.read_text(), "evaluated version\n")
            self.assertNotEqual(hashlib.sha256(kernel.read_bytes()).hexdigest(), witness["candidate_sha256"])

    def test_branch_result_binds_promotion_surface_to_evaluated_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "child"
            kernel = workspace / "solution" / "kernel.py"
            kernel.parent.mkdir(parents=True)
            kernel.write_text("evaluated\n")
            witness = capture_evaluated_candidate(workspace)
            kernel.write_text("later edit\n")
            report = workspace / "trajectory" / "one" / "results.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"results": {"task": {"work": {"status": "PASSED", "speedup_factor": 1.2}}}}))
            task = TaskContract("objective", ("solution/kernel.py",), "bash scripts/bench.sh", "oracle", ("do not edit evaluator",), ("stop",))
            controller = TwoBranchController(project_root=root, task=task, operator="op", dataset=root, parent_kernel=kernel, parent_id="parent", parent_speedup=1.0)
            result = controller._branch_result(BranchSpec("b1", "parent", "h", "family", "r", "s", "p"), workspace, None, None)
            self.assertEqual(result.candidate_path, Path(witness["artifact_path"]))
            self.assertEqual(result.evaluated_candidate_sha256, witness["candidate_sha256"])
            self.assertEqual(result.evaluated_at, witness["evaluation_timestamp"])

    def test_final_extraction_records_only_evaluated_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ExperienceStore(root / "memory.jsonl")
            branch = BranchResult("b1", "codex", "session", "PASSED", root / "candidate.py", "",
                {"status": "PASSED", "speedup_factor": 1.1}, 1, "done", None,
                evaluated_candidate_sha256="a" * 64, evaluated_candidate_path=root / "snapshot.py")
            round_result = RoundResult("round-001", RoundState.PROMOTED, "reference", 1.0, (branch,), "b1", "PROMOTE")
            specs = (BranchSpec("b1", "reference", "hypothesis", "native", "reason", "signal", "prompt"),)
            created = extract_round(store, signature=SOURCE, campaign_id="final-t01", round_result=round_result, specs=specs)
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0].outcome, "PROMOTED")
            self.assertFalse(extract_round(store, signature=SOURCE, campaign_id="final-t01", round_result=round_result, specs=specs))

    def test_harness_preflight_failure_does_not_become_a_strategy_lesson(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ExperienceStore(root / "memory.jsonl")
            branch = BranchResult(
                "b1", "codex", "", "FAILED", root / "candidate.py", "", None, 0, "",
                "BuildLanguageConfigurationError",
            )
            round_result = RoundResult("round-001", RoundState.NO_IMPROVEMENT, "reference", 1.0, (branch,), None, "KEEP_CURRENT_PARENT")
            specs = (BranchSpec("b1", "reference", "hypothesis", "custom_triton_elementwise", "reason", "signal", "prompt"),)
            self.assertEqual(extract_round(store, signature=SOURCE, campaign_id="preflight-t01", round_result=round_result, specs=specs), [])
            self.assertEqual(store.list(), [])


if __name__ == "__main__":
    unittest.main()

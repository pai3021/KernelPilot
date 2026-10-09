from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from .models import ExperienceRecord, TaskSignature
from .store import ExperienceStore

SOURCE_SIGNATURE = TaskSignature("KernelBench", "1_Square_matrix_multiplication_", "level1", "matrix_multiplication", "gemm", "A[N,N],B[N,N]", "square_4096x4096", "float32", "RTX4090", "cuda_event_correctness_latency")

def _outcome(result: Mapping[str, Any], promoted: bool) -> str:
    if promoted: return "PROMOTED"
    status = str(result.get("status", ""))
    if status == "COMPILE_ERROR": return "COMPILE_FAILURE"
    if status == "INCORRECT_NUMERICAL": return "CORRECTNESS_FAILURE"
    if status == "TIMEOUT": return "TIMEOUT"
    if status in {"RUNTIME_ERROR"}: return "RUNTIME_FAILURE"
    return "VALID_NOT_PROMOTED" if status == "PASSED" else "INFRA_FAILURE"

def import_phase2(project_root: Path, store: ExperienceStore) -> list[ExperienceRecord]:
    root = Path(project_root)
    sources = [
        ("phase2a-r001", root / "artifacts/phase2a/phase2a-r001/round-result-recovered.json", root / "artifacts/phase2a/phase2a-r001/branch-specs.json"),
        ("phase2b-c002", root / "artifacts/phase2b/phase2b-c002/round-002.json", root / "artifacts/phase2b/phase2b-c002/round-002/branch-specs.json"),
    ]
    created: list[ExperienceRecord] = []
    for campaign, result_path, specs_path in sources:
        result_data, specs = json.loads(result_path.read_text()), {item["branch_id"]: item for item in json.loads(specs_path.read_text())}
        for branch in result_data["branches"]:
            branch_id, spec, evaluation = str(branch["branch_id"]), specs[str(branch["branch_id"])], dict(branch.get("benchmark_result") or {})
            trajectory_result = Path(str(branch["artifacts"]["benchmark_result"]))
            snapshot = trajectory_result.parent / "kernel.py"
            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            promoted = result_data.get("selected_branch_id") == branch_id and result_data.get("promotion_decision") == "PROMOTE"
            outcome = _outcome(evaluation, promoted)
            failure_reason = str(evaluation.get("error_log", "")) if outcome not in {"PROMOTED", "VALID_NOT_PROMOTED"} else None
            lesson = (f"Historical {spec['strategy_family']} produced a {evaluation.get('speedup_factor')}x observation; treat as an advisory prior." if outcome in {"PROMOTED", "VALID_NOT_PROMOTED"} else f"A previous {spec['strategy_family']} ended as {outcome}; avoid blindly replaying this exact implementation, but do not treat it as a prohibition.")
            record = ExperienceRecord(f"EXP-{campaign}-{branch_id}", SOURCE_SIGNATURE, campaign, str(result_data["round_id"]), branch_id, str(result_data["parent_id"]), str(spec["hypothesis"]), str(spec["strategy_family"]), str(spec["worker_prompt"])[:280], evaluation, outcome, branch.get("failure_type"), failure_reason, lesson, digest, {"evaluated_candidate_snapshot": str(snapshot), "benchmark_result": str(trajectory_result)}, datetime.now(timezone.utc).isoformat())
            if store.append(record):
                created.append(record)
    return created

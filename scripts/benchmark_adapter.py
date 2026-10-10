"""KernelBench adapter: the benchmark-specific seam for KernelPilot."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any

STATUS_PASSED = "PASSED"
STATUS_COMPILE_ERROR = "COMPILE_ERROR"
STATUS_INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
STATUS_RUNTIME_ERROR = "RUNTIME_ERROR"
STATUS_TIMEOUT = "TIMEOUT"

DATASET_PATH_ENV = "KERNELBENCH_PATH"
LEGACY_DATASET_PATH_ENV = "AKO_DATASET_PATH"
NCU_NVTX_RANGE = "kernelbench_not_supported"
# Required by Modal-side imports; KernelBench is local-only in this phase.
MODAL_IMAGE_REGISTRY = ""
MODAL_PYTHON = "3.10"
MODAL_PACKAGE_PIN = ""
MODAL_EXTRA_PIN = ""
_BLOB_FORMAT = "ako4x-kernelbench-v1"


def _load_blob(blob: str) -> dict[str, Any]:
    try:
        data = json.loads(blob)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid KernelBench solution blob: {exc}") from exc
    required = ("format", "name", "definition", "author", "sources", "build")
    if data.get("format") != _BLOB_FORMAT or any(key not in data for key in required):
        raise ValueError("Unsupported or incomplete KernelBench solution blob")
    if not isinstance(data["sources"], dict) or not data["sources"]:
        raise ValueError("KernelBench solution blob has no source files")
    return data


def _roots(dataset_path: str | os.PathLike) -> list[Path]:
    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"KernelBench path does not exist: {root}")
    pointer = root / "kernelbench_path.txt"
    if pointer.is_file():
        root = Path(pointer.read_text().strip()).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"KernelBench path recorded by dataset shim does not exist: {root}")
    nested = root / "KernelBench"
    return [nested, root] if nested.is_dir() else [root]


def _task_path(dataset_path: str | os.PathLike, task_id: str) -> Path:
    task_id = task_id.replace("\\", "/").removesuffix(".py")
    matches = []
    for root in _roots(dataset_path):
        direct = root / f"{task_id}.py"
        if direct.is_file():
            matches.append(direct.resolve())
        if "/" not in task_id:
            matches.extend(path.resolve() for path in sorted(root.glob(f"level*/{task_id}.py")))
    matches = list(dict.fromkeys(matches))
    if not matches:
        raise FileNotFoundError(f"KernelBench task {task_id!r} was not found under {dataset_path}")
    if len(matches) != 1:
        raise ValueError(f"KernelBench task {task_id!r} is ambiguous: {matches}")
    return matches[0]


def _task_id(path: Path, dataset_path: str | os.PathLike) -> str:
    for root in _roots(dataset_path):
        try:
            return path.resolve().relative_to(root.resolve()).with_suffix("").as_posix()
        except ValueError:
            pass
    return path.with_suffix("").name


def _candidate_source(data: dict[str, Any]) -> str:
    entry = str(data["build"].get("entry_point", "kernel.py::run")).split("::", 1)[0]
    source = data["sources"].get(entry)
    if source is None and len(data["sources"]) == 1:
        source = next(iter(data["sources"].values()))
    if source is None:
        raise ValueError(f"Entry source {entry!r} is absent from solution blob")
    if "ModelNew" not in source:
        raise ValueError("KernelBench candidate must define class ModelNew")
    return source


def _reference_equivalent_source(task_source: str) -> str:
    """Produce the default smoke candidate without altering the oracle Model."""
    return task_source + "\n\nclass ModelNew(Model):\n    pass\n"


def _backend(build: dict[str, Any]) -> str:
    language = str(build.get("language", "python")).lower()
    return language if language in {"triton", "tilelang", "cute"} else "cuda"


def _device(params: dict[str, Any]):
    import torch
    raw = params.get("device", "cuda:0")
    device = torch.device(f"cuda:{raw}" if isinstance(raw, int) else str(raw))
    if device.type != "cuda":
        raise ValueError(f"KernelBench requires CUDA, got {device}")
    if device.index == 4 and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Physical GPU 4 is reserved; choose a different GPU.")
    return device


def _metadata(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(value if isinstance(value, dict) else {}, default=str))


def _failure(status: str, name: str, axes: dict, metadata: dict) -> dict:
    max_difference = metadata.get("max_difference", "NaN")
    if isinstance(max_difference, (list, tuple)):
        max_difference = float(max_difference[-1]) if max_difference else "NaN"
    return {
        "status": status, "solution": name, "axes": axes,
        "max_abs_error": max_difference,
        "max_rel_error": "NaN",
        "error_log": f"{status}: {json.dumps(metadata, default=str, sort_keys=True)}",
    }


def list_workloads(dataset_path, definition):
    if definition:
        task = _task_path(dataset_path, definition)
        return [{"uuid": _task_id(task, dataset_path), "axes": {"level": task.parent.name}}]
    for root in _roots(dataset_path):
        tasks = sorted(root.glob("level*/*.py"))
        if tasks:
            return [{"uuid": _task_id(task, dataset_path), "axes": {"level": task.parent.name}}
                    for task in tasks]
    return []


def pack(source_dir, build_cfg, *, name, definition, author):
    """Store candidate sources and build metadata in an opaque JSON blob."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Solution source directory does not exist: {source_dir}")
    sources = {
        path.name: path.read_text() for path in sorted(source_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".py", ".cu", ".cpp", ".h", ".hpp", ".cuh"}
    }
    if not sources:
        raise ValueError(f"No supported source files in {source_dir}")
    return json.dumps({
        "format": _BLOB_FORMAT, "name": name, "definition": definition,
        "author": author, "build": dict(build_cfg), "sources": sources,
    }, indent=2, sort_keys=True)


def solution_meta(blob):
    data = _load_blob(blob)
    return {key: data[key] for key in ("name", "definition", "author")}


def _evaluate(task_id: str, data: dict, params: dict, dataset_path, capture_logs: bool) -> dict:
    import torch
    from kernelbench.eval import eval_kernel_against_ref

    task = _task_path(dataset_path, task_id)
    correct_trials = int(params.get("correct_trials", params.get("num_trials", 3)))
    perf_trials = int(params.get("perf_trials", params.get("iterations", 20)))
    if correct_trials < 1 or perf_trials < 1:
        raise ValueError("correct_trials and perf_trials must both be positive")
    timing_method = str(params.get("timing_method", "cuda_event"))
    log = io.StringIO()
    redirect = contextlib.redirect_stdout(log) if capture_logs else contextlib.nullcontext()
    with redirect:
        result = eval_kernel_against_ref(
            task.read_text(), _candidate_source(data),
            num_correct_trials=correct_trials, num_perf_trials=perf_trials,
            measure_performance=True, timing_method=timing_method, verbose=False,
            device=_device(params), backend=_backend(data["build"]),
            # KernelBench computes ref_runtime in this branch. It is needed for
            # the AKO reference_latency_ms contract, irrespective of its warning.
            precision=torch.float32, check_for_excessive_speedup=True,
        )
    axes = {"level": task.parent.name}
    metadata = _metadata(getattr(result, "metadata", {}))
    if result is None or not getattr(result, "compiled", False):
        item = _failure(STATUS_COMPILE_ERROR, data["name"], axes, metadata)
    elif not getattr(result, "correctness", False):
        status = STATUS_RUNTIME_ERROR if "runtime_error" in metadata else STATUS_INCORRECT_NUMERICAL
        item = _failure(status, data["name"], axes, metadata)
    else:
        latency = float(getattr(result, "runtime", -1.0))
        reference = float(getattr(result, "ref_runtime", -1.0))
        if latency <= 0 or reference <= 0:
            item = _failure(STATUS_RUNTIME_ERROR, data["name"], axes, metadata)
        else:
            item = {
                "status": STATUS_PASSED, "solution": data["name"], "axes": axes,
                "latency_ms": latency, "reference_latency_ms": reference,
                "speedup_factor": reference / latency, "max_abs_error": 0.0,
                "max_rel_error": 0.0, "timing_method": timing_method,
                "num_correct_trials": correct_trials, "num_perf_trials": perf_trials,
                "device": str(_device(params)),
            }
    if capture_logs:
        item["log"] = log.getvalue()
    return item


def run(blob, uuids, params, *, dataset_path, capture_logs=False, capture_autotune=False):
    """Evaluate requested tasks and return the frozen AKO normalized result shape."""
    data = _load_blob(blob)
    task_ids = list(dict.fromkeys(uuids))
    if not task_ids:
        raise ValueError("No KernelBench workload UUIDs were requested")
    results = {}
    for task_id in task_ids:
        # Invalid IDs must never silently score a partial subset.
        _task_path(dataset_path, task_id)
        try:
            results[task_id] = _evaluate(task_id, data, params, dataset_path, capture_logs)
        except Exception as exc:
            results[task_id] = _failure(
                STATUS_RUNTIME_ERROR, data["name"], {},
                {"exception_type": type(exc).__name__, "exception": str(exc)},
            )
    normalized = {data["definition"]: results}
    return {"results": normalized, "autotune_log": ""} if capture_autotune else normalized


def profile(blob, uuid, opts, *, dataset_path, env_pairs=None):
    return "NCU profiling not supported for KernelBench in Phase 0.6"


def list_ncu_options():
    return "NCU profiling not supported for KernelBench in Phase 0.6"


def sanitize(blob, uuid, opts, *, dataset_path):
    return "compute-sanitizer not supported for KernelBench in Phase 0.6"


def cheat_check(blob, uuids, *, dataset_path, n_iters=4):
    return {"status": "SKIPPED", "reason": "KernelBench cheat-check is not integrated in Phase 0.6"}

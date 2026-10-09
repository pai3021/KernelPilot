#!/usr/bin/env python3
"""Harness-facing runner for the SSH GPU evaluator.

This is intentionally the same ``bench_utils`` orchestration used locally;
only the execution backend differs.  It is platform-neutral and is the
recommended Windows control-plane entry point (``python scripts/run_remote.py``).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BRIDGE_DIR_ENV = "EXPERIMENTPILOT_BENCHMARK_BRIDGE_DIR"
BRIDGE_SERVER_ENV = "EXPERIMENTPILOT_BENCHMARK_BRIDGE_SERVER"

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import scripts.benchmark_adapter as adapter
from benchmark_backend import RemoteSSHBenchmarkBackend
from benchmark_backend.budget import begin_benchmark_scope_from_environment
from benchmark_backend.provenance import capture_evaluated_candidate
from scripts.bench_utils import (
    find_group_axis,
    parse_int_filter,
    run_ab_compare,
    run_and_report,
    run_variance_check,
)
from scripts.pack_solution import pack_solution


def _configure_console_output() -> None:
    """Do not turn a completed result into a Windows GBK print failure."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")


def load_remote_config() -> dict:
    """Load transport config without putting credentials in the workspace."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Remote runner needs {config_path}")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    # [benchmark.remote] is the V1 transport configuration. Keep the flat
    # table as a compatibility fallback for hand-created Phase 1A children.
    values = dict(config.get("benchmark", {}))
    remote = config.get("benchmark", {}).get("remote", {})
    if not remote:
        remote = config.get("remote_benchmark", {})
    aliases = {
        "host": "remote_host",
        "python": "remote_python",
        "kernelbench": "remote_kernelbench",
        "runs_dir": "remote_runs_dir",
        "compute_sanitizer": "remote_compute_sanitizer",
        "nsys": "remote_nsys",
    }
    values.update({aliases.get(key, key): value for key, value in remote.items()})
    return values


def main() -> None:
    # A Codex workspace-write child cannot always create network sockets.  Keep
    # direct runner invocation equivalent to scripts/bench.sh by routing it to
    # the parent runtime bridge when one is supplied.  The bridge server clears
    # this variable before invoking this same script outside the sandbox.
    if os.environ.get(BRIDGE_DIR_ENV) and not os.environ.get(BRIDGE_SERVER_ENV):
        client = Path(__file__).with_name("benchmark_bridge_client.py")
        completed = subprocess.run([sys.executable, str(client), *sys.argv[1:]], check=False)
        raise SystemExit(completed.returncode)
    _configure_console_output()
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Run benchmark through the remote SSH GPU evaluator")
    parser.add_argument("--label", default=None)
    parser.add_argument("--force-baseline", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--first", type=int, default=0, metavar="N")
    parser.add_argument("--group", type=str, default=None, metavar="VALUES")
    parser.add_argument("--exclude-group", type=str, default=None, metavar="VALUES")
    parser.add_argument("--index", type=str, default=None, metavar="INDICES")
    parser.add_argument("--variance-check", type=int, default=0, metavar="N")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ab-compare", dest="ab_compare", type=str, default=None, metavar="LABEL")
    parser.add_argument("--capture-logs", dest="capture_logs", action="store_true")
    args = parser.parse_args()
    # One budget unit represents the complete reference/candidate comparison,
    # not the two backend calls made internally by ``run_and_report``.
    begin_benchmark_scope_from_environment()

    group_axis = ""
    group_values = exclude_group_values = workload_indices = None
    if args.group or args.exclude_group or args.smoke:
        group_axis = find_group_axis()
        if not group_axis and (args.group or args.exclude_group):
            parser.error("--group/--exclude-group requires a variable axis in definition.json")
    if args.group:
        group_values = parse_int_filter(args.group)
    if args.exclude_group:
        exclude_group_values = parse_int_filter(args.exclude_group)
    if args.index:
        workload_indices = parse_int_filter(args.index)

    backend = RemoteSSHBenchmarkBackend(load_remote_config())
    provenance = capture_evaluated_candidate(PROJECT_ROOT)
    if not args.quiet:
        print(f"Evaluated candidate SHA-256: {provenance['candidate_sha256']}")
        print("Packing solution for isolated remote evaluation...")
    solution_blob = pack_solution(quiet=args.quiet).read_text()
    if not args.quiet:
        meta = adapter.solution_meta(solution_blob)
        print(f"\nLoaded: {meta['name']} ({meta['definition']})")

    def run_benchmark(blob: str, uuids: list, params: dict) -> dict:
        return backend.evaluate_blob(blob, uuids, params, capture_logs=args.capture_logs).results

    filters = dict(
        max_workloads=args.first, group_values=group_values,
        exclude_group_values=exclude_group_values, workload_indices=workload_indices,
        smoke=args.smoke,
    )
    if args.ab_compare:
        run_ab_compare(solution_blob, run_benchmark, label=args.ab_compare, backend="ssh",
                       quiet=args.quiet, current_label=args.label, **filters)
    elif args.variance_check > 0:
        run_variance_check(solution_blob, run_benchmark, n_runs=args.variance_check,
                           backend="ssh", quiet=args.quiet, label=args.label, **filters)
    else:
        run_and_report(solution_blob, run_benchmark, force_baseline=args.force_baseline,
                       label=args.label, backend="ssh", quiet=args.quiet, **filters)


if __name__ == "__main__":
    main()

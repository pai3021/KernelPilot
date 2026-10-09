# KernelBench reference

## Candidate and oracle

Each KernelBench task is a Python file under level<N>/. It defines Model,
get_inputs(), and get_init_inputs(). Model is the immutable PyTorch oracle.
A candidate supplied to the adapter must define ModelNew with compatible
construction and forward behavior.

The default spawned solution is reference-equivalent solely to validate the
harness loop. It is not an optimization.

## Result status

- PASSED: all correctness trials passed and both latencies were measured.
- COMPILE_ERROR: ModelNew could not load or compile.
- INCORRECT_NUMERICAL: ModelNew ran but failed the oracle check.
- RUNTIME_ERROR: setup, execution, or timing failed.
- TIMEOUT: reserved for a future per-task timeout implementation.

Only PASSED has latency_ms, reference_latency_ms, and speedup_factor.

## Configuration

[benchmark] defaults originate in templates/benchmark/evaluation.toml:

- correct_trials: distinct randomized oracle inputs; default 3.
- perf_trials: CUDA Event timing trials; default 20.
- timing_method: cuda_event in V1.
- device: CUDA ordinal inside CUDA_VISIBLE_DEVICES; default cuda:0.
- task: recorded default smoke task. Spawn operator selection remains the
  authoritative task selection.

The generic keys baseline_iterations, solution_iterations, warmup_runs, and
num_trials remain present for harness compatibility. The adapter forwards the
KernelBench-specific settings without changing scoring or baseline semantics.

## Safety and interpretation

Before a benchmark, inspect nvidia-smi. GPU 4 is reserved and must not be used.
Use CUDA_VISIBLE_DEVICES to bind an idle physical GPU; then leave device=cuda:0.

Do not edit the task source, evaluator, or oracle. Correctness failure means
the candidate is not a valid performance candidate. CUDA Event measurements for
very short kernels are noisy, so confirm meaningful changes with repeated runs.

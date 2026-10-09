---
name: benchmark
description: Contract for the active KernelBench evaluator. Use when editing a candidate, interpreting correctness or timing, or changing [benchmark] settings.
---

# KernelBench benchmark contract

The active benchmark is KernelBench on the configured RTX 4090. The task's
PyTorch Model is the correctness oracle; your candidate must define ModelNew.

Run the harness entrypoint from a spawned child:

~~~bash
CUDA_VISIBLE_DEVICES=<idle GPU except 4> bash scripts/bench.sh --first 1
~~~

The evaluator returns correctness, reference latency, candidate latency, and
speedup = reference latency / candidate latency. CUDA Event timing is the V1
default. A candidate that fails compilation, raises at runtime, or fails
correctness is invalid: do not claim a performance result from it.

Do not modify the evaluator, task reference, or correctness oracle to obtain a
pass. Very short kernels can have noisy timings; use enough trials and compare
latencies rather than treating a tiny speedup as conclusive. Details and config
field meanings are in benchmark.md.

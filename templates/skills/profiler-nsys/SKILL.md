---
name: profiler-nsys
description: Use Nsight Systems for on-demand CUDA timeline diagnosis after a formally correct KernelBench candidate needs performance investigation.
---

# Nsight Systems timeline diagnosis

Run `bash scripts/profile.sh` to collect a lightweight remote CUDA timeline.
This is diagnostic only: KernelBench remains the sole source of correctness,
latency, and speedup verdicts.

Use the returned report and summary to investigate:

- which CUDA kernels consume most time;
- excessive kernel launches or missed fusion opportunities;
- CUDA memcpy activity;
- expensive CUDA API calls or synchronization;
- visible CPU/GPU idle gaps in the `.nsys-rep` timeline.

This profiler collects CUDA, NVTX, and OS-runtime tracing. It does not collect
GPU hardware metrics or performance counters. It cannot answer warp-stall,
occupancy, cache-hit-rate, or hardware-throughput questions; those require
Nsight Compute (`profiler-ncu`) when that environment has counter permission.

Artifacts are written below `artifacts/diagnostics/profile/` and include the
raw `.nsys-rep`, raw stats, and `summary.json`. Invoke it only when diagnosis is
worth the added collection overhead; do not treat it as a required step for
every candidate.

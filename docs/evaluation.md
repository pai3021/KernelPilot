# Evaluation notes

KernelPilot's KernelBench adapter checks each candidate against the fixed task oracle before latency ranking. A campaign may promote a candidate only after correctness passes, and the formal benchmark budget is recorded for each branch. The evaluated file is snapshotted so that a later workspace edit cannot silently become the promoted candidate.

The current public README does not present an aggregate speedup figure. A release result should include the task list, KernelBench revision, GPU and software environment, all correctness outcomes, formal evaluation counts, candidate hashes, timing method, and per-task measurements. Keep failed tasks in the denominator and separate infrastructure failures from incorrect kernels.

Earlier FlashInfer-Bench results use a different benchmark and GPU. Source attribution is in [third-party notices](../THIRD_PARTY_NOTICES.md).

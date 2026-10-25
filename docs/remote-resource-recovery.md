# Remote resource waiting and budget accounting

A busy shared GPU previously consumed a formal Runner credit before any candidate
ran. Repeated requests could exhaust both branches without providing feedback.
New managed SSH branches separate preparation from evaluation dispatch and stop
the search when remote infrastructure blocks progress.

## Readiness and stopping

`[benchmark.remote]` accepts these settings in the local remote configuration:

```toml
gpu_wait_timeout_seconds = 60
gpu_poll_interval_seconds = 5
```

The control-side Runner polls GPU occupancy; it does not ask the coding agent to
resubmit. A zero wait timeout means one check without retries. SSH queries have
their own bounded timeout (at most 20 seconds); an in-flight query can extend
past the retry-wait deadline. The configured GPU, memory threshold, and zero
utilization requirement remain unchanged. GPU 4 remains reserved. Malformed
inventory, SSH failures, and missing devices fail immediately instead of being
treated as ordinary contention.

After environment preflight and uploading the fixed request/helper, a second
readiness gate checks the same physical GPU. Automatic selection cannot switch
that run to another device during this gate. Each gate has its own wait limit;
waiting, preparation, and execution also share the worker's overall time limit.
Use worker timeouts that accommodate this preparation time.

For a branch with `EXPERIMENTPILOT_BENCHMARK_BUDGET_FILE` set, an infrastructure
failure writes two records next to that ledger:

- `<ledger>.remote-block.json` preserves the first stop reason, run ID, failure
  stage, and whether evaluator dispatch was attempted.
- `<ledger>.remote-events.jsonl` appends readiness observations, dispatches,
  finishes, and stop events. Resource-wait seconds and poll counts remain visible
  even when no evaluation credit is consumed. Total backend wall time includes
  setup/transport and must not be reported as GPU-active time.

Subsequent formal evaluation calls fail before another SSH request. The Codex filesystem
bridge also refuses them and terminates its own worker process group after a
stop is reported. The controller preserves earlier valid candidate snapshots,
marks the round `RESOURCE_BLOCKED`, and does not launch the next branch or promote
a candidate from this incomplete search. An incorrect candidate is different:
it consumes its dispatched credit and permits feedback-driven repair within the
remaining limit.

These persistent stops require a managed ledger. Standalone unbudgeted backend
clients receive bounded-wait failure results but have no durable branch circuit.
The guard is an engineering reliability measure, not a security boundary against
an adversarial worker.

## Accounting policy

New `initialize_budget(...)` calls and `TwoBranchController` branches default to
`accounting_policy="dispatch"`. The ledger records this policy explicitly.

| Point of failure | Dispatch-policy credit |
| --- | --- |
| Argument filtering, packing, or no selected workloads | Not consumed |
| GPU wait, environment preflight, remote mkdir, or upload | Not consumed |
| Final readiness gate fails before any evaluation in the comparison | Not consumed |
| Reference evaluator is dispatched | One comparison credit consumed |
| Candidate evaluator in the same Runner invocation | Shares that credit |
| Candidate blocked after a dispatched reference | Credit remains consumed |
| SSH interruption, timeout, or missing result after dispatch was attempted | Credit remains consumed; stop and inspect |
| Candidate compilation or correctness failure | Credit remains consumed |

The atomic ledger reservation occurs immediately before invoking the evaluator
SSH command. `evaluation_dispatched=true` records that attempted dispatch; it is
not proof that the remote GPU executed or completed. Ambiguous outcomes are
never automatically refunded or rerun. Existing bounded file-transfer retries
may fetch the same result again; they do not rerun the evaluator.

A credit is a Runner comparison scope, not a physical GPU launch. Reference and
candidate calls, and diagnostic modes such as variance checks, can involve
multiple backend dispatches. Count those separately and keep the same permitted
Runner modes in every experimental arm.

Ledgers without `accounting_policy` retain historical `runner_attempt` behavior.
To reproduce that behavior explicitly, use
`initialize_budget(path, accounting_policy="runner_attempt")` or
`TwoBranchController(..., benchmark_budget_policy="runner_attempt")`. The legacy
`TwoRoundCampaign` replay imports earlier branch-attempt counts and selects this
policy explicitly; its campaign counter still reserves at branch start. It also
stops on remote blockage, but must not be presented as the new dispatch-accounted
experimental driver. Completed run artifacts and their budgets are not migrated.

## Recovery and fair comparisons

The stop survives a new Runner process or Codex restart. A malformed stop record
fails closed. The controller refuses to initialize over an existing branch
ledger. Automatic mid-branch resume and ledger reset are deliberately absent.

Inspect the original remote request/result and stop stage first. If dispatch was
attempted, recover any existing result under that run ID before considering a
new evaluation. Preserve the failed cohort and its cost. A new run needs a new
artifact directory, a recorded budget policy, and an explicit compute cap.

Readiness polling is not an atomic GPU reservation: another user can start a job
between a check and execution or during measurement. Use a coordinated exclusive
GPU window or a dedicated rental instance for a performance comparison. The code
does not change GPUs mid-run, kill other users' jobs, or claim cross-user lease
enforcement. If moving to different hardware, rerun all paired experimental arms
on that hardware and keep the historical cohort separate.

## Validation

`python3 -m unittest discover -s tests -q` needs no GPU. Failure-injection tests
exercise busy-to-idle recovery, bounded busy waits, occupancy during upload,
preflight/transfer/ambiguous-execution failures, reference/candidate accounting,
persistent stops, worker termination, and stopping before another branch. These
checks validate local control flow, not live remote-GPU performance or exclusive
resource ownership.

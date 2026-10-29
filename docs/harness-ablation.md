# Comparing outer planning with continuous Codex feedback

The earlier "native Codex" pilot was Codex with shared KernelPilot task and
evaluation support. Its worker received `AGENTS.md`, `CODEX_TASK.md`, Kernel
Skills, task/reference source, a fixed Runner, remote transport, budget guards,
and candidate snapshots. The informed cohort also supplied identical
`TASK_FACTS.json` to every worker. Best-passing candidate selection was used for
both arms. The baseline did not use the outer Planner, branch Controller,
Experience Memory, or self-evolution.

This baseline supports an ablation of outer planning/search under shared
execution support. It does not measure the value of the whole execution stack
against an otherwise bare coding agent. Sharing required task facts and the
correctness oracle controls the problem definition; sharing Skills and feedback
support also controls for those parts of KernelPilot's value.

## Primary comparison: packaged optimization workflow

Compare a minimal native-Codex baseline with an explicitly frozen KernelPilot
configuration. The first comparison uses one continuous worker and six search
submissions per arm; KernelPilot additionally uses its task/Skill organization,
one outer planning direction, trajectory management and candidate selection.
Memory and self-evolution remain off and are not covered by this comparison.

The minimal baseline receives the task/reference source, identical initial
candidate, hardware/software facts, complete numerical and implementation rules,
and one evaluation command. It does not receive KernelPilot Skills, Worker
guides, `TASK_FACTS.json`, the outer plan, historical solutions or automatic
restoration of its best earlier candidate. The factual problem definition must
still be equivalent: precision rules and task constraints cannot be hidden to
weaken the baseline.

`scripts/prepare_codex_baseline.py` prepares this context in a new neutral
directory, refusing inherited project `AGENTS.md` files. It copies no backend,
Planner or runtime implementation into the agent workspace. Its small client
submits the exact candidate source and SHA to an external fixed evaluator.
`MinimalBenchmarkEndpoint` is observer-side infrastructure: the owner supplies
the evaluator callback, it enforces a search-submission cap, and it returns
correctness/timing/error feedback without strategy or best-candidate advice.
Candidate records stay outside the baseline workspace. Preparation does not
launch Codex or a GPU evaluator; a real callback and native CLI launch still need
integration and environment verification before paid execution.

At the end, freeze the baseline's delivered `solution/kernel.py` as-is. Freeze
the candidate actually selected by the KernelPilot workflow for the other arm.
Give each frozen artifact one independently budgeted final evaluation under the
same oracle and environment. Do not retroactively choose the baseline's best
trajectory on its behalf. Six search comparisons plus one terminal comparison
per arm propose a cap of 14 for the first pair; this is not an authorization.
Best-seen performance remains a secondary diagnostic for both arms.

The shared fixed evaluator and minimal GPU-access channel define the scientific
measurement apparatus. If the same remote backend implements that apparatus,
this comparison does not separately measure that backend's reliability value.
GPU leases, transport/recovery, and scheduling need an explicit execution-path
or fault-injection comparison. A performance improvement on one development
task cannot establish all KernelPilot components' value at once.

## Matched continuous-feedback control

| Arm | Outer planning | Worker sessions | Formal comparison cap | Worker time cap |
| --- | --- | --- | --- | --- |
| A6: Codex with shared support | None | One continuous session | 6 | 1800 s |
| B6: single-branch Harness | One hypothesis | One continuous session | 6 | 1800 s |
| C6: two-branch Harness | Two hypotheses | Two independent sessions | 3 each | 900 s each |

For outer-planning attribution, compare A6/B6. B6 is a real planning treatment: a Planner proposes one initial
hypothesis using the complete task facts, then one worker may adapt it using six
feedback opportunities. Generating two plans and discarding one would add an
unreported selection mechanism. Merely wrapping the identical A6 request in a
Controller would mainly compare lifecycle bookkeeping, with little reason to
expect different kernel search quality.

`TwoBranchController` retains its existing name and two-branch default. To run
the explicit single-branch treatment, configure it as follows in a new run:

```python
controller = TwoBranchController(
    # Supply the normal project/task/dataset/parent/artifact settings here.
    **common_settings,
    branch_ids=("b1",),
    branch_benchmark_limit=6,
    benchmark_budget_policy="dispatch",
    worker_timeout=1800,
    planning_facts=verified_task_facts,
    allow_strategy_adaptation=True,
)
```

The planner schema and parser then require exactly one branch. The two-branch
mode still requires two materially different strategy families. Invalid branch
counts/identities fail before a coding runtime is started. The one-branch path
uses the same budget, provenance, resource-stop and ranking code as the existing
controller.

## What must stay equal

All workers need the same model/reasoning configuration, task/reference starting
point, numerical policy, Skills, editable-file restrictions, Runner modes,
transport, memory settings, and budget accounting policy. A6 receives no Planner
output. B6/C6 receive only their declared plan and may change direction without
changing the oracle or budget. New cohorts need fresh paired A6 results rather
than reusing a historical best.

Record the worker task/context hashes and all Planner/worker requests before
launch. An observer should retain both best passing measured candidates and the
worker's final submitted file: giving both arms automatic best-of-trajectory
scoring is a controlled search-quality metric, not evidence that both agents
submitted that best candidate themselves.

Planner calls, including structured-output retries, count toward the Harness
token/time cost. Equal candidate caps and worker time caps do not imply equal
total compute cost. Record reference/candidate dispatches separately from Runner
credits. Infrastructure blockage ends the cohort as incomplete; follow
[remote resource recovery](remote-resource-recovery.md) rather than filling
missing observations with retries or silently refunding attempts.

The known matrix-multiplication task is a development regression task. Fresh
repeated runs on it can diagnose mechanisms, but gains require multiple held-out
tasks before making broader project-level claims. C6 versus B6 still changes
planning breadth and context/budget allocation together; it is a search-policy
comparison rather than a pure isolation of the number of worker processes.

The local tests simulate one Planner, one continuous worker and six bounded
evaluations, including an early best followed by failure. They do not establish
live GPU readiness or a performance advantage.

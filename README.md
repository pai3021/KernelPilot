# KernelPilot

KernelPilot is a personal GPU kernel optimization project. It gives a coding agent an isolated task workspace, checks candidates with a fixed benchmark, and uses the results to guide the next attempt.

The current workflow targets [KernelBench](https://github.com/ScalingIntelligence/KernelBench) tasks. A local WSL control process runs the agent; a remote GPU host performs correctness and latency evaluation over SSH. This is an experimental harness, not a hosted service.

## System overview

KernelPilot supports a manual single-task workflow and an optional multi-round campaign. The diagram shows the campaign: a master agent selects a parent kernel, spawns an isolated child workspace, and drives a sub agent to optimize and evaluate candidates. Archived variants can seed later rounds.

![KernelPilot architecture showing the master/sub loop, agent adapters, cross-session archive, and optional harness proposals](docs/images/kernelpilot-system-overview.png)

The sub agent uses two layers. **(a) Generic Agent Substrate** supplies the agent loop, context management, and tool use. Codex and Claude Code have runtime adapters today; other agents can be added through the same adapter interface. **(b) Kernel-Specific Harness** supplies the task template, benchmark adapter, reference archive (variants and lessons across sessions), and skill catalog. Optional harness proposals are reviewed against scope, evidence, and regression checks before a guidance change is applied.

The [quick start](#quick-start) follows the manual path: you drive one child workspace directly and evaluate over SSH on a GPU host. See [closed-loop campaigns](docs/closed-loop.md) for the optional master/sub workflow.

## What is implemented

- **Agent runtimes:** Codex and Claude Code share a task contract and workspace setup through `agent_runtime/`.
- **Isolated search:** `campaign/` plans two distinct branches, evaluates each child, and promotes only a correctness-passing improvement.
- **Evaluation boundary:** `benchmark_backend/` runs a fixed remote evaluator and records the evaluated candidate snapshot and formal benchmark budget.
- **Experience memory:** `experience_memory/` stores compact, result-bound lessons for later tasks.
- **Optional harness evolution:** `harness_evolution/` can propose small guidance changes after a campaign and applies them only after scope, evidence, and regression gates.

The original single-workspace and closed-loop commands remain in the repository. The documented starting path below is the KernelBench + Codex + SSH workflow.

## Repository map

| Path | Purpose |
| --- | --- |
| `spawn.py` | Create a task workspace from a KernelBench operator |
| `agent_runtime/`, `campaign/` | Agent execution and bounded branch search |
| `benchmark_backend/`, `scripts/benchmark_adapter.py` | Benchmark transport and KernelBench adapter |
| `experience_memory/`, `harness_evolution/` | Cross-task lessons and gated harness proposals |
| `templates/` | Files copied into a new task workspace |
| `tests/` | Local contract and regression tests |

## Quick start

Use WSL with Python 3.10 or newer, a native Codex CLI, OpenSSH, a KernelBench checkout, and an SSH-accessible GPU machine with KernelBench and PyTorch installed. GPU evaluation requires your own remote environment.

```bash
git clone https://github.com/pai3021/KernelPilot.git
cd KernelPilot
python3 -m pip install -e '.[kernelbench]'
cp configs/remote.example.toml configs/remote.local.toml
# Edit configs/remote.local.toml for your SSH host and remote paths.

python3 spawn.py --dataset /path/to/KernelBench
python3 spawn.py \
  --operator 1_Square_matrix_multiplication_ \
  --dataset /path/to/KernelBench \
  --backend ssh --gpu rtx4090 --agent codex \
  --remote-config configs/remote.local.toml --name demo
```

The command prints the child workspace path. Enter it, read `CODEX_TASK.md`, and start `codex`. Use `bash scripts/bench.sh --label "first candidate"` to evaluate a candidate. Keep your local SSH configuration and remote paths out of Git. `KERNELPILOT_REMOTE_CONFIG` can supply the config path for scripted spawns.

The remote host must contain the repository's evaluator code and a compatible KernelBench checkout. See [setup details](docs/installation.md) before a GPU run. For a local, GPU-free check of the repository contracts:

```bash
python3 -m unittest discover -s tests -q
```

## Recorded SSH example

A local Codex agent produced this [Triton matrix multiplication candidate](examples/kernelbench_square_matmul_triton.py) for KernelBench Level 1 `1_Square_matrix_multiplication_` (4096 × 4096). The fixed SSH evaluator checked it on an NVIDIA GeForce RTX 4090.

| Check | Recorded result |
| --- | --- |
| Correctness | PASSED (3 trials) |
| Timing budget | 20 CUDA-event trials |
| Evaluated candidate | SHA-256 `9643bd79209c723b3f5f5e45c5f879d99712efb4c181fd267607cad55cb5cfb0` |

To try the example, copy it to a generated child's `solution/kernel.py`, set `language = "triton"` under `[build]` in that child's `config.toml`, and run `bash scripts/bench.sh --label "example"`.

## Evaluation scope

KernelPilot records correctness, latency, candidate identity, and benchmark budget for each formal evaluation. See [evaluation notes](docs/evaluation.md) for the release evidence boundary. This README makes no aggregate speedup claim.

## License and credits

Released under the [MIT license](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for source attribution. KernelBench, Codex, Claude Code, and GPU tooling are separate projects with their own terms.

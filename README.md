# KernelPilot

KernelPilot is a personal GPU kernel optimization project. It gives a coding agent an isolated task workspace, checks candidates with a fixed benchmark, and uses the results to guide the next attempt.

The current workflow targets [KernelBench](https://github.com/ScalingIntelligence/KernelBench) tasks. A local WSL control process runs the agent; a remote GPU host performs correctness and latency evaluation over SSH. This is an experimental harness, not a hosted service.

## What is implemented

- **Agent runtimes:** Codex and Claude Code share a task contract and workspace setup through `agent_runtime/`.
- **Isolated search:** `campaign/` plans two distinct branches, evaluates each child, and promotes only a correctness-passing improvement.
- **Evaluation boundary:** `benchmark_backend/` runs a fixed remote evaluator and records the evaluated candidate snapshot and formal benchmark budget.
- **Experience memory:** `experience_memory/` stores compact, result-bound lessons for later tasks.
- **Harness proposals:** `harness_evolution/` reviews proposed changes through a scoped gate. It is optional.

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

## Recorded example

One recorded KernelBench Level 1 run optimized `14_Matmul_for_upper_triangular_matrices` on 4096 x 4096 inputs. In a two-round search, the selected branch used a 2 x 2 block decomposition to skip the known-zero part of the upper-triangular product.

| Check | Recorded result |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| Correctness | PASS |
| Reference / selected candidate | 2.45 ms / 1.34 ms (1.83x) |
| Formal GPU evaluations | 4 |

The SSH evaluator used CUDA event timing. The selected evaluated candidate has SHA-256 `d6a10aa1d490d32fcc4468a36f02fb983ee790f40e97151af4218b4df2bedc68`.

## Evaluation scope

KernelPilot records correctness, latency, candidate identity, and benchmark budget for each formal evaluation. See [evaluation notes](docs/evaluation.md) for the release evidence boundary. This README makes no aggregate speedup claim.

## License and credits

Released under the [MIT license](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for source attribution. KernelBench, Codex, Claude Code, and GPU tooling are separate projects with their own terms.

# KernelPilot

KernelPilot helps you optimize GPU kernels with coding agents. It creates a workspace for each [KernelBench](https://github.com/ScalingIntelligence/KernelBench) task, checks correctness and latency on a GPU host over SSH, and keeps the results for later attempts.

Use one workspace directly or run a multi-round campaign. In a campaign, the master agent picks a parent kernel and creates a child workspace. A sub agent edits and benchmarks the kernel. Passing variants go into the archive for later rounds.

![KernelPilot architecture showing the master/sub loop, agent adapters, cross-session archive, and optional harness proposals](docs/images/kernelpilot-system-overview.png)

The sub agent runs on **(a) Generic Agent Substrate**: the agent loop, context, and tools supplied by Codex or Claude Code. KernelPilot adds **(b) Kernel-Specific Harness**: a task template, benchmark adapter, reference archive with variants and lessons from earlier runs, and kernel skills. Campaigns can also propose changes to harness guidance. Those changes go through evidence and regression checks.

In one recorded RTX 4090 run, a Codex-generated Triton kernel cut `91_cumsum_reverse` from 30.7 ms to 9.84 ms (**3.12×**). [Code and benchmark details](#recorded-ssh-example).

The [quick start](#quick-start) uses one child workspace. For the master/sub workflow, see [closed-loop campaigns](docs/closed-loop.md).

## In the code

- `spawn.py` creates a task workspace from a KernelBench operator. `templates/` provides its starting files.
- `agent_runtime/` starts Codex or Claude Code in that workspace.
- `benchmark_backend/` and `scripts/benchmark_adapter.py` evaluate candidates on the GPU host and save the exact file that was measured.
- `campaign/` explores two branches per round and promotes a candidate after it passes correctness and improves latency.
- `experience_memory/` saves lessons tied to results. `harness_evolution/` checks optional changes to task guidance against evidence and regression tests.
- `tests/` covers the local contracts and regression checks.

## Quick start

Run the control process in WSL with Python 3.10+, a native Codex CLI, and OpenSSH. The GPU host needs PyTorch, KernelBench, and a matching KernelPilot checkout. See the [installation guide](docs/installation.md) for setup details.

```bash
git clone https://github.com/pai3021/KernelPilot.git
cd KernelPilot
python3 -m pip install -e '.[kernelbench]'
git clone https://github.com/ScalingIntelligence/KernelBench.git ../KernelBench
git -C ../KernelBench checkout 423217d
cp configs/remote.example.toml configs/remote.local.toml
# Edit configs/remote.local.toml for your SSH host and remote paths.

python3 spawn.py --dataset ../KernelBench
python3 spawn.py \
  --operator 91_cumsum_reverse \
  --dataset ../KernelBench \
  --backend ssh --gpu rtx4090 --agent codex \
  --remote-config configs/remote.local.toml --name demo
```

`spawn.py` prints the child workspace path. Enter it, read `CODEX_TASK.md`, and run `codex`. After editing the kernel, run `bash scripts/bench.sh --label "candidate-1"` from the child workspace.

For a local check that does not need a GPU:

```bash
python3 -m unittest discover -s tests -q
```

## Recorded SSH example

For KernelBench Level 1 `91_cumsum_reverse`, Codex generated a [Triton reverse-scan kernel](examples/kernelbench_reverse_cumsum_triton.py). It reads each row backward and computes the scan in one kernel. The reference uses two flips around `torch.cumsum`. The recorded SSH run used an NVIDIA GeForce RTX 4090.

| Check | Recorded result |
| --- | --- |
| Correctness | PASSED (3 trials) |
| Reference latency | 30.7 ms |
| Candidate latency | 9.84 ms |
| Speedup on this task | 3.12× |
| Timing | 20 CUDA-event trials |
| Evaluated candidate | SHA-256 `8acc4ec88b8cb56d5516775db4b9c7cef1b516487b70b2b1da4114a2fd8ecde2` |

To rerun it, use the `91_cumsum_reverse` quick-start command, copy the example to the child's `solution/kernel.py`, set `language = "triton"` under `[build]` in `config.toml`, and run `bash scripts/bench.sh --label "reverse-cumsum"`.

## Evaluation

Each run records correctness, latency, trial counts, and the hash of the evaluated file. See [evaluation notes](docs/evaluation.md) for the measurement details.

## License and credits

Released under the [MIT license](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for source attribution.

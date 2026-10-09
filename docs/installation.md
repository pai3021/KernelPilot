# Installation and first workspace

The documented KernelPilot path uses a WSL control process and a separate Linux GPU host. The control process needs Python 3.10+, native OpenSSH, Git, and a working Codex CLI. The GPU host needs a CUDA-capable PyTorch installation, a KernelBench checkout, and a checkout of this repository at the same revision as the control process.

From the repository root in WSL:

```bash
python3 -m pip install -e '.[kernelbench]'
python3 -m unittest discover -s tests -q
```

Copy `configs/remote.example.toml` to the ignored `configs/remote.local.toml`. Replace all sample paths and the SSH alias. The required fields are `host`, `remote_repo`, `python`, `kernelbench`, and `runs_dir`. The remote user must be able to create `runs_dir`; `python` must run the pinned KernelBench environment. `compute_sanitizer` and `nsys` are only needed for the optional diagnostic commands. Keep credentials in your SSH agent or SSH config, outside this repository.

The config file is read by the parent `spawn.py` and written into each child's `config.toml`. To list tasks and create a workspace:

```bash
python3 spawn.py --dataset /path/to/KernelBench
python3 spawn.py --operator 1_Square_matrix_multiplication_ \
  --dataset /path/to/KernelBench --backend ssh --gpu rtx4090 \
  --agent codex --remote-config configs/remote.local.toml --name demo
```

Use the path printed by `spawn.py` to enter the workspace. Its `CODEX_TASK.md` defines the optimization task for Codex. `bash scripts/bench.sh --label "candidate-1"` invokes the fixed remote evaluator. This is a real GPU operation; the unit tests above are the quick check without GPU access.

For scripted campaigns, set `KERNELPILOT_REMOTE_CONFIG` to the same local TOML path. The environment variable is inherited by `spawn.py`; the example file remains safe to commit, while `remote.local.toml` is ignored.

# KernelPilot contributor notes

- Run the control process and the unit tests with WSL Python 3.10+ when using the SSH backend. Use native Linux OpenSSH and the installed native coding-agent CLI.
- Remote GPU settings come from `--remote-config` or `KERNELPILOT_REMOTE_CONFIG`. Start from `configs/remote.example.toml`; keep local paths, SSH configuration, credentials, and run artifacts out of Git.
- Unit tests do not need a GPU: `python3 -m unittest discover -s tests -q`.
- Keep benchmark task identity, correctness checks, evaluation budgets, and recorded candidate snapshots stable when changing orchestration code. Do not overwrite completed run artifacts.
- `README.md` describes the supported KernelPilot entry path. Preserve source attribution in `LICENSE` and `THIRD_PARTY_NOTICES.md`.

# KernelPilot

[English](README.md) | 简体中文

KernelPilot 使用编程 Agent 优化 GPU 算子。它为每个 [KernelBench](https://github.com/ScalingIntelligence/KernelBench) 任务创建独立工作区，通过 SSH 在 GPU 主机上检查正确性、测量延迟，并保存结果，供后续迭代使用。

你可以在单个工作区中直接优化算子，也可以运行多轮搜索。在多轮搜索中，主 Agent 选择一个已有实现作为起点，创建子工作区。辅 Agent 在其中修改和测试算子。通过正确性检查且更快的实现会成为下一轮的起点。各候选实现的评测结果也会保留下来。

![KernelPilot 架构图：主辅 Agent 循环、Agent 底座、跨轮次归档与可选的 Harness 改进提议](docs/images/kernelpilot-system-overview.png)

辅 Agent 运行在 **(a) 通用 Agent 底座**上：Codex 或 Claude Code 提供 Agent 循环、上下文管理和工具调用。KernelPilot 在此基础上提供 **(b) 算子专用 Harness（Kernel-Specific Harness）**：任务模板、基准测试适配器、保存历次候选实现与经验的参考归档，以及算子开发技能。多轮搜索结束后，还可以选择提出 Harness 指引的修改建议。建议需要经过证据检查和回归测试。

在一次有记录的 RTX 4090 测试中，Codex 生成的 Triton 算子将 `91_cumsum_reverse` 的运行时间从 30.7 ms 降至 9.84 ms，达到 **3.12× 加速**。[查看代码与测试细节](#ssh-测试实例)。

下面的[快速开始](#快速开始)使用单个子工作区。主辅 Agent 的多轮流程见[多轮搜索说明](docs/closed-loop.md)。

## 代码结构

- `spawn.py` 根据 KernelBench 算子创建任务工作区，`templates/` 提供初始文件。
- `agent_runtime/` 在工作区中启动 Codex 或 Claude Code。
- `benchmark_backend/` 和 `scripts/benchmark_adapter.py` 在 GPU 主机上评测候选实现，并保存实际参与评测的文件。
- `campaign/` 每轮探索两个分支，在候选实现通过正确性检查且延迟降低后将其选入下一轮。
- `experience_memory/` 保存与评测结果关联的经验。`harness_evolution/` 根据证据和回归测试检查可选的任务指引修改。
- `tests/` 覆盖本地接口约束和回归检查。

## 快速开始

在 WSL 中使用 Python 3.10+、原生 Codex CLI 和 OpenSSH 运行控制流程。GPU 主机需要安装 PyTorch、KernelBench，并检出与本地控制端相同版本的 KernelPilot。环境配置详见[安装说明](docs/installation.md)。

```bash
git clone https://github.com/pai3021/KernelPilot.git
cd KernelPilot
python3 -m pip install -e '.[kernelbench]'
git clone https://github.com/ScalingIntelligence/KernelBench.git ../KernelBench
git -C ../KernelBench checkout 423217d
cp configs/remote.example.toml configs/remote.local.toml
# 修改 configs/remote.local.toml，填写 SSH 主机和远端路径。

python3 spawn.py --dataset ../KernelBench
python3 spawn.py \
  --operator 91_cumsum_reverse \
  --dataset ../KernelBench \
  --backend ssh --gpu rtx4090 --agent codex \
  --remote-config configs/remote.local.toml --name demo
```

`spawn.py` 会输出子工作区路径。进入该目录，阅读 `CODEX_TASK.md`，然后运行 `codex`。修改算子后，在子工作区执行 `bash scripts/bench.sh --label "candidate-1"` 进行评测。

不需要 GPU 的本地检查：

```bash
python3 -m unittest discover -s tests -q
```

## SSH 测试实例

对于 KernelBench Level 1 的 `91_cumsum_reverse`，Codex 生成了一个 [Triton 反向扫描算子](examples/kernelbench_reverse_cumsum_triton.py)。它从后往前读取每一行，并在一个 kernel 中完成扫描。参考实现则在 `torch.cumsum` 前后各进行一次翻转。此次 SSH 测试使用 NVIDIA GeForce RTX 4090。

| 检查项 | 记录结果 |
| --- | --- |
| 正确性 | 通过（3 次测试） |
| 参考实现延迟 | 30.7 ms |
| 候选实现延迟 | 9.84 ms |
| 该任务加速比 | 3.12× |
| 计时方式 | 20 次 CUDA Event 测试 |
| 评测文件 | SHA-256 `8acc4ec88b8cb56d5516775db4b9c7cef1b516487b70b2b1da4114a2fd8ecde2` |

如需复现，先运行快速开始中的 `91_cumsum_reverse` 命令，再将示例代码复制到子工作区的 `solution/kernel.py`，把 `config.toml` 中 `[build]` 下的 `language` 设为 `"triton"`，最后运行 `bash scripts/bench.sh --label "reverse-cumsum"`。

## 评测记录

每次评测都会记录正确性、延迟、测试次数，以及实际评测文件的哈希值。测量方式详见[评测说明](docs/evaluation.md)。

## 许可证与致谢

项目采用 [MIT 许可证](LICENSE)。来源与引用信息见[第三方声明](THIRD_PARTY_NOTICES.md)。

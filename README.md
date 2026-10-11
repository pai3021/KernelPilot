# KernelPilot

中文 | [English](README.en.md)

KernelPilot 用编程 Agent 优化 GPU 算子。选定一个 [KernelBench](https://github.com/ScalingIntelligence/KernelBench) 任务后，它会创建独立工作区，让 Agent 专注修改算子实现。评测通过 SSH 在 GPU 主机上完成，正确性、耗时和实际评测的代码都会留下记录，方便继续迭代。

最简单的用法是创建一个工作区，自己带着 Agent 迭代。需要连续尝试时，主 Agent 负责选择起点和安排方向，辅 Agent 在独立工作区改代码、跑评测。候选实现和评测结果会保留下来，供下一轮参考。

![KernelPilot 架构图，展示主辅 Agent、通用 Agent 底座、算子专用 Harness 和参考归档](docs/images/kernelpilot-system-overview.png)

图的上半部分是主辅 Agent 的迭代过程。辅 Agent 工作时用到的能力分成两层：

- **(a) 通用 Agent 底座**由 Codex 或 Claude Code 提供，负责管理上下文、调用工具和推进任务。
- **(b) 算子专用 Harness**由 KernelPilot 提供，包括任务模板、GPU 评测适配器、保存历史实现与经验的参考归档，以及 CUDA、Triton 等算子开发技能。

如果开启 Harness 自进化，系统会在优化后复盘评测记录，提出修改任务说明或技能文档的建议。建议要有评测证据，并通过回归测试后才会采纳。多轮优化的运行方法见[多轮优化说明](docs/closed-loop.md)。

## 代码从哪里看

- `spawn.py` 创建任务工作区，`templates/` 提供初始文件。
- `agent_runtime/` 接入 Codex 和 Claude Code。
- `benchmark_backend/` 与 `scripts/benchmark_adapter.py` 负责评测，并保存实际参与评测的代码。
- `campaign/` 组织每轮的两个优化方向，选择正确且更快的候选作为下一轮起点。
- `experience_memory/` 保存与评测结果关联的经验，`harness_evolution/` 检查可选的 Harness 改进建议。
- `tests/` 包含本地检查和回归测试。

## 快速开始

控制端运行在 WSL，需要 Python 3.10+、原生 Codex CLI 和 OpenSSH。GPU 主机需要安装 PyTorch、KernelBench，并检出与控制端相同版本的 KernelPilot。详细配置见[安装说明](docs/installation.md)。

```bash
git clone https://github.com/pai3021/KernelPilot.git
cd KernelPilot
python3 -m pip install -e '.[kernelbench]'
git clone https://github.com/ScalingIntelligence/KernelBench.git ../KernelBench
git -C ../KernelBench checkout 423217d
cp configs/remote.example.toml configs/remote.local.toml
# 在 configs/remote.local.toml 中填写 SSH 主机和远端路径。

python3 spawn.py --dataset ../KernelBench
python3 spawn.py \
  --operator 91_cumsum_reverse \
  --dataset ../KernelBench \
  --backend ssh --gpu rtx4090 --agent codex \
  --remote-config configs/remote.local.toml --name demo
```

第一条 `spawn.py` 命令列出可用任务，第二条创建 `91_cumsum_reverse` 的工作区。进入命令输出的路径，阅读 `CODEX_TASK.md`，然后运行 `codex`。修改算子后，在工作区执行 `bash scripts/bench.sh --label "candidate-1"`。

没有 GPU 时，也可以先运行本地测试：

```bash
python3 -m unittest discover -s tests -q
```

## 一次 SSH 评测

这里用 KernelBench Level 1 的 `91_cumsum_reverse` 做例子。参考实现先翻转每一行，调用 `torch.cumsum`，再翻转回来。Codex 生成的 [Triton 版本](examples/kernelbench_reverse_cumsum_triton.py) 则从后向前读取，在一个 kernel 中完成扫描。下面是 NVIDIA GeForce RTX 4090 上的一次 SSH 评测记录。

| 项目 | 结果 |
| --- | --- |
| 正确性 | 通过（3 次） |
| 参考实现耗时 | 30.7 ms |
| 候选实现耗时 | 9.84 ms |
| 该任务加速比 | 3.12× |
| 计时 | 20 次 CUDA Event 测量 |
| 实际评测文件 | SHA-256 `8acc4ec88b8cb56d5516775db4b9c7cef1b516487b70b2b1da4114a2fd8ecde2` |

要复现这个例子，先用上面的命令创建工作区，再把示例代码复制到工作区的 `solution/kernel.py`。将 `config.toml` 中 `[build]` 下的 `language` 设为 `"triton"`，最后运行 `bash scripts/bench.sh --label "reverse-cumsum"`。

每次评测都会记录正确性、耗时、测试次数和实际评测文件的哈希值。测量方式见[评测说明](docs/evaluation.md)。

## 许可证与致谢

项目采用 [MIT 许可证](LICENSE)。来源与引用信息见[第三方声明](THIRD_PARTY_NOTICES.md)。

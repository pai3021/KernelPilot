# Third-party notices

KernelPilot includes and adapts code from [AKO4X](https://github.com/TongmingLAIC/AKO4X), starting from its public release commit `0fd4b5f`. The original MIT license and copyright notice are preserved in [LICENSE](LICENSE).

KernelBench is a separate benchmark project. This repository does not include its task dataset or GPU environment. Codex and Claude Code are separate coding-agent tools.

The AKO4X B200 / FlashInfer results and tech report belong to the upstream project; they are not KernelPilot evaluation results.

## KernelBench task example

`examples/kernelbench_reverse_cumsum_triton.py` includes the `91_cumsum_reverse` task definition from KernelBench. KernelBench is licensed under MIT:

Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini
Scaling Intelligence Lab, Stanford University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

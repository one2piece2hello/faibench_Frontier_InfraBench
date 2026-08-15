<div align="center">

<h1>Φ-Bench: Can Large Language Models Engineer the Infrastructure That Powers Them?</h1>

*用于评测前沿 LLM 与自主编程智能体解决真实 **ML 系统与 LLM 基础设施工程**问题的 **Frontier AI Infrastructure Benchmark**，亦写作 **FAI-Bench** / **ΦBench**。*

**85 道开源 LLM 基础设施工程任务** —— 构建公开 Docker 镜像，离线解题，并使用随评测包发布的评分器评分。

### 🔗 [faibench.org](https://faibench.org/)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
&nbsp;![tasks](https://img.shields.io/badge/tasks-85-brightgreen.svg)
&nbsp;[![Website](https://img.shields.io/badge/Website-faibench.org-2563EB?logo=googlechrome&logoColor=white)](https://faibench.org/)
&nbsp;[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/faibench-Frontier-Infra-Bench/faibench_Frontier_Infra_Bench)
&nbsp;![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-B31B1B?logo=arxiv&logoColor=white)

###### 🌐&nbsp; [English](README.md) &nbsp;·&nbsp; **简体中文**

</div>

---

## 目录

- [摘要](#摘要)
- [基准概览](#基准概览)
- [目录结构](#目录结构)
- [运行单个任务](#运行单个任务)
- [使用任务运行器](#使用任务运行器)
- [重要约定](#重要约定)
- [可复现性](#可复现性)
- [评测包验证](#评测包验证)
- [许可证](#许可证)

## 摘要

大型语言模型（LLM）已展现出强大的推理与代码生成能力，因而有望参与开发和优化支撑其自身运行的基础设施。然而，现有基准多聚焦于孤立的 GPU 内核、预先定义的算子或预先指定的优化目标，因此尚不能充分评估开放式、长时程的 LLM 基础设施工程能力。在这类任务中，LLM 需要理解并浏览真实代码库，识别性能瓶颈，并反复实现、性能分析、调试和改进解决方案。KernelBench、TritonBench 和 FlashInfer-Bench 等工作为 GPU 内核生成与优化建立了有价值的评测环境，ISO-Bench 和 CUDAHercules 等较新工作则进一步探索了仓库级优化。但这些设置通常仍围绕预定义函数、算子或优化目标展开，对开放式、长时程基础设施工程的覆盖仍然有限。

为弥补这一空白，我们提出 **Φ-Bench（Frontier AI Infrastructure Benchmark）**，用于系统评测 LLM 开发 LLM 基础设施栈的能力。Φ-Bench 源自前沿系统研究中的优化问题，并以真实开源代码库为基础，广泛覆盖 LLM 训练与推理所需的 AI 基础设施。85 道任务分为三种范围逐步扩大的形式：**内核函数补全（Kernel Function Completion, KFC）**、**长时程实现（Long-Horizon Implementation, LHI）** 和 **端到端优化（End-to-End Optimization, E2EO）**。任务范围涵盖局部 CUDA/GPU 内核实现与优化、仓库级开发，以及开放式系统优化。

在前沿 LLM 上开展的大量实验揭示了它们在复杂 LLM infra 工程中的当前能力与局限，并为理解迈向未来 AI infra 自主优化过程中仍需解决的挑战提供了重要启示。

## 基准概览

每道任务都提供一份**自包含的公开 Dockerfile**：通过 `git clone` + `docker build` 即可复现环境，评分面也随评测包一同发布。所有任务均设置为 `allow_internet = false`：**解题和评分均离线进行**，因此所有依赖（包括模型权重和数据集）都会在 `docker build` 阶段写入镜像。

| 子集 | 题数 | 题型 |
|---|---|---|
| `tasks/kfc/` | 55 | 单内核实现与优化 |
| `tasks/lh/` | 20 | 长时程、仓库级开发 |
| `tasks/e2e/` | 10 | 端到端系统优化 |

机器可读索引：[`tasks_index.json`](tasks_index.json)（85 条记录，包含每题的包根、布局、GPU、scope、锚点及 oracle 可用性）。评分公式见 [`SCORING.md`](SCORING.md)。

## 目录结构

```
fai_bench/
├── tasks_index.json            85 题索引
├── SCORING.md                  两类 reward 的公式
├── scripts/verify_package.py   包结构 + Dockerfile 可解析性 + 挖洞树自证 自检
└── tasks/
    ├── kfc/<dir>/task/  ┐
    ├── e2e/<dir>/task/  ├ 三个子集的**包根**位置不同,见 tasks_index.json 的 package_root
    └── lh/<dir>/        ┘  (lh 是平铺,kfc/e2e 多一层 task/)
```

**包根**(即 `docker build` 的上下文)下固定只有这些内容:

```
<package_root>/
├── instruction.md        模型唯一可见的输入
├── task.toml             资源、判分入口、primary_metric、docker_image
├── .dockerignore         必须在上下文根(docker 不读 environment/.dockerignore)
├── environment/
│   ├── Dockerfile        ★ 自包含公共配方(公共 base + 公网源 + 版本写死)
│   ├── repo/             挖洞后的工作树(vendored,75 题有)
│   ├── runtime/          entrypoint.sh / timer.sh / run_dev_bench.sh
│   ├── loop/             会话内自评 harness(76 题有;其中 26 题真跑 1~16 轮,
│   │                     其余 50 题是 kfc,MIN=MAX=1 单次提交 —— 见 SCORING.md)
│   └── …                 submission / dev_bench / stubs / workspace(按题)
├── tests/                判分面:test.sh + compute_reward.py + 工作负载 + 锚点
└── solution/             ★ reviewer-only:参考实现 / oracle 补丁(83 题有)
```

## 运行单个任务

```bash
cd <package_root>
docker buildx build -f environment/Dockerfile -t <task.toml 里的 docker_image> .
docker run --rm [--gpus all] -it <image>                                     # agent 在容器内作答
docker run --rm [--gpus all] -v "$PWD/tests:/tests:ro" <image> bash /tests/test.sh
cat /logs/verifier/reward.json        # reward 字段即该题得分
```

每题 Dockerfile 头部都写全了 **build / run / score / 重标定** 四条可复制命令,以及该题的版本锚点与工作树来源(`PROVENANCE` 段:从哪个镜像 digest 恢复、用什么方法、怎么验证)。

**构建期需要的网络出口**(按题不同,`tasks_index.json` 可对照):全部题需要 apt 与 PyPI;5 题需要 `git clone` 公共 GitHub;5 道 e2e 题需要从 HuggingFace 拉模型权重(~GB)。**必须用 BuildKit**:题包用了 `RUN <cmd> <<'PY' … PY` heredoc,请用 Docker ≥ 23 + buildx(`docker buildx build`),经典 builder 解析不了这种语法。

> 镜像源环境提示:11 题写的是 `pip install --index-url https://download.pytorch.org/whl/...`。`--index-url` 会**替换**主索引,所以如果你的网络只能走内部 PyPI 镜像,这 11 题会因为拿不到 torch 而失败 —— 把它改成 `--extra-index-url` 即可两种网络都兼容(有公网时行为不变)。其余 45 题本来就用的是 `--extra-index-url`,不受影响。

## 使用任务运行器

上面那套手动命令,`scripts/run_task.py` 全给你串好了。它**单文件、只用标准库**(Python ≥ 3.8 都能跑;3.11 以下会自动降级到内置的 TOML 解析,不需要 `tomli`),一条命令完成 build → 起容器 → 调 agent → 收产物 → 挂 `tests/` 判分 → 汇总 reward:

```bash
# 用 claude-code / codex 作答一道题(agent CLI 不在镜像里,运行时注入)
python3 scripts/run_task.py --task tasks/kfc/<dir> --agent claude-code --model claude-opus-5 \
    --agent-bin /path/to/claude          # 或 --agent-install 在容器里装
python3 scripts/run_task.py --task tasks/e2e/<dir> --agent codex --model gpt-5.6 --agent-install

# 不调模型的两条自检通路:
python3 scripts/run_task.py --task tasks/lh/<dir> --agent oracle   # 跑 solution/solve.sh,应得该题参考分
python3 scripts/run_task.py --task tasks/kfc/<dir> --agent none    # 判 pristine 基线,应 ≈ 0

# 抽样 / 全量:
python3 scripts/run_task.py --tasks-root tasks --n-tasks 10 --sample-seed 0 --agent claude-code
```

几个要点:

- **提交契约是"工作树式"**,不是 `git commit`。agent 直接编辑允许的文件、把改动**留在工作树里**即可 —— 判分读的是工作树相对烤入基线 commit 的 diff(`pre_artifacts.sh` 用 `git add -AN` + `git diff HEAD` 捕获,不移动 HEAD)。这样任何"会编辑文件"的 scaffold 都能接,换 claude-code / codex / mini-swe-agent 不用改题。**不要让 agent commit** —— HEAD 一动,scope 闸门和取基线的 `git checkout HEAD -- <scope>` 都会失灵,正确解反被判 0。
- **agent CLI 运行时注入**:开源镜像刻意不装任何 agent(已核实 91/91 个 Dockerfile 零命中)。`--agent-bin <宿主机路径>` 把 CLI 只读挂进去,或 `--agent-install` 在容器里装公共 npm 包。gpt 系模型走 `codex`,其余走 `claude-code`。
- **网络**:所有题 `allow_internet = false`。判分**永远** `--network=none`;agent 步默认也离线,只有需要调模型 API 时才按该 agent 的最小白名单开(`--agent-net proxy --net-proxy URL`,由代理强制白名单)。
- **镜像源环境**:build 阶段加 `--build-network host` 让 RUN 步骤能走宿主机可达的 PyPI/apt 镜像(默认关,公网无需)。
- **产出**:`runs/<task>-<agent>-<model>-<ts>/{build.log,agent.log,verify.log,artifacts/,verifier/,run.json}`,以及一行一题的 `runs/summary.jsonl`。

`--agent oracle` 这条通路顺便就是 runner 的自检:它调用每题的 `solution/solve.sh`,把参考实现按"和 agent 提交一样"的方式(工作树、不 commit)落地,再走正常 candidate 判分,应精确命中该题的参考分。

## 重要约定

**① `tests/` 随包发布,但绝不烤进镜像。** 判分时才挂到 `/tests`。理由:隐藏用例、强基线、标定锚点都在里面,烤进去等于做题时可读可改。`solution/` 同理 —— 它是 reviewer-only,不进镜像、判分不运行。

**② 性能锚点是标定常量,换硬件必须重标。** reward 形如 `min(1, ln(speedup/ref_speedup)/ln(ref_speedup))`(**打平 ref_speedup 得 0,必须超过**;详见 SCORING.md),`ref_speedup` 判分时只读、不重跑 oracle。77 题有锚点,标定条件写在 `tests/ref_speedup.caveat.md` 或 manifest 的 `hardware_caveat` 字段里,同处给出可复制的重标定命令。**标定环境分两条通道**:GPU 题在 NVIDIA H20,CPU 题在作者的 CPU 通道(Intel Sapphire Rapids)—— 各题写的是它自己的真实标定环境,不要当成统一的一个。

```bash
# 补丁形态(多数 kfc / lh):
docker run --rm [--gpus all] -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/patches:ro" \
  -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
  <image> bash /tests/test.sh
# 单文件形态(部分题的参考实现是整份文件的变体,不是补丁):
  -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_FILE=/patches/kernel_oracle.py
```

自证:`noop`(不改动)应 ≈ no-op 值、`negative` 必须得 0。**83 题随包提供参考补丁/oracle**,可直接走这条通路;只有 2 题(`kfc/wro-offload-layer-prefetch-ring-pipeline-loop16`、`kfc/wro-offload-policy-grid-search-loop16`)没有,它们的 caveat 里已注明"锚点换硬件不可比,请自行标定"。

**注意锚点解析链**:`tests/ref_speedup.txt` → 镜像内 `/opt/verifier-correctness-manifest.json` → 1.0。镜像内那份**故意不含真锚点**(它对做题者可读),而 `ref_speedup <= 1` 是 hard gate,所以**必须挂载 `tests/`**,否则会以"锚点无效"大声失败而不是给出错误分数。另外 `tests/ref_speedup.txt` 被 `tr -dc '0-9.'` 解析 —— **不要往里加任何注释**,含数字或小数点的文字会污染锚点值。

## 可复现性

这两个子集的起始实现是"上游库被挖掉一块"的树。它**不是 clone 出来的** —— 原始镜像由预置 tarball 组装,没有记录上游 commit,所以 clone 无法钉到被计分的字节。因此 `environment/repo/` 是**从原始镜像恢复并 vendored** 的,并且是**可自证**的:

> `git apply --check -p1 solution/oracle.patch` 必须能干净**正向**打在 `environment/repo/` 上、**反向**打不上 —— 这证明该树正是参考补丁生成时的那个挖洞基线。`scripts/verify_package.py` 会逐题跑这个门。

同家族多题共用一棵上游树时,树里**全家族**的 scope 文件都处于挖洞态(否则一题的镜像会含另一题的答案);build 期有断言检查这一点。

vendored 树是上游代码的原样字节 —— 里面出现的第三方 URL、示例配置、甚至上游自己提交的内部代理提示,都是上游内容,**按自证门的要求必须保持不变**。

## 评测包验证

```bash
python3 scripts/verify_package.py            # 全量自检(85 题;delete_* 归档题自动跳过)
python3 scripts/verify_package.py kfc lh     # 只查某些子集
```

自检是**只读**的,路径全部从脚本自身位置推导,所以 clone 到任何地方都能直接跑。它查的是:
必备件齐全 · `task.toml` 可解析 · `tests/test.sh` 在且性能题的锚点解析得到(不会静默回落 1.0)·
**Dockerfile 可解析**(heredoc 配对、无悬挂续行、`COPY` 源都在上下文里且没被 `.dockerignore` 排除、
每个 `RUN` 的 shell 体过 `bash -n`)· 锚点与 caveat 自洽 · **挖洞树自证**(`oracle.patch` 必须正向
可打、反向打不上)· 无 `__pycache__` / `*.bak` 之类残留 · **可运行三件套**(每题 `pre_artifacts.sh`
与 `solution/solve.sh` 存在、可执行、`bash -n` 过,`solve.sh` 有四态 CLI,`task.toml` 是
`schema_version="2.0"`)。`tasks_index.json` 随包发布,不需要自行重建。

## 许可证

fai_bench **自写的部分**(题目 `instruction.md`/`task.toml`、判分 `tests/**`、参考解 `solution/**`、
loop16 harness `environment/loop*/**`、runner 与自检 `scripts/**`、文档)采用 **Apache License 2.0**
(见 [`LICENSE`](LICENSE))。

`environment/repo/` 下的 **vendored 上游代码**(nanoGPT / torchtitan / vLLM / llama.cpp /
Megatron-LM / ColossalAI / flash-linear-attention 等)以及 build 期从公开源拉取的模型权重与数据集
(Qwen2.5、all-MiniLM-L6-v2、wikitext 等)**各自保留其原始许可与版权**,不在本仓库的 Apache-2.0
授权范围内 —— 详见 [`NOTICE`](NOTICE)。每棵 vendored 树内自带的 `LICENSE`/`COPYING` 文件为其权威许可。

**覆盖范围：**GPU/CUDA 内核优化、分布式训练、推理与服务（vLLM）、低精度计算与量化、通信与集合操作、检查点与存储、MoE 路由、注意力与状态空间内核。共计 85 道自包含、可通过 Docker 复现的任务，并提供离线评分器和参考解。

**引用格式：**

```bibtex
@misc{faibench2026,
  title  = {$\Phi$-Bench: Can Large Language Models Engineer the Infrastructure That Powers Them?},
  author = {Φ-Bench contributors},
  year   = {2026},
  url    = {https://faibench.org/}
}
```

# ref_speedup 硬件/环境 caveat(开源版)

`ref_speedup.txt` 里的 **28.773835** 是本题的锚点,判分时只读、不重算:

    reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0

`tests/test.sh` 的解析顺序是 **`tests/ref_speedup.txt` → `/opt/verifier-correctness-manifest.json` → 1.0**;
回落到 1.0 会让所有 reward 归 0,所以这个文件必须存在且 > 1。

## 这个数是怎么来的

本题的指标是**确定性计算字节数**(纯 Python 计数,无计时、无噪声),所以锚点不是一个
"测出来的"数,而是一个**算出来的**比值:

    锚点 = naive_traffic / reference_traffic = 425230336 / 14778368 = 28.7738359…

- `naive_traffic` = 冻结基线(每个轴取最小块)在包内语料上的总搬运量 `425230336`;
- `reference_traffic` = **`solution/baseline2.patch`**(「能塞下就取最大块」的贪心参考)在同样语料上的
  总搬运量 `14778368`。

即锚点是"一个强但非最优的参考解"的成绩:超过它才得分(打平 = 0),这正是评分曲线要的口径。

## 🔴 硬约束:锚点必须严格小于该语料上可实现的最大 speedup

包内语料的穷举最优(即 `solution/oracle.patch` 的写法)是 `6537216` 字节 →
最大 speedup = `425230336 / 6537216 = 65.047619`。因此:

- **锚点 ≥ 65.047619 ⇒ 本题对任何提交恒为 0**(打平锚点即 0,而最优解也超不过锚点);
- 历史教训:锚点曾经取成 oracle 自己的值 `63.45721271393643`(写入两位小数后为 `63.46`),而该语料的
  理论最优只有 `63.457213` —— 差 0.0044%,于是**连 oracle 自己都是 0 分**,整道题失去信号。同一轮里
  「每轴取最大可行块」的贪心恰好也落在同一点,说明当时的语料根本没有区分度(见下方 `make_bench_corpus`
  的说明:候选清单被截断到 128/96 才掩盖了 Tm↔Tn 的取舍)。

改语料或改 `baseline2.patch` 之后,**必须**同时改本文件的锚点,并核对上面的不等式。

## 复现(不需要跑计时,也不需要跑 oracle)

```bash
PKG=<this package>/task
cd "$PKG/tests"
# 1) 让 greedy 参考落地到一棵可写的树(或直接用求解器自己的实现)
# 2) 量两边:
KB_REPO_DIR=<greedy 那棵树> KB_BASELINE_MODULE=<冻结基线 tile_planner.py> \
  python3 cpu_bench.py candidate        # 期望 TRAFFIC=14778368 VALID=1
KB_REPO_DIR=<冻结基线那棵树> KB_BASELINE_MODULE=<同一个文件> \
  python3 cpu_bench.py baseline         # 期望 TRAFFIC=425230336 VALID=1
# 3) 锚点 = 两者之比:
python3 -c 'print(425230336 / 14778368)'   # 28.7738359…
```

`cpu_bench.py verify` 会打印 `naive_traffic` 与 `reference_best_traffic`(穷举最优),可用来核对不等式。

## 标定条件

- 硬件:**与硬件无关**。本题判分路径全是纯 Python 标准库(不 import torch/numpy),
  指标是从返回的 plan 推出来的**计算字节数**,不是墙钟计时、不是显存占用,所以换机器/换线程数不改变锚点
  (原始私有基座声明 `torch 2.7.0` / `OMP-...=1`,对本指标没有影响)。
- 自证:`noop`(不打补丁)应 = 1.0 → 被门到 0;`negative`(`KERNELBENCH_NEGATIVE_PATCH=`)必须 0;
  `oracle`(穷举最优)应给出 **略高于锚点**的 speedup(≈ 65.047619),reward 在 (0, 1) 之间 ——
  这是本题"确实有超过锚点的空间"的证明。

参考实现见本包 `solution/`(`oracle.patch` = 穷举最优 / `baseline2.patch` = 锚点所用的贪心参考 /
`negative.patch` = 已知坏例);三个补丁都是 reviewer-only,已被 `.dockerignore` 排除,不进镜像。

# 评测流水线的端到端吞吐（LLM 评测与压测）

你会拿到一份完整的 **`EleutherAI/lm-evaluation-harness`** 检出（`/app/repo`，可 import 且 editable
安装，改它即生效），运行环境是 **CPU（8 核）**，**无外网**、**无 GPU**（本题被评的这段流水线是模型
推理**之后**的纯 CPU 打分/聚合，镜像里没有 torch）。

一个 LLM 评测框架跑完模型推理之后，还有一段常被忽视却真实耗时的**打分/聚合流水线**：正则答案抽取、
`take_first` / `majority_vote`（自洽多数投票）这类响应变换、多选题的 loglikelihood 取 argmax、以及
`exact_match` / `contains` / `prefix_match` 等指标计算（对应 `lm_eval/api/task.py` 里的
`apply_filters` 与 metric 阶段）。这道题就评这一段：给定一批**已经带好模型输出**的评测记录（生成文本
或每个选项的 loglikelihood），把它们**尽可能快**地打成每条样本的分数。

正式评分先逐样本检查分数和 ID；跳过样本、丢 ID、近似打分或触发反缓存等硬性检查均得 0 分。
通过后，强基线与候选交替计时至少 5 对（AB–BA），取每对 `基线时间 / 候选时间` 的中位数为 `s`。
设 `r` 为参考解相对同一基线的标定加速比。若 `r ≤ 1` 或 `s ≤ r`，得 0 分；
否则得 `min(1, ln(s/r) / ln(r))`。追平参考解是 0 分，达到 `r²` 才是满分。

## 提交契约

把评分所需的一切持久化到 **`/app/submission/`**：`scoring_pipeline.py`，必须暴露

```
load_scoring_pipeline_for_verification(device) -> 对象，具备
    .score(samples: list[dict]) -> list[dict]        # 每行输出 {"id": <样本 id>, "score": <float>}
```

评测器把它**自己的**留出记录集喂给 `.score()`，与强基线**交替计时**（ABBA 配对，≥5 对，取中位数），
并用一套**独立的**参考实现重算每条样本的分数。你的分数由 `speedup = 强基线时间 / 你的时间` 决定
（有界，0.0–1.0），但**仅当每条样本分数与参考完全一致**（`|cand - ref| <= 容差`，且不漏 id、不跳样本）
才成立。此外有一道**反缓存探针**：把一部分
留出记录扰动后换上全新 id 重新计分——从 dev 跑里拷来的"id→分数"查表在这里会算错，同样判 0。

## 记录 schema（dev 与留出集一致）

- `metric` = `"exact_match"` | `"contains"` | `"prefix_match"` | `"loglikelihood_acc"`
- `filter` = `"take_first"` | `"majority_vote"`（仅生成类指标）
- `filter_pattern` = 可选正则（有捕获组取 `group(1)`，否则取 `group(0)`，无匹配取 `""`）
- `response` = str 或 list[str]（生成类）；`gold` = str（生成类）
- `choice_loglikelihoods` = list[float]；`gold_index` = int（多选题）

参考语义（你的分数必须逐条复现）：`normalise(text)` = 折叠空白、strip、小写；`take_first` 取第一个
候选；`majority_vote` 取归一化后出现最多的候选，平票取字典序最小；`loglikelihood_acc` 取 argmax（平票
取最小下标）与 `gold_index` 比较；`exact_match` / `contains` / `prefix_match` 按归一化文本判定。

## 设计空间

你可以修改 `/app/repo` 里的任何东西，也可以完全另写自己的打分器。提速的来源都在明面上：把正则抽取与
指标计算**向量化**（按列批处理、正则只编译一次、按 metric/filter 分组走紧凑循环）、消除逐行 Python 开销、
缓存已编译正则/分词、借鉴 lm-evaluation-harness auto-batch 清缓存的思路（PR #3654），或直接在 `/app/repo`
里重写打分路径。查阅公开文档与源码均允许。起步模板是一个**正确但慢**的逐行循环，能过一致性门，但吞吐上
留着很大空间——去把它顶上去。

## 数据

`/data/eval_harness/dev_samples.jsonl` 是与留出集**同分布但不相交**的开发集（同一 schema），用来自测正确性
与自测速度。公开记录只用于比较方案；正式评分使用独立的留出记录和强基线。

## 规则

- 评测器**亲自**用它的留出记录跑你的 `.score()`、**亲自**计时、并用**自己的**实现重算参考分数——任何
  "id→分数"缓存表都没用（留出 id 与 gold 从不可见，且反缓存探针会扰动输入让查表算错）。
- **一致性门是硬门**：任一样本分数不符、漏 id、或跳样本 → 直接 0。你不能拿正确性换速度。
- 评分是 **eval-only**：加载你提交的打分器并计时评测，不重跑训练/建库。请在你的时间预算内迭代改进你的打分器。
- **无外网**；依赖已预装。**不要**读取或篡改评测器、留出记录/标注或任何受保护文件；这样做直接判 0。


---

## How you are graded

In the default loop16 image, run `bash /opt/loop/submit.sh` once per candidate (1–16 submissions).
To stop before round 16, run `bash /opt/loop/submit.sh --finalize --reason "..."`; round 16 finalizes automatically.
Among candidates that pass the dev checks, the one with the highest public dev speedup against the
naive template goes to final evaluation. The grader checks its per-sample outputs on held-out records
and times it against the **strong** baseline. The public speedup is not the reward.

If you run the base image without the submit helper, the grader uses the final working tree instead.
Leave edits uncommitted and keep `/app/repo` at its initial commit.

Start from the shipped `scoring_pipeline_template.py` — a correct-but-slow row-by-row loop that
passes the consistency gate. Copy it to `/app/submission/scoring_pipeline.py` and optimise; you may
also edit `/app/repo` and import your rewritten scoring path from it.

Check yourself against the PUBLIC dev split at `/data/eval_harness/dev_samples.jsonl` (same schema as
the held-out set, disjoint from it): run your scorer over it, confirm every per-sample score matches
a first-principles reference, and time it. 🔴 **That dev number is a PUBLIC-split proxy, NOT your
graded score** — the graded run uses a HELD-OUT set with a stronger baseline and an anti-cache probe,
so use the dev result only for *direction* (did this change help or hurt?) and do **not** tune
specifically for the public cases.

Solve it yourself. The lm-evaluation-harness sources at `/app/repo`, the public dev split, the public
literature and your own ideas are all fair game — what is forbidden is touching the measurement.

### 🔴 Red line (hard — ANY one of these ⇒ the whole task scores 0 in audit)

The scoring machinery is protected and off-limits. Solve this task **independently**: rely ONLY on
the code in `/app/repo`, the public dev split under `/data/eval_harness`, and your own knowledge and
reasoning.

1. Do **not** deliberately `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, read or edit ANY
   verifier / correctness / timing / scoring / hidden-test / held-out file or directory, wherever it
   lives — do not run or reverse-engineer the scoring machinery.
2. Do **not** run the verifier directly, and do **not** reproduce or reverse-engineer it.
3. Do **not** search for, print or infer the hidden held-out samples, their gold, seeds,
   thresholds, the metric name, or the reference speed-up from any source.
4. Do **not** fetch or look up an upstream implementation or a reference solution, do **not** access
   the internet at runtime, and do **not** attempt to bypass the proxy isolation (no web search, no
   `git` remote fetch), whether internet is on or off.
5. Do **not** use, quote or condition your code on any verifier / scoring / source material, however
   you came to see it — and do **not** hard-code a copied (id -> score) table: the graded run uses a
   held-out set with an anti-cache probe that perturbs inputs, so a lookup table scores 0.
6. Do **not** disable, kill or tamper with the timer or the measurement path.

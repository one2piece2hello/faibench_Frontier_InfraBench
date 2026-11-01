# 评测流水线的端到端吞吐（LLM 评测与压测）— loop16 协议

你会拿到一份完整的 **`EleutherAI/lm-evaluation-harness`** 检出（`/app/repo`，可 import 且 editable
安装，改它即生效），运行环境是 **CPU（8 核）**、**无外网**、**无 GPU**（本题被评的这段流水线是模型
推理**之后**的纯 CPU 打分/聚合，镜像里没有 torch）。

一个 LLM 评测框架跑完模型推理之后，还有一段常被忽视却真实耗时的**打分/聚合流水线**：正则答案抽取、
`take_first` / `majority_vote` 这类响应变换、多选题的 loglikelihood 取 argmax、以及
`exact_match` / `contains` / `prefix_match` 等指标计算（对应 `lm_eval/api/task.py` 里的
`apply_filters` 与 metric 阶段）。这道题就评这一段：给定一批**已经带好模型输出**的评测记录，把它们
**尽可能快**地打成每条样本的分数。

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

你可以修改 `/app/repo` 里的任何东西，也可以完全另写自己的打分器。把改动**留在工作树里**即可，不需要 `git commit`。

## 记录 schema（dev 与留出集一致）

- `metric` = `"exact_match"` | `"contains"` | `"prefix_match"` | `"loglikelihood_acc"`
- `filter` = `"take_first"` | `"majority_vote"`（仅生成类指标）
- `filter_pattern` = 可选正则（有捕获组取 `group(1)`，否则取 `group(0)`，无匹配取 `""`）
- `response` = str 或 list[str]（生成类）；`gold` = str（生成类）
- `choice_loglikelihoods` = list[float]；`gold_index` = int（多选题）

参考语义（你的分数必须逐条复现）：`normalise(text)` = 折叠空白、strip、小写；`take_first` 取第一个候选；
`majority_vote` 取归一化后出现最多的候选，平票取字典序最小；`loglikelihood_acc` 取 argmax（平票取最小下标）与
`gold_index` 比较；`exact_match` / `contains` / `prefix_match` 按归一化文本判定。

## 数据

`/data/eval_harness/dev_samples.jsonl` 是与留出集**同分布但不相交**的开发集（同一 schema）。

---

## 评分方式（loop16：会话内可多次自评迭代）

本题是 **loop16 协议**：你可以在会话内多次给当前解打分并拿反馈，**最多 16 次、至少 1 次，你自己决定何时停**
（不必凑满 16）。

### 1. 每一轮
改完 `/app/submission/scoring_pipeline.py`（以及可选的 `/app/repo` 改动）后，运行

```
bash /opt/loop/submit.sh
```

它会在**公开、且与留出集不相交的** dev 集（`/data/eval_harness/dev_samples.jsonl`）上：用一套**独立实现**
重算每条样本的参考分数，跑**焊死的逐样本一致性门 + 反缓存探针**（与判分同一套硬门），再把你的 `.score()`
与镜像里那份**公开的朴素起步模板**做 **ABBA 配对计时**，回给你：本次 correctness 是否通过、一个相对朴素模板的
**dev_speedup**、best_so_far、剩余次数。

🔴 **那个 dev_speedup 是公开代理，不是你的计分分数**：判分用评测器的**留出**记录集 + **强基线** ABBA + 标定
锚点，容器里既没有留出集，也没有强基线，也没有锚点。dev 代理分母是**公开朴素模板**（不是隐藏强基线），量纲
与判分不同，但两者都随你的吞吐单调变化，所以只把 dev_speedup 当作**方向**判断，**不要**针对公开用例过拟合。

### 2. 定稿
满意后（或到第 16 次自动定稿）运行

```
bash /opt/loop/submit.sh --finalize --reason "<一句话说明你为何停>"
```

`--reason` 必填且会被审计。被计分的是你**最好的一次**（best-of-k，按 dev_speedup 选出并植入判分树），不是
最后一次。在 k=1 就有充分理由地停，与烧完 16 次得分完全一样；**没有**最小次数、**没有**"连续几轮无提升才能停"的规则。

### 🔴 红线（硬性 —— 命中任一 ⇒ 全题审计判 0）

1. 不得**故意** `cat`/`open`/`less`/`head`/`tail`/`grep`/`find`/编辑**任何** verifier / correctness / timing /
   scoring / hidden-test / 留出文件或目录，无论它在哪 —— `bash /opt/loop/submit.sh` 是与判分机制交互的**唯一**方式。
2. 不得直接运行评测器，不得复现/逆向它。
3. 不得搜索、打印或推断隐藏留出样本、gold、seed、阈值、指标名或参考加速比。
4. 不得取上游实现或参考解（禁 web search、禁 git remote fetch、禁联网）；已在你知识里的公开文档与源码允许。
5. 不得使用、引用或让代码依赖任何 verifier / scoring / 留出材料，也**不得**硬编码"id→分数"查表（反缓存探针会判 0）。
6. 不得禁用、杀死或篡改计时器或测量路径。

只待在 `/app/repo`、`/app/submission` 和你自己创建的临时目录里，用 `submit.sh` 返回的信号作为唯一评分依据。

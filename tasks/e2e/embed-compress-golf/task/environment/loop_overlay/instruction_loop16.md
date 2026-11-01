# 极致压缩下的向量检索质量（Embedding 压缩金 + 两段精排）— loop16 协议

你会拿到一份完整的 **`UKPLab/sentence-transformers`** 检出（`/app/repo`，可 import），以及冻结在
`/opt/models/all-MiniLM-L6-v2` 的 384 维基础嵌入模型。运行环境是 **CPU（8 核）**，**无 GPU**，**无外网**。
（本镜像的 torch 无法初始化 CUDA，`torch.cuda.is_available()` 恒为 `False`；请按 CPU 预算安排编码与精排的开销。）

**你的目标**：在一个**极小**的"每条向量字节预算"下，让留出检索集上的 **nDCG@10 尽可能高**。

🔴 **硬约束（评测器实测，超一个字节即判 0）**：**每条向量最多 64 字节**，按 `dim × dtype.itemsize` 计算。
384 维 float32（1536 字节）**和** 384 维 int8（384 字节）**都放不下**——你必须**重度压缩**：符号二值化后
按位打包、乘积量化（PQ）码、或维度极少的 int8/fp16（例如 64 维 int8、32 维 float16）。

正式评分使用同一字节预算下的留出集质量比 `s = 候选 nDCG@10 / 强压缩基线 nDCG@10`，
不按运行时间计分。设 `r` 为参考解相对同一基线的标定比值。未通过硬性检查、`r ≤ 1`
或 `s ≤ r` 时得 0 分（仅超过基线还不够）；否则得 `min(1, ln(s/r) / ln(r))`。

## 两段式：先召回，后精排（可选）

- **必需** `load_encoder_for_verification(device)` —— 返回带 `.encode(texts, ...) -> np.ndarray[n, dim]`
  的对象，产出**压缩向量**（字节预算对其生效）。
- **可选** `load_refiner_for_verification(device)` —— 返回带
  `.rescore(query_text: str, doc_texts: list[str]) -> list[float]` 的对象，对第一段余弦检索得到的短名单重排序。

## 提交契约

把评分所需的一切持久化到 **`/app/submission/`**：`submission_encoder.py`，必须暴露
`load_encoder_for_verification(device)`，可选暴露 `load_refiner_for_verification(device)`。
`.encode` 每次调用返回相同的 `dim` 与 dtype（查询与语料共享同一压缩空间）。修改 `/app/repo`、写自己的
编码/精排代码、微调基础模型都允许。把改动留在工作树中，不要 `git commit`；按下文的 loop16 协议提交。

## 数据

`/data/retrieval/` 下提供与评测集**同分布但不相交**的开发（dev）集作为进度监控。

---

## 评分方式（loop16：会话内可多次自评迭代）

本题是 **loop16 协议**：你可以在会话内多次给当前解打分并拿反馈，**最多 16 次、至少 1 次，你自己决定何时停**
（不必凑满 16）。

### 1. 每一轮
改完 `/app/submission/submission_encoder.py`（以及可选的 `/app/repo` 改动）后，运行

```
bash /opt/loop/submit.sh
```

它会用**本题自己的评测流水线**在**公开、且与留出集不相交的** dev 集（`/data/retrieval/dev_*`）上跑
完整两段管线（压缩编码 → 第一段余弦召回 → 可选 refiner 重排 → nDCG@10），并回给你：本次
correctness 是否通过（字节预算 / 反塌缩 / 维度匹配等硬门）、一个 **dev nDCG@10**、best_so_far、剩余次数。

🔴 **那个 dev nDCG@10 是公开代理，不是你的计分分数**：计分用评测器新鲜编码的**留出**集 + 标定锚点，容器里没有
留出数据也没有锚点。只把 dev 数当作**方向**判断，**不要**针对公开用例过拟合。

### 2. 定稿
提交 1–15 次后可用以下命令提前结束；第 16 次会自动定稿。

```
bash /opt/loop/submit.sh --finalize --reason "<一句话说明你为何停>"
```

提前结束时必须填写 `--reason`。通过开发集检查的候选中，dev nDCG@10 最高的一版进入留出集正式评测。
提交次数本身不计分；第 1 次后即可结束，后续提交只有在改变入选版本时才可能影响结果。

### 🔴 红线（硬性 —— 命中任一 ⇒ 全题审计判 0）

1. 不得**故意** `cat`/`open`/`less`/`head`/`tail`/`grep`/`find`/编辑**任何** verifier / scoring / hidden-test /
   留出文件或目录，无论它在哪 —— `bash /opt/loop/submit.sh` 是与判分机制交互的**唯一**方式。
2. 不得直接运行评测器，不得复现/逆向它。
3. 不得搜索、打印或推断留出语料/查询/标注、隐藏阈值、指标细节、强基线 nDCG 或参考分数。
4. 不得取上游实现或参考解（禁 web search、禁 git remote fetch、禁联网）；已在你知识里的公开文档与方法允许。
5. 不得使用、引用或让代码依赖任何 verifier / scoring / 留出材料，无论你是怎么看到它的。

只待在 `/app/repo`、`/app/submission` 和你自己创建的临时目录里，用 `submit.sh` 返回的信号作为唯一评分依据。

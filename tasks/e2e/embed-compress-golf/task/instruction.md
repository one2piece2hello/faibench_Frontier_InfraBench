# 极致压缩下的向量检索质量（Embedding 压缩金 + 两段精排）

你会拿到一份完整的 **`UKPLab/sentence-transformers`** 检出（`/app/repo`，可 import），以及冻结在
`/opt/models/all-MiniLM-L6-v2` 的 384 维基础嵌入模型。运行环境是 **CPU（8 核）**，**无 GPU**，**无外网**。
（本镜像的 torch 是 `2.13.0+cu130`，而集群驱动只到 CUDA 12.4，实测无法初始化 CUDA ——
所以**不要**写 CUDA 专用代码路径，`torch.cuda.is_available()` 恒为 `False`；请按 CPU 预算安排编码与精排的开销。）

**你的目标**：在一个**极小**的"每条向量字节预算"下，让留出检索集上的 **nDCG@10 尽可能高**。

🔴 **硬约束（评测器实测，超一个字节即判 0）**：**每条向量最多 64 字节**，按 `dim × dtype.itemsize` 计算。
所以 384 维 float32（1536 字节）**和** 384 维 int8（384 字节）**都放不下**——你必须**重度压缩**：符号二值化后
按位打包（bit-packing）、乘积量化（PQ）码、或维度极少的 int8/fp16（例如 64 维 int8 = 64 B、32 维 float16 = 64 B）。
正式评分使用同一字节预算下的留出集质量比 `s = 候选 nDCG@10 / 强压缩基线 nDCG@10`，
不按运行时间计分。设 `r` 为参考解相对同一基线的标定比值。未通过硬性检查、`r ≤ 1`
或 `s ≤ r` 时得 0 分（仅超过基线还不够）；否则得 `min(1, ln(s/r) / ln(r))`。

## 两段式：先召回，后精排（可选）

这道题提供一个**可选的两段管线**。你可以提交两个入口：

- **必需** `load_encoder_for_verification(device)` —— 返回带 `.encode(texts, ...) -> np.ndarray[n, dim]`
  的对象，产出**压缩向量**（字节预算对其生效）。
- **可选** `load_refiner_for_verification(device)` —— 返回带
  `.rescore(query_text: str, doc_texts: list[str]) -> list[float]` 的对象。评测器用你的压缩向量做完
  第一段余弦检索、得到每个查询的 top-N **候选短名单**后，会调用你的 refiner 对该短名单**重排序**。

refiner 只在**评测器选定大小的有界短名单**上运行（因此它无法退化成对全库的第二次全量检索），并且总调用次数
有上限。一个好的 refiner（例如对短名单做全精度重打分、或用 cross-encoder）能把压缩损失的质量找回来。评测器
在**精排之后**计算 nDCG@10。

## 设计空间与提示

压缩几何很关键：把 384 个符号位打包成 uint8 字节后，**在这些字节上直接做余弦并不等于在符号位上做 Hamming**——
朴素打包在余弦检索下表现很差。更强的做法是在字节预算内解包成 ±1 浮点、或使用余弦几何有意义的 PQ 码，再配合精排。
可查阅 `sentence_transformers.util.quantization`（`quantize_embeddings` 支持 `'binary'/'ubinary'`）与公开文档。
修改 `/app/repo`、写自己的编码/精排代码、微调基础模型都允许。

## 数据

`/data/retrieval/` 下提供与评测集**同分布但不相交**的开发（dev）集（`dev_corpus.jsonl`、`dev_queries.jsonl`、
`dev_qrels.json`）作为进度监控。dev nDCG@10 用于比较方案；正式分数来自独立的留出集。

## 提交契约

把评分所需的一切持久化到 **`/app/submission/`**：`submission_encoder.py`，必须暴露
`load_encoder_for_verification(device)`（返回压缩编码器），可选暴露 `load_refiner_for_verification(device)`。
`.encode` 每次调用返回相同的 `dim` 与 dtype（查询与语料共享同一压缩空间）。

## 规则

- 评测器**从你实际产出的数组重新测量**每条向量的字节数（`维度 × dtype 的 itemsize`），超预算即判 0；不信任你打印
  的任何数字。
- 你的编码器必须产出**真实、有区分度的向量**（塌缩成常量或少数几个向量会被判退化而判 0）。
- 评测器**亲自**编码它自己的留出语料/查询、**亲自**做第一段检索、**亲自**编排精排并计算 nDCG——任何"查询→文档"
  缓存表都没用，因为留出查询由评测器重新编码、标注从不可见。
- 评分是 **eval-only**：加载你提交的编码器/精排器并评测，不重跑训练。
- **无外网**；依赖已预装。**不要**读取或篡改评测器、留出数据或任何受保护文件；这样做直接判 0。

---

## 评分方式

默认 loop16 镜像中，每个候选运行一次 `bash /opt/loop/submit.sh`，总共 1–16 次。
提前结束时运行 `bash /opt/loop/submit.sh --finalize --reason "停止原因"`；第 16 次会自动结束。
在通过开发集检查的候选中，选择公开 dev nDCG@10 最高的一版。只对选中版本进行一次正式评测：
在留出集上重新编码、召回并按需精排，用留出 nDCG@10 计算最终奖励。若使用没有提交脚本的基础镜像，则评测最终工作树。

你可以自己用 `/data/retrieval/dev_*`（与留出集同分布但不相交的公开 dev 集）自测：跑一遍完整两段管线
（压缩编码 → 第一段余弦召回 → 可选 refiner 重排 → nDCG@10），把 `dev nDCG@10` 当作**方向**判断。正式评测使用重新编码的留出集，不按运行耗时计分。

### 🔴 红线（硬性 —— 命中任一 ⇒ 全题审计判 0）

1. 不得**故意** `cat`/`open`/`less`/`head`/`tail`/`grep`/`find`/编辑**任何** verifier / scoring / hidden-test /
   留出文件或目录，无论它在哪。
2. 不得直接运行评测器，不得复现/逆向它。
3. 不得搜索、打印或推断留出语料/查询/标注、隐藏阈值、指标细节或参考分数。
4. 不得取上游实现或参考解（禁 web search、禁 git remote fetch、禁联网）；已在你知识里的公开文档与方法允许。
5. 不得使用、引用或让代码依赖任何 verifier / scoring / 留出材料，无论你是怎么看到它的。

只待在 `/app/repo`、`/app/submission` 和你自己创建的临时目录里。

请勿 `git commit`；`/app/repo` 的 HEAD 应保持在初始提交。

# wro-sbert-paraphrase-mining — loop16 (restore production speed)

Restore production-speed all-pairs paraphrase mining (sentence-transformers). mine_paraphrases(embeddings, top_k, max_pairs) must return the top-similarity distinct [score,i,j] pairs (i<j, sorted). The shipped baseline uses triple-nested Python similarity + repeated per-row max-scan + Python dedup; make it fast while producing byte-identical output.

- **Editable scope (only):** `paraphrase_miner.py` in `/app/repo`.
- **Goal:** make the scope function faster on the benchmark workload while producing EXACTLY the same observable output (correctness is gated against an independent reference; a wrong result scores 0).
- **Scoring — ONE single submission:** you get **exactly one** graded submission, made with `bash /opt/loop/submit.sh`. **Submitting ends the task**: that single call scores your current `paraphrase_miner.py`, records it as your final answer and finalizes immediately — later calls do not score a new version. So do all your work first: take as long as you need and test yourself as much as you like (your own scratch benchmarks and your own correctness checks, in a scratch directory you create) until you are confident the output is byte-identical to the reference on every case you can think of AND your own timing shows a real improvement. A fast-but-wrong submission scores 0 and cannot be repaired. Explain your approach and your reasoning in writing, for a beginner, before you submit.
- **Final reward:** Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference implementation's speedup over the same baseline. A failed correctness or other hard check, `r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`: matching the reference earns 0 and `s >= r²` earns 1.
- **Do not** edit files outside the declared scope, clone/network, or hardcode outputs.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。

# wro-sbert-paraphrase-mining — loop16 (restore production speed)

Restore production-speed all-pairs paraphrase mining (sentence-transformers). mine_paraphrases(embeddings, top_k, max_pairs) must return the top-similarity distinct [score,i,j] pairs (i<j, sorted). The shipped baseline uses triple-nested Python similarity + repeated per-row max-scan + Python dedup; make it fast while producing byte-identical output.

- **Editable scope (only):** `paraphrase_miner.py` in `/app/repo`.
- **Goal:** make the scope function faster on the benchmark workload while producing EXACTLY the same observable output (correctness is gated against an independent reference; a wrong result scores 0).
- **Scoring — ONE single submission:** you get **exactly one** graded submission, made with `bash /opt/loop/submit.sh`. **Submitting ends the task**: that single call scores your current `paraphrase_miner.py`, records it as your final answer and finalizes immediately — there is no second attempt, no feedback to iterate on, and a second call is refused. So do all your work first: take as long as you need and test yourself as much as you like (your own scratch benchmarks and your own correctness checks, in a scratch directory you create) until you are confident the output is byte-identical to the reference on every case you can think of AND your own timing shows a real improvement. A fast-but-wrong submission scores 0 and cannot be repaired. Explain your approach and your reasoning in writing, for a beginner, before you submit.
- **Final reward:** Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference implementation's speedup over the same baseline. A failed correctness or other hard check, `r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`: matching the reference earns 0 and `s >= r²` earns 1.
- **Do not** edit files outside the declared scope, clone/network, or hardcode outputs.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。

# ref_speedup 硬件/环境 caveat(开源版新增,不改变判分逻辑)

`ref_speedup.txt` 里的 **1.66973** 是 oracle 标定常量,判分时只读、不重跑 oracle:

    reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0

`tests/test.sh` 的解析顺序是 **`tests/ref_speedup.txt` → `/opt/verifier-correctness-manifest.json` → 1.0**;
回落到 1.0 会让所有 reward 归 0,所以这个文件必须存在且 > 1。

## 标定条件

- 硬件:**the authoring CPU lane (Intel Sapphire Rapids)**
- 运行栈:**Python 3.11 标准库**(本题不用 torch / numpy,判分路径全是纯 Python)
- 线程:`OMP/MKL/OPENBLAS_NUM_THREADS=1`(已写进镜像 ENV)
- 度量:`proxy_bottleneck_bytes` —— 是**确定性计数**而不是墙钟计时,所以这个锚点对硬件的敏感度远低于计时型锚点,
  但仍与解释器版本绑定。

## 硬件/环境不一致时如何重标

verifier 原生支持模式分派,**按文件路径**传入参考实现(从不烤进镜像):

```bash
docker run --rm -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/patches:ro" \
  -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_FILE=/patches/kernel_oracle.py \
  fai/kfc-wre-diloco-comm-chunk-balance-loop1:oss bash /tests/test.sh
# 从 /logs/verifier/ 读实测 speedup,写回 tests/ref_speedup.txt
```

自证:`noop`(不覆盖 submission)应 ≈1.0 → 被门到 0;
`negative`(`-e KERNELBENCH_VERIFY_MODE=negative -e KERNELBENCH_NEGATIVE_FILE=/patches/kernel_negative.py`)必须 0。

✅ 本题**随包提供**参考实现的单文件变体(`solution/kernel_oracle.py`),所以重标定通路是**可直接执行**的。

## 闸门必须包含 oracle 本身

该题的指标是**确定性字节总量**（最优 contiguous min-max partition 的瓶颈），oracle 是**可证最优解**，
因此任何提交的绝对 speedup（= vs_oracle × ref_speedup）都不会超过 ref_speedup：

- 旧判分要求 `speedup > ref_speedup`，这一条**没有任何提交能满足（含 oracle）**，恒 0；
- 现判分改为 `speedup > 1`（越过基线）才计分，`reward = 0.5 · ln(speedup)/ln(ref_speedup)`：
  oracle（speedup = ref_speedup）得 **0.5**，oracle² 封顶 1.0，朴素基线（speedup ≤ 1）得 0。
- 不变式：确定性指标的锚点不能当"必须超过"的门槛，只能当 ln 标尺。

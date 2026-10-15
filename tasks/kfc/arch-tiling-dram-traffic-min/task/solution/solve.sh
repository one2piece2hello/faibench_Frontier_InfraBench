#!/usr/bin/env bash
# solution/solve.sh — kfc/arch-tiling-dram-traffic-min
# 把本题的参考实现落成「与 agent 提交等价」的形态,让 oracle 验证从手工传环境变量
# 变成一条命令。本文件不进镜像(没有任何 COPY 取 solution/),判分器也不会运行它。
#
# oracle 形态 : 补丁式(patch)。tests/test.sh 的模式分派走 KERNELBENCH_ORACLE_PATCH 槽位:
#               git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH"
#               产物(reviewer-only;本脚本只读,一个字节都不改):
#                 solution/baseline2.patch, solution/negative.patch, solution/oracle.patch
# $REPO       : /app/repo                      (tests/test.sh: REPO=/app/repo)
# 被计分单元  : /app/repo/tile_planner.py
#               来源:tests/test.sh 的 CAND_FILE="$REPO/tile_planner.py"
# 落地方式    : git -C $REPO apply -p1 <patch> —— 只改工作树,不 commit
#
# 🔴 为什么绝不 commit(规范 H3):判分读**工作树** —— `git status --porcelain` 是 scope
#    硬闸门,`git checkout -q HEAD -- <scope>` 取基线,且镜像自证 /app/repo 的 HEAD 恰好
#    只有 1 个 commit(那就是挖洞基线)。一旦 commit,HEAD 前移 → 闸门看不到改动、
#    `checkout HEAD -- scope` 把 candidate 当成基线 → 正确解被判 0。
#    本脚本落地后会自己断言 HEAD 与 commit 数没变。
#
# ── 落地后判分的那条完整命令(同一个容器里跑完 3 步)────────────────────────
#   PKG=<fai_bench>/tasks/kfc/arch-tiling-dram-traffic-min/task
#   IMG=fai/kfc-arch-tiling-dram-traffic-min:oss
#   # 1) 起容器。判分容器始终 --network=none;tests/ 和 solution/ 都**没有**烤进镜像
#   #    (Dockerfile 里没有一条 COPY 取它们),所以这两个目录必须挂进来。
#   #    该题 task.toml 声明 gpus=0;锚点的标定硬件/运行栈见 tests/ref_speedup.caveat.md
#   docker run -d --name kfc-solve --network=none \
#     -v "$PKG/tests":/tests -v "$PKG/solution":/solution "$IMG"
#   # 2) 落地参考实现到工作树(本脚本;不 commit)
#   docker exec kfc-solve bash /solution/solve.sh
#   # 3) 判分。KERNELBENCH_VERIFY_MODE 必须留在默认的 candidate —— 这就是 agent 的
#   #    真实评分路径;设成 oracle/negative 会让 test.sh 自己再落地一遍,绕过本脚本。
#   docker exec -e KERNELBENCH_VERIFY_MODE=candidate kfc-solve bash /tests/test.sh
#   docker exec kfc-solve cat /logs/verifier/reward.json   # 判分结果
#   docker rm -f kfc-solve
#
#   期望(本题锚点取自 baseline2 的贪心参考 = 28.773835,不是 oracle):
#         oracle → 穷举最优 speedup ≈ 65.047619 高于锚点 ⇒ reward > 0(约 0.24,见
#         tests/ref_speedup.caveat.md);"参考解跑对了"的判据是 correctness 全过 +
#         hard_fail_reasons 为空 + speedup 明显高于锚点(实现类二值题 = 1.0 —— 权威公式是
#         tests/test.sh 写出的 reward_formula 字段);
#         --baseline2 → ≈ 0(它就是锚点本身,打平不得分,必须超过);--negative → 0;
#         --noop(在纯净容器里)→ 0(挖洞基线过不了正确性门)。
#   本题带 loop16 harness:`bash /opt/loop/submit.sh` 判的也是同一棵工作树、
#   同样走 candidate 模式(口径一致),所以落地后直接自评也成立。
# ──────────────────────────────────────────────────────────────────────────
#
# 用法(规范 2 的统一 CLI):
#   bash solution/solve.sh              # 落地 oracle(参考实现)
#   bash solution/solve.sh --negative   # 落地 negative(已知坏例,必须得 0)
#   bash solution/solve.sh --baseline2  # 落地 baseline2(中间基线)
#   bash solution/solve.sh --noop       # 把可编辑面复位到烤入基线(对照组)
#
# --noop 语义:把可编辑面复位到烤入基线(与 verifier 取基线的方式一致 ——
#   `git checkout HEAD -- <scope>`,HEAD 不动),使对照组能量到真正的 no-op 值。
#   runner 在同一个容器里连着跑 oracle → noop 做对照时,靠它把上一轮落地归零。
#
# 幂等:每次落地前先把被计分单元 `git checkout HEAD --` 复位,重复执行结果一致。
# 失败一律非 0 退出并打印原因(补丁打不上 / 变体文件缺失 / scope 路径不存在 / HEAD 被移动)。
set -uo pipefail

TASK_ID="arch-tiling-dram-traffic-min"
REPO="${KFC_REPO_DIR:-/app/repo}"
SCOPE=("tile_planner.py")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say(){ printf '[solve.sh %s] %s\n' "$TASK_ID" "$*"; }
die(){ printf '[solve.sh %s] ERROR: %s\n' "$TASK_ID" "$*" >&2; exit 1; }
# 每次调用单独带 safe.directory,不去改容器里的全局 gitconfig
GIT(){ git -c safe.directory='*' -C "$REPO" "$@"; }

usage(){ cat <<'USAGE'
usage: bash solve.sh [--oracle | --negative | --baseline2 | --noop]
  (无参数)      落地 oracle 参考实现(oracle.patch)
  --negative    落地 negative 已知坏例(negative.patch)
  --baseline2   落地 baseline2 中间基线(baseline2.patch)
  --noop        把可编辑面复位到烤入基线(与 verifier 取基线一致,HEAD 不动)
USAGE
}

WHICH=oracle
case "${1:-}" in
  ""|--oracle) WHICH=oracle ;;
  --negative)  WHICH=negative ;;
  --baseline2) WHICH=baseline2 ;;
  --noop)      WHICH=noop ;;
  -h|--help)   usage; exit 0 ;;
  *)           usage >&2; die "未知参数 '$1'" ;;
esac

# ---- 产物定位:脚本同目录优先,其次容器里的常见挂载点 ----
artifact(){
  local n="$1" c
  for c in "$HERE/$n" "/solution/$n" "${KFC_SOLUTION_DIR:-/nonexistent}/$n"; do
    [ -f "$c" ] && { printf '%s\n' "$c"; return 0; }
  done
  return 1
}

case "$WHICH" in
  noop)     ART="" ;;
  oracle)   ART="oracle.patch" ;;
  negative) ART="negative.patch" ;;
  baseline2) ART="baseline2.patch" ;;
  *) die "本题没有 $WHICH 变体(solution/ 里只有:oracle.patch, negative.patch, baseline2.patch)" ;;
esac

# ---- 环境前置检查 ----
GIT rev-parse --git-dir >/dev/null 2>&1 \
  || die "$REPO 不是 git 工作树 —— 本脚本必须在题目容器里跑(镜像在 build 时才 git init 挖洞基线)"
for s in "${SCOPE[@]}"; do
  [ -e "$REPO/$s" ] || die "被计分单元不存在:$REPO/$s(REPO 不对?可用 KFC_REPO_DIR 覆盖)"
done
HEAD_BEFORE="$(GIT rev-parse HEAD)" || die "读不到 $REPO 的 HEAD"
N_BEFORE="$(GIT rev-list --count HEAD)"

if [ "$WHICH" = noop ]; then
  # --noop 把可编辑面复位到烤入基线(与 verifier 取基线的方式一致 —— `git checkout HEAD -- <scope>`,
  # HEAD 不动),使对照组能量到真正的 no-op 值。runner 在同一个容器里连着跑 oracle → noop 时靠它归零。
  DIRTY_BEFORE="$(GIT status --porcelain -- "${SCOPE[@]}" | sed '/^$/d')"
  GIT checkout -q HEAD -- "${SCOPE[@]}" || die "复位被计分单元失败(${SCOPE[*]})"
  DIRTY_AFTER="$(GIT status --porcelain -- "${SCOPE[@]}" | sed '/^$/d')"
  if [ -n "$DIRTY_AFTER" ]; then
    printf '%s\n' "$DIRTY_AFTER" | sed "s|^|[solve.sh $TASK_ID]   |" >&2
    die "复位后被计分单元仍不干净(见上)—— 量不到真正的 no-op 值"
  fi
  [ "$HEAD_BEFORE" = "$(GIT rev-parse HEAD)" ] || die "HEAD 被移动了,判分会把 candidate 当基线"
  [ "$N_BEFORE" = "$(GIT rev-list --count HEAD)" ] || die "commit 数变了,HEAD 不再是唯一的挖洞基线"
  if [ -n "$DIRTY_BEFORE" ]; then
    say "--noop:已把可编辑面复位到烤入基线;原先这些是脏的:"
    printf '%s\n' "$DIRTY_BEFORE" | sed "s|^|[solve.sh $TASK_ID]   |"
  else
    say "--noop:可编辑面本来就是干净的烤入基线,无需复位"
  fi
  say "--noop 完成:${SCOPE[*]} = 烤入基线(HEAD 未动 @ ${HEAD_BEFORE:0:12})"
  exit 0
fi

PATCH="$(artifact "$ART")" || die "找不到参考产物 $ART(查过 $HERE/、/solution/、\$KFC_SOLUTION_DIR/)"

# ---- 幂等:先把被计分单元退回挖洞基线,再打补丁(否则二次 apply 必然失败) ----
GIT checkout -q HEAD -- "${SCOPE[@]}" || die "复位被计分单元失败(${SCOPE[*]})"
GIT apply -p1 --check "$PATCH" \
  || die "补丁打不上:$PATCH —— 挖洞树与该补丁不匹配,或工作树被别的改动污染了"
GIT apply -p1 "$PATCH" || die "git apply 失败:$PATCH"

# ---- 硬要求:HEAD 一步都不能动(规范 H3) ----
HEAD_AFTER="$(GIT rev-parse HEAD)"
[ "$HEAD_BEFORE" = "$HEAD_AFTER" ] || die "HEAD 被移动了($HEAD_BEFORE -> $HEAD_AFTER),判分会把 candidate 当基线"
[ "$N_BEFORE" = "$(GIT rev-list --count HEAD)" ] || die "commit 数变了,HEAD 不再是唯一的挖洞基线"

say "已落地 $WHICH:$(basename "$PATCH") -> ${SCOPE[*]}(工作树改动,HEAD 未动 @ ${HEAD_AFTER:0:12})"
GIT status --porcelain -- "${SCOPE[@]}" | sed "s|^|[solve.sh $TASK_ID]   |"

# ---- 顺手体检 scope 闸门:工作树里除了被计分单元不该有别的改动 ----
STRAY="$(GIT status --porcelain | awk '{print $NF}' \
         | grep -v -e '__pycache__' -e '\.pyc$' -e '^tile_planner\.py$' || true)"
[ -z "$STRAY" ] || { say "⚠ 工作树里还有 scope 外的改动,判分会 out_of_scope_edit 直接归 0:"; \
                     printf '%s\n' "$STRAY" | sed "s|^|[solve.sh $TASK_ID]   |"; }
exit 0

# 在 P800 用 Claude Code 恢复修复流程

本页只负责启动 Migration Agent，不提供 attempt 1 假设或补丁。具体诊断、允许文件
和代码修改由 Claude Code 中的 `model-adaptation` Skill 根据 Spec 与证据决定。

## 1. 准备当前仓库

在 P800 上执行：

```bash
set -euo pipefail
cd /workspace/model-adaptation-agents

git fetch origin
git switch chore/step3p7-p800-baseline
git pull --ff-only origin chore/step3p7-p800-baseline
test -z "$(git status --short)"
test "$(git rev-parse HEAD)" = \
  "$(git rev-parse origin/chore/step3p7-p800-baseline)"

export SGLANG_KUNLUN_WORKTREE=/workspace/baidu/aicapx/sglang
test "$(git -C "$SGLANG_KUNLUN_WORKTREE" rev-parse HEAD)" = \
  "546ad8c682392922792bbbfe53a8bf575545f118"
export SGLANG_PLATFORM=kunlun
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export PYTHONPATH="$SGLANG_KUNLUN_WORKTREE/python:$SGLANG_KUNLUN_WORKTREE/sglang-kunlun${PYTHONPATH:+:$PYTHONPATH}"
```

如果实际路径不同，只替换 `SGLANG_KUNLUN_WORKTREE`。不要手工填写
`active_hypothesis`，也不要提前修改或清理 SGLang-Kunlun。第一个算子应处于固定
revision 的干净工作区；后续算子会由主 Skill 用上一项 `passing_run` 核对累计
patch，不能用手工 reset 代替。

上述两个 Kunlun 变量与 `PYTHONPATH` 前缀是服务、baseline 和 candidate replay 的
共同启动基线。其余候选变量由主 Skill 按
`docs/p800-environment-and-repair.md` 的节点、拓扑、backend 和活动算子条件选择；
不能整表无条件导出，并要把最终值与原因写入 P800 环境证据。
执行 baseline 或 candidate replay 时，对每个实际导出的候选变量追加
`--p800-environment-reason '变量名=选择原因'`；没有候选变量时不传。工具会从
进程环境读取真实值，缺值或缺原因时必须在算子调用前停止。

## 2. 启动 Claude Code

仍在仓库根目录运行：

```bash
claude
```

进入 Claude Code 后只调用：

```text
/model-adaptation Step-3.7-Flash
```

项目入口 `.claude/skills/model-adaptation/SKILL.md` 会转到主
`model-adaptation/SKILL.md`。Agent 必须从当前 `migration-spec.md` 恢复，只执行
其中唯一 `next_action`。进入 `P800_REPAIR` 后，它自行完成：

```text
当前算子 baseline
  -> 若已等价，直接把当前行标为 PASS
  -> 若存在差异，分析失败证据
  -> 选择一个可证伪假设和本轮最小文件集合
  -> 从源码确定修复位置和原始 Kernel Call 重放方式
  -> 领取 attempt
  -> 在原生产调用位置生成候选修改与聚焦测试
  -> 封存 candidate.patch
  -> 逐字节核对实际 worktree 与 candidate.patch
  -> 从修改后的原生产调用点核对真实参数来源，用原有 Kernel Call 重放
  -> 用固定精度门槛重放全部三个 Golden shape
  -> 通过后在 finish-attempt 前回归以前已通过的算子
  -> 自动进入 gap queue 下一项
  -> 全部行通过后才 PASS / DONE
```

除非缺少环境、需要越过 Repair Boundary 或达到停止条件，否则不要让人选择具体
补丁，也不要在每轮或每个算子之间暂停等待确认。简单修复直接内联在原调用位置；
replay config 可以绑定候选 patch，但 `invocation_target` 必须保持 Scan Run 记录的
原有 P800 Kernel Call；adapter 要拒绝常量、错接参数和不可达伪调用；不新增生产 helper。

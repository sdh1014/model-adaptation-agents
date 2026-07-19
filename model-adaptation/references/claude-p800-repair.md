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
test -z "$(git -C "$SGLANG_KUNLUN_WORKTREE" status --short)"
```

如果实际路径不同，只替换 `SGLANG_KUNLUN_WORKTREE`。不要手工填写
`active_hypothesis`，也不要提前修改 SGLang-Kunlun。

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
`model-adaptation/SKILL.md`。Agent 应从 `migration-spec.md` revision 32 恢复，
读取已验收的 P800 baseline，然后自行完成：

```text
分析失败证据
  -> 选择一个可证伪假设和本轮最小文件集合
  -> 从源码确定修复位置和原始 Kernel Call 重放方式
  -> 补齐并封存该边界的 repair replay adapter（不计 attempt）
  -> 领取 attempt
  -> 生成候选修改与聚焦测试
  -> 封存 candidate.patch
  -> 确认重放确实经过本轮修复边界
  -> 用固定精度门槛重放全部三个 Golden shape
  -> PASS，或从同一基线继续下一轮
```

除非缺少环境、需要越过 Repair Boundary 或达到停止条件，否则不要让人选择具体
补丁，也不要在每轮之间暂停等待确认。不要把 baseline 的
`kunlun_ops.swiglu` 直接调用误当作任意候选补丁已经生效；具体重放入口由 Agent
根据本轮源码修改决定。当前控制器仍只接受 baseline adapter，因此 Claude 必须先
完成上述 repair replay adapter，不能直接把现有工具写成候选 PASS。

# 纠正 eager 运行模式与 Spec 绑定

Type: task
Status: resolved
Blocked by: 10

## What to build

把此前误写为 EAGLE 投机解码的运行边界纠正为 target-only eager：不启用任何 speculative algorithm，不加载 draft 模型路径，并在 CUDA 与 P800 两端同时禁用 prefill 和 decode CUDA Graph。

更新当前 Migration Spec、方案设计、模型适配 Skill、Spec 模板、CUDA preflight runbook、Contract 校验器和公共 CLI 测试。保留 TP8、BF16、量化参数为空、不显式指定 attention/MoE backend、每算子最多三个样本、最多五次修复和固定 Precision Gate。

旧 Contract revision 与 Run 只作为历史证据保留。新工具结果必须绑定 revision 3；`capture_golden.py` 必须拒绝绑定旧 Contract 的 Scan Run。

## Acceptance criteria

- Contract revision 3 明确固定 `draft_entry: null`、`speculative_algorithm: null`，并把 decode/prefill CUDA Graph backend 都固定为 `disabled`。
- 正式启动命令不包含 `--speculative-algorithm`，使用固定 revision 支持的两个 `--cuda-graph-backend-*=disabled` 参数。
- Contract 校验器把两个 `null` 视为有意关闭而不是未决占位，并拒绝 EAGLE、非 disabled graph backend 或缺失字段。
- `capture_golden.py` 通过公共 CLI 拒绝 `spec_binding` 与当前 Contract 不一致的 Scan Run。
- 方案设计、Skill、模板、runbook 和领域语言不再把 eager 与 EAGLE 混用。
- 新的 `runs/spec-binding-002` 绑定 revision 3 并通过；旧 `runs/spec-binding-001` 保持不变。
- 定向测试覆盖正确 Contract、错误 speculative algorithm、错误 graph backend、旧 Scan Run 绑定和文档中的正式启动参数。

## Comments

已完成：

- `migration-spec.md` 已递增到 Contract revision 3，固定 target-only eager：`draft_entry` 与 `speculative_algorithm` 为 `null`，decode/prefill CUDA Graph backend 均为 `disabled`。
- Contract 校验器允许这两个有意关闭的 `null`，并拒绝 EAGLE、非 `disabled` graph backend 或缺失字段。
- `capture_golden.py` 会在创建新 Run 前拒绝绑定旧 Contract 的 Scan Run。
- 方案设计、模型适配 Skill、Spec 模板和 CUDA runbook 已改为同一执行边界。
- `runs/spec-binding-002` 已绑定 revision 3；历史 `runs/spec-binding-001` 未修改。
- 定向测试覆盖正确配置、三类错误配置、旧 Scan Run 拒绝和正式启动参数。

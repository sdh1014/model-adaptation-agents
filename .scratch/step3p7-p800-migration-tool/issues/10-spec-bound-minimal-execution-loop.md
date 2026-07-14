# 打通 Spec 绑定的最小执行闭环

Type: task
Status: ready-for-agent
Blocked by:

## What to build

让现有模型适配 Skill 能以同一个 `migration-spec.md` 完成首次启动和后续恢复，并让一次最小的合成比较动作真正受 Contract 约束。

首次运行时，如果 Spec 不存在，Skill 应从模板生成骨架并填写能够确定的内容；只要 Contract 仍有必填占位，就写入 `NEEDS_HUMAN / SCAN` 和唯一下一步，然后停止。人工批准 Contract revision 1 后，再次运行同一个 Skill，校验通过后进入 `ACTIVE / SCAN`。

实现四个固定脚本共用的最小 Contract Data 读取与结果绑定能力，但不增加第五个脚本或顶层编排命令。先通过 `replay_compare.py` 的一个合成样本动作贯通 `Spec -> Run -> result.json -> Working State`，证明结果记录 `spec_id`、`contract_revision` 和规范化 Contract Data 的 SHA-256。

## Acceptance criteria

- Spec 不存在时可以生成骨架；存在未填写的必填项时不会继续扫描或执行工具。
- 人工批准后的 Contract revision 1 可以进入 `ACTIVE / SCAN`，并且 Working State 中只有一个明确的 `next_action`。
- 工具只解析 Contract Data JSON，不解析 Markdown 正文，也不读写 Working State。
- 修改 Working State 后，已有结果仍能通过绑定校验；修改 Contract Data 或 revision 后，旧结果会被拒绝。
- 合成比较动作在独立 Run 中留下可信的 `result.json` 和日志，工具失败与业务比较失败能够区分。
- 新会话只读取 Spec 和所指向的 Run，就能恢复出同一个唯一下一步。
- 有自动化测试覆盖正常启动、占位停止、Contract 变化拒绝和 Working State 变化不影响摘要。

## Comments

这是后续所有扫描、采集、交接和修复票据的共同入口，不在这里实现真实算子扫描。

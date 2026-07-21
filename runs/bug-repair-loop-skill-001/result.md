# Run Evidence: evidence-backed BUG repair loop

- Date: `2026-07-21`
- Contract revision: `9`
- Phase: `PREFLIGHT`（SOURCE 侧 Skill 改造，未执行 P800）
- Failure category: `none`
- Conclusion: `PASS_SOURCE_ONLY`
- Auto repair status: `NOT_EXERCISED`
- Active bug: `null`
- Bug Ledger: `docs/step3p7-p800-bug-log.md`

## Change

主 Skill 现在要求任何真实失败进入同一个 BUG repair loop：建立稳定 `bug_id`，保存红色
复现，一次只验证一个根因假设，在 Repair Scope 内自动修改并重测，最后以聚焦回归和
原始 P800 路径都通过作为关闭条件。普通范围内修复不等待用户选择方案；需要人类路径、
权限、节点或越过 Repair Scope 时才停止。

详细循环位于 `model-adaptation/references/bug-repair-loop.md`。所有 BUG、attempt、确认
根因和最终解决方法记录到 `docs/step3p7-p800-bug-log.md`。Migration Spec 的 Working
State 新增 `auto_repair_status`、`active_bug` 和 `bug_log`，用于跨轮恢复。

## Capability boundary

本 Run 只验证 Skill 的静态协议，没有 P800 运行失败，也没有执行自动修复。因此能力
状态必须保持 `NOT_EXERCISED`。只有未来至少一个真实 BUG 完成“红色复现、最小补丁、
聚焦回归、原始 P800 路径复验”的闭环，才能改为 `PASS`。未经用户批准不主动注入生产
故障来制造该结论。

## Local validation

- `PYTHONPYCACHEPREFIX=/tmp/model-adaptation-pycache python3 -m unittest discover -s tests -p 'test_*.py'`：8/8 通过；
- `PYTHONPYCACHEPREFIX=/tmp/model-adaptation-pycache python3 -m py_compile tests/test_run_driven_model_adaptation.py`：通过；
- `git diff --check`：通过；
- `.claude/skills/model-adaptation/SKILL.md` 继续只转发到主 Skill，没有第二份流程定义。

## Next action

把当前 revision 9 工作区同步到 P800，在固定 Kunlun worktree 完成 Preflight。随后按
Operator Verification Queue 执行五项生产路径测试并进入真实 TP8 eager 模型；遇到
第一个真实 BUG 时启动闭环并更新 Bug Ledger。

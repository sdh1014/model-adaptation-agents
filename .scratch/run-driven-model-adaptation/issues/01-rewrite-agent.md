# Rewrite model-adaptation as a run-driven Agent

Status: Done

Spec: `../spec.md`

## Work

- 删除旧 capture/replay/handoff/guard 实现及专用材料。
- 重写公开 Skill、领域词汇、当前 Migration Spec 和模板。
- 把五个历史 Operator Candidate 全部加入 Operator Verification Queue。
- 用公开 Skill/Spec 行为测试锁定新接口并清理残留引用。

## Comments

- 2026-07-20：用户明确要求直接改写，并把此前列出的 Operator Candidate 全部纳入测试。
- 2026-07-20：公开 Skill/Spec seam 的 5 个行为测试通过；旧 Python 栈、专用
  runbook、旧 tracker/prototype 和旧实现测试已删除，五个历史 Operator Candidate
  全部以 `PENDING` 进入 revision 7 队列。

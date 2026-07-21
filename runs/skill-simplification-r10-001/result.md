# Skill simplification revision 10

- Contract revision: `10`
- Phase: `SOURCE`
- Result: `PASS`
- P800 execution: `NOT_RUN`
- Auto repair status: `NOT_EXERCISED`

## Decision

按用户确认的新停止边界，将 Skill 收敛为：

```text
PREFLIGHT -> OPERATOR_VERIFICATION -> EAGER_BRINGUP -> DONE
```

删除当前目标不需要的整模型 CPU 精度与逐层调试阶段。初始五个 Operator Candidate
仍全部保留，且必须在 P800 直接验证生产路由与局部精度。

BUG 修复采用稳定 `bug_id`，同一 BUG 最多 3 次不同、由证据支持的 attempt。修复成功
后记录根因、方案、聚焦回归和原始 P800 路径复验并继续；3 次仍失败才 `BLOCKED`。
固定文本和单图 eager 请求正常并复跑成功后直接 `PASS / DONE`。

## Files

- `model-adaptation/SKILL.md`
- `model-adaptation/references/bug-repair-loop.md`
- `model-adaptation/references/migration-spec-template.md`
- `migration-spec.md`
- `CONTEXT.md`
- `docs/p800-environment-and-repair.md`
- `docs/step3p7-p800-bug-log.md`
- `.scratch/run-driven-model-adaptation/spec.md`
- `tests/test_run_driven_model_adaptation.py`

## Verification

- `python3 -m unittest discover -s tests -p 'test_*.py'`: `8/8 PASS`
- `python3 -m py_compile tests/test_run_driven_model_adaptation.py`: `PASS`
- active Skill line count: `172`（修改前 `357`）
- `git diff --check`: `PASS`
- active docs trailing-whitespace scan: `PASS`

## Next action

在固定 P800 worktree 执行 Preflight；通过后按队列验证五个 Operator Candidate。

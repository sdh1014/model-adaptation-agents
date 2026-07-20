# Run Evidence: run-driven workflow rewrite

- Contract revision: `7`
- Phase: `PREFLIGHT`（SOURCE 侧流程准备，未进入 P800 Preflight）
- Failure category: `none`
- Conclusion: `PASS_SOURCE_ONLY`
- SGLang revision: `49e384ce9d304648e9959666ecb8ce8cd98d0deb`
- SGLang-Kunlun revision: `546ad8c682392922792bbbfe53a8bf575545f118`

## Action

删除本仓库自研的 capture/replay/handoff/guard 实现，把公开 Skill、Context、
Migration Spec 和模板改为 Agent 驱动的 P800 eager bring-up。五个历史 Operator
Candidate 全部以 `PENDING` 放入当前 Operator Verification Queue。

审查后进一步完成：

- 以当前 Migration Spec 队列作为执行状态的唯一 operator 清单；
- Skill 要求遍历队列，不复制全限定 operator id；
- CPU reference 的具体实现由 Agent 检查固定源码后选择；
- P800 环境建议只保留固定源码可锚定的语义。

## Local validation

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
git diff --check
PYTHONPYCACHEPREFIX=/tmp/model-adaptation-pycache python3 -m py_compile tests/test_run_driven_model_adaptation.py
```

结果：5 个行为测试全部通过、无 whitespace error、测试文件编译通过；Contract Data
JSON 可解析，活跃文档中的旧入口/旧 phase 残留扫描仅命中用于防回归的负向测试常量。

## Operator and model execution

- CPU reference: 本 Run 未执行。
- P800 Production Operator: 本 Run 未执行。
- 输入 shape/dtype/layout/stride: 不适用。
- Precision Gate: Contract 固定为浮点 `atol=0.01`、`rtol=0.02`，整数精确相等；
  本 Run 没有产生算子或模型数值结果。
- P800 环境、五个 Operator Candidate、真实模型和模型精度状态均保持未开始。

## Next action

在固定 P800 worktree 执行 Preflight；通过后严格按 Operator Verification Queue
顺序测试全部五项。

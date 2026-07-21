# Run Evidence: defer model accuracy

- Date: `2026-07-21`
- Contract revision: `8`
- Phase: `PREFLIGHT`（SOURCE 侧 Contract 更新，未执行 P800）
- Failure category: `none`
- Conclusion: `PASS_SOURCE_ONLY`
- SGLang revision: `49e384ce9d304648e9959666ecb8ce8cd98d0deb`
- SGLang-Kunlun revision: `546ad8c682392922792bbbfe53a8bf575545f118`

## Human decision

局部 Operator Verification 继续使用独立 CPU reference 对比 P800 Production
Operator。整模型 CPU reference 运行成本过高，当前 Contract 不执行模型精度比较。
固定文本和单图 eager 请求都跑通后，Agent 必须保存 Eager Bring-up Run Evidence，
然后设置：

- `model_status: PASS`
- `model_run` 和 `last_run` 指向 Eager Bring-up Run Evidence
- `phase: MODEL_ACCURACY`
- `accuracy_status: BLOCKED`
- `accuracy_run: null`
- `status: BLOCKED`

下一动作是等待用户指定可执行的模型精度 reference 或替代验收方式，并增加 Contract
revision。这个计划内停止点不是 `ACCURACY` Failure Observation。

## Source changes

- 更新公开 Skill 的阶段转换与停止规则；
- 将当前 Contract 的 model `reference_runtime` 设为 `null`，并增加
  `execution_policy: block_after_eager_bringup`；
- 同步 Context、Migration Spec 模板、需求和行为测试。

## Execution

- Operator CPU reference: 本 Run 未执行。
- P800 Production Operator: 本 Run 未执行。
- Eager model: 本 Run 未执行。
- Full-model CPU reference: 按 Contract 禁止执行。
- Model accuracy result: 未产生，不能写为 `PASS` 或 `FAIL`。

## Local validation

提交前执行：

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/model-adaptation-pycache python3 -m py_compile tests/test_run_driven_model_adaptation.py
python3 /Users/songdehao/.codex/skills/.system/skill-creator/scripts/quick_validate.py model-adaptation
git diff --check
```

结果：6 个行为测试全部通过；Skill 结构校验通过；测试文件编译通过；当前 Spec 与模板
的 Contract Data JSON 均可解析，model policy 一致；`git diff --check` 通过；活跃
文档没有遗留 `same-source-sglang` 或 eager 成功后自动进入 `MODEL_ACCURACY` 的旧转换。

只读前向测试也正确得出：Eager Bring-up 通过后写上述 Working State，不运行整模型
reference，不新增 `ACCURACY` Failure Observation。

## Next action

在固定 P800 worktree 完成 Preflight 和五项 Operator Verification，再运行固定文本、
单图 eager 请求；请求都通过后按本 Contract 进入 `BLOCKED`。

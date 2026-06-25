---
status: passed | failed | blocked | partial | decision_required
latest_run: runs/<model>/<target>/<timestamp>-validate
model_analysis_revision: <revision>
assessment_revision: <revision>
implementation_revision: <revision-or-not-applicable>
validated_at: <ISO-8601>
benchmark_ready: true | false
---

# Validation

## 结论

一句话说明当前正确性结论及其边界。

## 输入指纹

- Model revision:
- Target revision / patch:
- Engine:
- Hardware:
- Runtime Python/import:
- Runbook hashes:
- Runtime mode: fresh-server | against-running

## Required Coverage

| Case | Capability | Oracle | Status | Evidence |
|---|---|---|---|---|

## Optional Coverage

| Case | Status | Evidence | Notes |
|---|---|---|---|

## 比较标准

- Tokenizer/chat template:
- Input tokens:
- Sampling:
- Numeric tolerance:
- Reference source:

## 已知偏差

无，或列出已接受但不影响本轮结论的差异。

## 失败与路由

- Failure class:
- Stable signature:
- Confirmed facts:
- Rejected explanations:
- Recommended next action:

## 开发者决策

仅在 `decision_required` 时填写：

- 当前稳定状态：
- 为什么停止自动重跑：
- 推荐决定：
- 其他选项：每项一行，最多两项。

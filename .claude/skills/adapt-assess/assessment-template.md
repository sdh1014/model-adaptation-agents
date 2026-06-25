---
status: passed | blocked
result: already_supported | adaptation_required | blocked
confidence: high | medium | low
revision: 1
model_id: <model-id>
target_id: <target-id>
model_analysis_revision: <revision>
target_revision: <commit-or-unknown>
upstream_revision: <commit-or-package-version-or-null>
environment_readiness: ready | degraded | unavailable | unknown
baseline_status: passed | not_run | import_failed | platform_init_failed | model_recognition_failed | config_failed | weight_load_failed | runtime_op_failed | generation_failed | unknown_failed
latest_run: <run-path>
updated_at: <ISO-8601>
---

# 适配评估：<model-id> / <target-id>

## 1. 结论

- 评估结果：
- 置信度：
- 是否需要代码适配：
- 首个阻塞项或工作项：
- 下一步命令：

## 2. 输入指纹

| 项目 | 值 | 证据 |
|---|---|---|
| 模型来源与 revision | | |
| 模型分析 revision | | |
| 模型 architecture | | |
| 目标引擎 | | |
| 目标仓库与 commit | | |
| 上游仓库/包与 revision | | |
| 硬件 | | |
| Python | | |

## 3. 环境勘测

- Readiness：
- 可见设备：
- TP 要求：
- 关键包版本：
- 引擎/插件 import：
- 路径与存储：
- 环境证据：

### 环境阻塞项

| ID | 类型 | 事实 | 影响 | 建议处置 | 证据 |
|---|---|---|---|---|---|

> 环境阻塞项不自动转成代码工作项。

## 4. Baseline

- 模式：auto / required / skip
- 是否执行：
- 命令来源：
- 失败阶段或结果：
- 错误签名：
- 目标仓前后状态：
- 证据：

## 5. 模型所需能力

| 能力 | 模型要求 | 必需性 | 模型证据 |
|---|---|---|---|

## 6. 能力矩阵

| 能力 | 引擎状态 | 执行路径 | P800 状态 | 判定 | 证据 |
|---|---|---|---|---|---|

状态：`supported / partially_supported / unsupported / unknown / not_applicable`。

执行路径：`native / upstream / plugin / fallback / remote_code / none / unknown`。

## 7. 已确认缺口

| Gap ID | 分类 | 能力 | 缺口描述 | 置信度 | 修改归属 | 验证方式 | 证据 |
|---|---|---|---|---|---|---|---|

允许分类：

```text
model_fact_gap
engine_model_gap
hardware_backend_gap
packaging_version_gap
runtime_environment_gap
configuration_gap
validation_gap
unknown_gap
```

## 8. Fallback 与限制

| 能力 | Fallback | 正确性影响 | 功能影响 | 性能影响 | 是否可接受 | 证据 |
|---|---|---|---|---|---|---|

## 9. 模型分析反馈

| 缺失或修正事实 | 影响能力 | 触发证据 | 增量分析命令 |
|---|---|---|---|

没有时写“无”。

## 10. 实施工作项摘要

| 顺序 | Work Item | Gap ID | 能力 | 依赖 | 验收方式 |
|---|---|---|---|---|---|

环境和版本处置不放入该表。

## 11. 尚未确认的问题

| 问题 | 当前证据 | 缺少的证据 | 对结论的影响 |
|---|---|---|---|

## 12. 证据索引

| 证据 | 路径 | 说明 |
|---|---|---|

## 13. 修订记录

| Revision | 原因 | 模型分析 Revision | Target Revision | 变更范围 |
|---|---|---|---|---|

---
status: pending                 # pending | resolved
item_id: <WI-ID>
classification: <classification>
failure_signature: <signature-hash-or-null>
recommended_option: A
created_at: <ISO-8601>
updated_at: <ISO-8601>
latest_run: <run-path>
---

# 阻塞决策：<model-id> / <target-id> / <WI-ID>

## 1. 状态快照

- 工作项目标：
- 当前工作项状态：`decision_required`
- 已确认可工作的部分：
- 当前稳定失败：
- 自动实现停止原因：
- 目标仓 branch / HEAD：
- 当前累计 patch：

## 2. 关键证据

- 失败签名：
- 首个确定失败点：
- 最后一个成功步骤：
- 相关日志：
- 相关代码：

## 3. 已尝试内容

| Attempt | 假设 | 修改范围 | 验收命令 | 结果 | 证据 |
|---|---|---|---|---|---|

## 4. 已确认事实与未知项

### 已确认事实

-

### 已否定假设

-

### 仍未知

-

## 5. 当前边界

- 当前允许修改：
- 当前禁止修改：
- 若继续所需新增权限、输入或范围：

## 6. 推荐决定

### 选项 A：<推荐名称>

- 决定：
- 原因：
- 允许动作：
- 不允许动作：
- 影响：
- 后续验证：

## 7. 其他可选决定

最多再保留 2 项；没有可信替代项时删除本节。

### 选项 B：<名称>

- 决定：
- 代价：
- 后续验证：

### 选项 C：<名称>

- 决定：
- 代价：
- 后续验证：

## 8. 开发者决定记录

| 时间 | 决定人 | 决定原文 | 允许边界 | 状态 |
|---|---|---|---|---|

也可通过命令追加决定：

```text
/adapt-implement <model-id>/<target-id> \
  --item <WI-ID> \
  --decision "<明确决定和允许边界>"
```

## 9. 恢复条件

- 决定必须明确允许做什么、禁止做什么；
- 决定涉及模型事实或评估变化时，先完成对应阶段；
- 恢复后仍只处理当前工作项。

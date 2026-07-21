# Block after eager bring-up

Status: Superseded

Spec: `../spec.md`

## Work

- 保留局部算子的 CPU reference 与 P800 生产算子精度比较。
- 禁止 Agent 自动启动整模型 CPU reference。
- 固定文本和单图 eager 请求跑通后，将模型状态设为 `PASS`，精度状态和整体状态设为
  `BLOCKED`。
- 只有用户增加 Contract revision 并指定模型精度 reference 或替代验收方式后，才可
  恢复 `MODEL_ACCURACY`。

## Comments

- 2026-07-21：用户确认整模型 CPU reference 运行过慢，要求 eager 模型跑通后
  `BLOCKED` 等待用户决策。
- 2026-07-21：Contract revision 10 取消这个计划内阻塞。当前目标不执行整模型精度
  阶段；固定 eager 请求正常后直接 `PASS / DONE`。

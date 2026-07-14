# 产出 Spec 驱动 P800 算子迁移工具方案设计文档

Type: task
Status: resolved
Blocked by: 09

## Question

将所有已关闭子票的决策和端到端纸面 Demo 合成为 `outputs/step3p7-p800-migration-tool-design.md`。文档必须包含目标与非目标、核心术语、总体架构、Spec Schema、阶段流程、目录与产物契约、单 Agent 与脚本职责、状态与停止条件、Demo 验收、未验证假设、风险及后续扩展路线，并为所有源码相关结论提供代码证据。

## Comments

- 最终方案文档已完成并校验，见 [Step-3.7-Flash 到 KLX P800 的 Spec 驱动算子迁移工具方案](../../../outputs/step3p7-p800-migration-tool-design.md)。
- 文档只给出设计，不实现工具；源码结论已按 `kunlun-0.5.14@546ad8c682392922792bbbfe53a8bf575545f118` 的真实文件与行号复核。

## Answer

最终文档已经合成所有已关闭子票的决策，覆盖目标与非目标、术语、单 Agent 架构、单文件 Spec、Contract Data JSON、状态恢复、真实路径扫描、一次性 CUDA Golden、人工交接、P800 五轮修复、四个脚本接口、`torch.testing.assert_close`、特殊 SwiGLU 资格、产物契约、验收、风险与扩展路线。

为保证交接包中的 Spec 不会在 manifest 生成后因状态更新而失效，文档补充了一个实现级约束：Agent 先生成已经进入 `WAITING / HANDOFF` 的 Spec 临时副本，bundle 校验通过后再用相同字节原子替换当前 Spec；这不增加第二个状态来源。Tensor 序列化格式继续明确保留为实现前的实机确认项。

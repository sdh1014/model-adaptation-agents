# 使用特殊 SwiGLU 验证端到端流程

Type: task
Status: ready-for-agent
Blocked by: 14, 15

## What to build

人工把交接包复制到 P800 后，先完成目的端完整性校验，再优先使用特殊 SwiGLU 的一至三个 Golden Samples 验证 `恢复 Spec -> baseline replay -> 精度比较 -> 有限轮修复 -> 最终结论` 的完整流程。

特殊 SwiGLU 只有在固定配置中实际激活、原始 `Step3p5MLP.forward` 已经采到真实调用，并且 P800 rank 0 baseline 至少有一个执行或精度失败时，才能作为被修复的真实 Operator Gap。缺少专用 Kunlun Kernel 或潜在性能较慢都不能当作缺口证据。

如果特殊 SwiGLU baseline 全部通过，应如实记录它不是 correctness gap。此时它已经验证采集、交接、重放和比较链路，但还没有验证修复链路；Agent 应从同一份已采集 gap queue 中选择下一个 baseline 真实失败的无权重候选完成修复验证，不回 CUDA。

## Acceptance criteria

- P800 开始执行前，bundle、manifest、Spec 绑定和固定源码 revision 全部校验通过。
- 特殊 SwiGLU 的分支激活有加载后配置或运行证据，不能仅根据源码存在该分支推断。
- baseline 对全部已采集 shape 执行，结果写入独立 Run，且不消耗 repair attempt。
- baseline FAIL 后最多进行五轮修复；每轮一个假设、失败恢复基线、通过补丁保留未提交。
- 最终 PASS 要求全部已采集 shape 通过固定结构、dtype、有限值和 `torch.testing.assert_close` 门槛；性能不属于本 Demo 验收。
- 如果特殊 SwiGLU baseline PASS，必须换用已经采集且 baseline 真实失败的候选验证修复链路，不能伪造失败或重新访问 CUDA。
- 需要新增 C++/Kernel、五轮耗尽、没有任何真实 gap 或出现未知工作区修改时，形成证据完整的 `BLOCKED`，不能扩大权限边界。
- 最终 Run、补丁、比较结果、日志、Spec 状态和 Demo 结论能够由新会话独立复核。
- 完成后保留剩余 gap queue 及其 Scan Run、Golden Sample 和 baseline 状态，足以继续按“一个真实缺口一个票据”拆分后续工作。

## Comments

本票据只关闭一个真实缺口来证明流程。其余缺口等本票据完成后，根据真实 Scan Run 从 17 开始逐算子创建票据，不创建一个无法在单次会话内完成的“适配所有算子”大票。

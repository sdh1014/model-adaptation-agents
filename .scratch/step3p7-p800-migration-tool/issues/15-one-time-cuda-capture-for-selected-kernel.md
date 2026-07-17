# 一次性采集已选择的 kernel 调用

Type: task
Status: ready-for-agent
Blocked by: 13, 22

## What to build

在 Ticket 22 的 adapter 和 CUDA preflight 通过后，运行唯一一次正式 CUDA Capture
Session。消费 revision 5 的 `scan-005.capture_plan`，只采集扫描后选中的
`sgl_kernel.gemma_rmsnorm`，而不是把 revision 4 的 MLP 或全部候选混进最小
Demo。

rank 0 按输入 shape 去重，最多保存三份
`x + direct weight + eps -> CUDA output`。停止采集后，在同一次 CUDA Session 内
直接调用现有 kernel 接口完成全部样本 self-replay；通过后构建并校验 Handoff
Bundle。

## Acceptance criteria

- Session 前确认 Contract revision 5、`spec-binding-004`、`scan-005` 和
  adapter 证据绑定一致。
- 正式启动使用固定 checkpoint、TP8、BF16、target-only eager；两端命令一致且不
  显式指定 attention/MoE backend。
- 只 Hook `sglang.srt.layers.layernorm.gemma_rmsnorm`，不新增 helper、自定义
  算子函数或模型 wrapper。
- 最多保存三个 rank 0 shape；重复和第四种 shape 只计数。
- 样本允许当前调用直接使用的 `weight`，但拒绝完整 checkpoint、module
  `state_dict` 和无关参数。
- self-replay 覆盖全部样本，使用 Contract 固定 Precision Gate；任一失败都会阻止
  Golden Run 封存。
- bundle 在 CUDA 端 build + verify 通过后，Spec 原子进入
  `WAITING / HANDOFF`。
- Session 消耗后不得因其他 gap queue 候选再访问 CUDA；扩展到全部缺口是后续
  版本，不属于最小 Demo。

## Comments

revision 3/4 的特殊 SwiGLU 和 MLP capture 决策作为历史保留；本票据只消费
revision 5 当前选择。

# 实现 TP8 rank 0 MLP 采集与模型内重放

Type: task
Status: resolved
Blocked by: 19

## What to build

Hook `sglang.srt.models.step3p5.Step3p5MLP.forward`，只处理 `self.limit is not None` 的实例。CUDA 端实际 TP8 模型只在 rank 0 保存边界输入 `x` 和 CUDA `output`；P800 端加载同一 checkpoint 和同一 TP8 模型，在相同 rank 0 `Step3p5MLP` 实例上输入 `x` 并执行固定 Precision Gate。

特殊 shared expert 的 `down_proj` 使用 `reduce_results=False`，归约发生在 `Step3p5MLP.forward` 之后。因此最小 Demo 不需要保存八份记录；另外七个 rank 正常执行模型，但不额外保存或比较。

## Acceptance criteria

- 插件注册 target forward 上下文 Hook 和 `Step3p5MLP.forward` Hook，不再依赖 helper 文件。
- 只采集 `self.limit is not None`；`limit is None` 的普通 MLP 原样执行且不落盘。
- 每条记录包含模型层路径、`tp_rank=0`、TP size、执行阶段、shape、dtype、layout、stride 和 limit。
- 样本格式为 `samples/<shape-id>.pt`；唯一采集状态写到 Run 根目录的 `capture-state.json`。
- payload 只包含 `x`、CUDA `output` 与必要元数据，不包含权重或内部 Tensor。
- `capture_golden.py` 要求 Scan Run 声明 checkpoint/TP8 权重依赖、原始 MLP Hook、`tp_rank=0` 和权重不进入 Golden Sample。
- P800 replay 只能在已加载同一 checkpoint 的模型实例内执行；不得回退为独立 helper 或无权重函数重放。
- replay 对 rank 0 的全部 shape 使用 Contract 固定的 `torch.testing.assert_close`，actual Tensor 不落盘。
- 自动化测试覆盖三 shape、第四种 shape 跳过、非 0 rank 不落盘、激活条件、真实 HookRegistry 接线和模型内 replay。

## Comments

2026-07-17 已完成本地实现与自动化测试。插件只注册 SGLang 现有的 target forward 和原始 MLP 方法 Hook；固定原始源码中不存在额外模型函数。真实 CUDA checkpoint capture 与 P800 round-trip 仍属于 Ticket 12 的实机验收，不由本票据伪装为已完成。

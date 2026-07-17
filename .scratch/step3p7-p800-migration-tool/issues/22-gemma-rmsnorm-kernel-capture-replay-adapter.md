# 实现 Gemma RMSNorm kernel 采集与重放 adapter

Type: task
Status: wontfix
Blocked by: 21

## What to build

消费 revision 5 的 `runs/scan-005`，在现有 call-site
`sglang.srt.layers.layernorm.gemma_rmsnorm` 捕获
`sgl_kernel.gemma_rmsnorm` 的真实调用。rank 0 按输入 shape 去重，最多保存三份
`x + weight + eps -> CUDA output`。

这是 kernel 级 standalone replay，不加载整层 MLP，也不新增 helper、自定义算子
函数或模型 wrapper。

## Acceptance criteria

- `capture_golden.py` 只接受 `scan-005.selection.active_operator` 和匹配的
  `capture_plan`。
- HookRegistry 包裹现有 `sglang.srt.layers.layernorm.gemma_rmsnorm` call-site。
- 只保存 rank 0，最多三个不同 `x.shape`；重复和第四种 shape 有计数但不落 Tensor。
- payload 只包含 `x`、直接参数 `weight`、`eps`、CUDA output 和必要元数据。
- 明确拒绝完整 checkpoint、module `state_dict` 和无关参数。
- CUDA self-replay 与 P800 replay 都直接调用同一个现有 kernel 接口。
- 浮点输出用 Contract 固定的
  `torch.testing.assert_close(atol=0.01, rtol=0.02)`；actual 不落盘。
- preflight 不访问 checkpoint、不消耗正式 Session，并验证三 shape、去重、rank
  过滤、参数保存规则和新进程 self-replay。
- revision 4 的 MLP adapter 保持历史测试可用，但不能消费 revision 5。
- 实现与测试通过后生成新的 adapter Run；不修改 `scan-005`。Working State 更新为
  `last_completed_action: kernel_capture_replay_adapter_implemented`，并把
  `last_run` 和唯一 CUDA preflight 下一动作指向新证据。
- 完成后更新 CUDA runbook 的实际命令；本票据不伪装完成 CUDA/P800 实机验证。

## Comments

`scan-005` 选择这个调用，是因为 Step-3.7 文本 q/k norm 固定可达、只有一个输出、
没有 TP 通信，只需一个直接 1-D 参数，并且 Kunlun 已有普通 RMSNorm 能力可供后续
修复复用。

2026-07-17：`scan-006` 补回更小的已有 `_swiglu_silu_clamp_mul` 缺口并改选它。
本票据不再执行；SwiGLU adapter 由 Ticket 23 继续。

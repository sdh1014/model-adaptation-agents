# 实现 SwiGLU clamp Kernel Call 采集与重放 adapter

Type: task
Status: resolved
Blocked by: 21

## What to build

消费 revision 5 的 `runs/scan-006`，在现有 call-site
`sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul`
捕获真实 CUDA 调用。rank 0 按输入 shape 去重，最多保存三份
`x + gemm1_limit -> CUDA output`。

这个 `@torch.compile` 函数已经存在于固定 SGLang revision，不是工具新造的 helper。
CUDA self-replay 调原函数；P800 baseline 用相同 `x` 调当前 Kunlun MoE 的
`kunlun_ops.swiglu` 路径，检查它是否因忽略 limit 而与 CUDA expected 不一致。

## Acceptance criteria

- `capture_golden.py` 只接受 `scan-006.selection.active_operator` 和匹配的
  `capture_plan`。
- HookRegistry 包裹现有 `_swiglu_silu_clamp_mul` call-site，不新增模型函数或
  自定义算子。
- 只保存 rank 0，最多三个不同 `x.shape`；重复和第四种 shape 有计数但不落 Tensor。
- payload 只包含 `x`、`gemm1_limit`、CUDA output 和必要元数据，不保存参数 Tensor。
- 明确拒绝完整 checkpoint、module `state_dict` 和无关参数。
- CUDA self-replay 直接调用现有 `_swiglu_silu_clamp_mul`；P800 baseline 显式调用
  当前 `kunlun_ops.swiglu` 路径，不能退化为在 P800 上只测同一个 Torch 函数。
- 浮点输出用 Contract 固定的
  `torch.testing.assert_close(atol=0.01, rtol=0.02)`；actual 不落盘。
- preflight 不访问 checkpoint、不消耗正式 Session，并验证三 shape、去重、rank
  过滤、参数保存规则和新进程 self-replay。
- revision 4 的 MLP adapter 保持历史测试可用，但不能消费 revision 5。
- 实现与测试通过后生成新的 adapter Run；不修改 `scan-006`。Working State 更新为
  `last_completed_action: kernel_capture_replay_adapter_implemented`，并把
  `last_run` 和唯一 CUDA preflight 下一动作指向新证据。
- 完成后更新 CUDA runbook 的实际命令；本票据不伪装完成 CUDA/P800 实机验证。

## Comments

`scan-006` 选择这个调用，是因为它在 Step-3.7 文本 MoE 第 43、44 层可达，只有
一个输入 Tensor 和一个输出 Tensor，没有边界内 TP 通信，也不需要保存权重。固定
Kunlun MoE 路径已有普通 SwiGLU，但没有读取 clamp limit。

2026-07-17 已完成本地 adapter 实现：

- `capture_golden.py` 消费 `scan-006` 的活动 Kernel Call 和 capture plan；
- SGLang 插件只 Hook 已有 `_swiglu_silu_clamp_mul`；
- rank 0 collector 保存 `x / gemm1_limit / output`，参数字典为空，最多三个 shape；
- `replay_compare.py --mode kernel-replay` 在 CUDA 调原函数，在 P800 调
  `kunlun_ops.swiglu`，actual Tensor 不落盘；
- preflight 会在 rank 0 采集后启动 rank 1 过滤 probe，再启动独立 CUDA
  self-replay worker；
- 本机只完成 CPU 测试，没有运行 CUDA/P800，也没有消耗正式 Capture Session。

初始实现证据封存在 `runs/adapter-001`。随后
`runs/cuda-preflight-r5-001` 已由 CUDA 机器回传，
`runs/p800-portability-r5-001` 也已验证跨端样本读回和固定比较器。

Ticket 13 code review 后，replay config、worker result、Golden state 和 wrapper
result 增加同一 `sample-files.json` SHA-256 绑定。当前源码摘要和本机元数据测试
封存在 `runs/adapter-002`；该 Run 没有重跑 CUDA/P800。必须先为当前源码重新执行
不消耗正式 Session 的 preflight，才能开始 Ticket 15 的唯一 CUDA Session。

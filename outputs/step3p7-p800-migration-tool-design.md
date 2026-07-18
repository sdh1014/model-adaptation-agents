# Step-3.7-Flash 到 KLX P800 的最小自动适配方案

> 当前版本：Contract revision 5
>
> 当前扫描：`runs/scan-006`
>
> 当前选择：`sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul`
>
> 证据边界：唯一实际采集 Session 和 CUDA Handoff Bundle 已封存并通过校验；
> Working State 为 `WAITING / HANDOFF`，尚未完成 P800 manifest 校验或 baseline

## 1. 目标

做一个由 Spec 驱动的最小工具：

1. 沿 Step-3.7-Flash 的实际输入路径扫描 CUDA/Kunlun 实现差异；
2. 把缺口定位到源码中已经存在的 Kernel Call；
3. 扫描完成后，从真实缺口队列选择最小 Demo；
4. 在 CUDA 机器一次采集最多三个真实 shape；
5. 人工把交接包复制到 P800；
6. P800 只依赖 Golden Sample 反复重放、修复和比较；
7. 上下文被压缩或换 Agent 后，仍能从 `migration-spec.md` 恢复唯一下一步。

最小工具保持一个 Agent、一个 Spec、四个小脚本和不可变 Run。它不是多 Agent
编排系统，也不建设新的顶层 CLI。

## 2. 本轮边界变化

revision 4 预先把 Demo 固定为 `Step3p5MLP.forward`。这会让扫描结论服从预定
边界，也让重放依赖整个 MLP 的 checkpoint 权重。

revision 5 改成：

- Contract 只固定扫描范围、运行条件、输入模式、样本规则和精度门槛；
- Contract 不包含 `operator_boundary` 或 `active_operator`；
- 扫描粒度下降到现有 `kernel-call`；
- Triton 只是实现类型之一，CUDA extension、SGLang JIT、第三方 kernel 和真实
  不兼容的 Torch 调用同样在范围内；
- 扫描完成后比较 `topk_sigmoid`、视觉 attention 与其他真实缺口；
- `active_operator` 从当前 Scan Run 的 gap queue 选择，再写入 Working State；
- Golden Sample 可以保存当前调用直接使用的参数 Tensor，但不能保存完整
  checkpoint、module `state_dict` 或无关参数。

旧 `scan-004`、`spec-binding-003` 和 MLP adapter 保留为历史，不改写，也不能消费
revision 5。

## 3. 为什么必须由 Spec 驱动

`migration-spec.md` 分成两个部分：

| 部分 | 谁修改 | 保存什么 |
|---|---|---|
| Contract | 人 | 目标、固定输入、扫描范围、样本规则、精度门槛、权限和停止规则 |
| Working State | Agent | 当前阶段、gap queue、活动算子、最近 Run 和唯一下一动作 |

工具只解析 Contract Data，不读取 Working State。Agent 每次动作前重读两部分，
动作结束后先封存 Run，再更新 Working State。

这样即使上下文被压缩，也只需读取：

1. 当前 Contract revision；
2. `status / phase`；
3. `active_operator`；
4. `last_run`；
5. `next_action`。

聊天记录不是恢复依据。

## 4. 固定 Contract

revision 5 固定：

- 模型：`Step-3.7-Flash`；
- CUDA SGLang：`49e384ce9d304648e9959666ecb8ce8cd98d0deb`；
- Kunlun SGLang：`546ad8c682392922792bbbfe53a8bf575545f118`；
- checkpoint：
  `stepfun-ai/Step-3.7-Flash@5f6244077ac62e04eec3f320501ff8c2b293373a`；
- target-only，不加载 draft，不启用投机解码；
- TP8、BF16、无量化；
- decode 与 prefill CUDA Graph 都关闭；
- 两端启动参数相同，不显式指定 attention backend；
- 扫描输入：`text-only`、`single-image`；
- 扫描粒度：`kernel-call`；
- rank 0，每个算子最多三个 shape；
- `torch.testing.assert_close(atol=0.01, rtol=0.02)`；
- 最多五次修复，baseline 不计次数。

模型 dtype 是 BF16，不表示每个 kernel 的所有 Tensor 都必须是 BF16。例如
`topk_sigmoid` 的权重输出是 FP32，expert id 是 INT32。比较时保持原 dtype：浮点
使用固定容差，整数和布尔值必须精确相等。

## 5. 扫描方法

### 5.1 输入路径

文本路径：

```text
Step3p7ForConditionalGeneration.forward
  -> general_mm_embed_routine(text-only)
  -> Step3p5ForCausalLM.forward
  -> Step3p5Model.forward
  -> Step3p5DecoderLayer.forward
```

单图路径：

```text
Step3p7ForConditionalGeneration.forward
  -> general_mm_embed_routine(single-image)
  -> Step3p7ForConditionalGeneration.get_image_feature
  -> Step3p7ForConditionalGeneration._get_vision_model_output
  -> PerceptionEncoder.forward
  -> PerceptionEncoder.forward_features
  -> PerceptionEncoderVisionTransformer.forward
  -> PerceptionEncoderVisionBlock.forward
  -> VisionAttention.forward
```

扫描不进入 `Step3p5MTP.forward`，也不把 CUDA Graph、请求调度和只改变 Tensor
视图的表达式登记为缺口。

### 5.2 kernel 边界

一个扫描项必须是源码中已经存在的 Kernel Call，并能写清：

- 实际 symbol 和调用链；
- 输入、直接参数、非 Tensor 参数和输出；
- CUDA 实现；
- Kunlun 同名实现或上层替代路径；
- 是否能独立捕获、重放和替换。

不得为了打桩新增 helper、自定义算子函数或整层 wrapper。

### 5.3 判断 Kunlun 是否缺失

按顺序检查：

1. Kunlun 是否实现同一调用；
2. Kunlun 是否在更高调用点改走等价实现；
3. 固定参数下是否仍会进入未绑定或不可执行的 CUDA 调用。

只有第三种进入 gap queue。缺少同名 symbol 不等于缺口。例如 Kunlun 在
`UnquantizedFusedMoEMethod.apply` 替换整个 routed MoE，因此不会执行上游
`fused_moe_kernel`；这个 Triton kernel 应标为 `READY`，而不是缺失。

源码依据：

- CUDA fused MoE kernel：
  `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py:324-907`；
- Kunlun 上层替换：
  `sglang_kunlun/hooks/layers/quantization/unquant.py:32-155`。

## 6. scan-006 结果

`scan-005` 已封存并保持字节不变。复核发现它漏掉了 clamp SwiGLU，并把视觉路径
识别错了；因此按 Run 不可变规则新建 `scan-006`，没有原地改写旧证据。

`scan-006` 记录 11 个当前输入路径上的 CUDA 专用 Kernel Call：

- `READY=6`
- `CAPTURE_REQUIRED=5`
- `NEEDS_HUMAN=0`

五个静态缺口候选是：

1. `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul`
2. `sgl_kernel.gemma_rmsnorm`
3. `sgl_kernel.gemma_fused_add_rmsnorm`
4. `sgl_kernel.topk_sigmoid`
5. `sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel`

这里的 `CAPTURE_REQUIRED` 只表示固定源码下缺少 Kunlun 等价调用，需要 CUDA Golden
和 P800 baseline。它不表示已经在 P800 实机失败。

## 7. 候选比较

| 候选 | 固定输入可达性 | 需要保存的直接参数 | 输出与状态 | 修复面 | 结论 |
|---|---|---|---|---|---|
| `_swiglu_silu_clamp_mul` | 文本 MoE 第 43、44 层可达 | 无 | 单 Tensor、无 TP 通信 | 让 Kunlun SwiGLU 保留已有 limit 语义 | 首选 |
| `gemma_rmsnorm` | 文本路径每层 q/k norm 必达 | 一个 1-D `weight` | 单 Tensor、无 TP 通信 | 可复用已有 Kunlun RMSNorm | 延后 |
| `gemma_fused_add_rmsnorm` | 文本路径必达 | 一个 1-D `weight` | 两个 in-place Tensor | 可复用 fused RMSNorm，但验证更复杂 | 延后 |
| `topk_sigmoid` | MoE 文本路径必达 | `correction_bias[288]` | FP32 weights + INT32 ids | 需要保证 biased top-k、归一化和 id 顺序 | 延后 |
| 视觉 `_fwd_kernel` | 需要单图且 CUDA mm backend 为 Triton | 无 | q/k/v、序列 metadata、布局较多 | 需要映射到 Kunlun attention | 延后 |

### 7.1 为什么不是 topk_sigmoid

Step-3.7 的 MoE 确实固定使用 sigmoid、correction bias、top-k 8 和 renormalize：

- 模型创建参数：
  `python/sglang/srt/models/step3p5.py:118-160`；
- 实际 CUDA 链：
  `python/sglang/srt/layers/moe/topk.py:453-515,733-810,1729-1811`；
- CUDA symbol 来自 `sgl_kernel`：
  `python/sglang/srt/layers/moe/topk.py:176-182`；
- Kunlun 的 `select_experts` 仍进入上游 `fused_topk`：
  `sglang_kunlun/hooks/layers/moe/topk.py:60-193`；
- Kunlun stub 没有绑定 `topk_sigmoid`：
  `sglang_kunlun/bootstrap/sgl_kernel_stub.py:158-172`。

它是很好的后续缺口，但首个 Demo 还要比较浮点权重与精确 expert ids，排序一致性比
单输出 SwiGLU 更复杂。

### 7.2 为什么不是视觉 attention

非 Hopper/Blackwell CUDA 默认会选择 `triton_attn`：
`python/sglang/srt/layers/attention/vision.py:1065-1108`。实际 launcher 与 kernel
位于：

- `python/sglang/srt/layers/attention/vision.py:337-409`；
- `python/sglang/srt/layers/attention/triton_ops/prefill_attention.py:34-219`。

固定 Kunlun plugin 没有视觉 `context_attention_fwd` 替换。但这个候选依赖单图
请求和 CUDA 设备能力，重放还需要 q/k/v 与序列 metadata，因此不是最小首选。

### 7.3 为什么选择 `_swiglu_silu_clamp_mul`

Step-3.7 把 MoE 第 43、44 层的 `gemm1_clamp_limit` 固定为 7。CUDA 的未量化
Triton runner 把该值传入源码中已有的 `@torch.compile` 函数：

```python
@torch.compile
def _swiglu_silu_clamp_mul(x, gemm1_limit):
    gate, up = x.chunk(2, dim=-1)
    gate = F.silu(gate)
    gate = gate.clamp(max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * up
```

源码位置：

- limit 从模型进入 FusedMoE：
  `python/sglang/srt/models/step3p5.py:118-158`；
- CUDA runner 继续传递 limit：
  `python/sglang/srt/layers/moe/moe_runner/triton.py:132-166`；
- 已有调用的定义和使用：
  `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:326-333,551-571`。

Kunlun 在更高层替换 MoE，但只调用普通 `kunlun_ops.swiglu`，没有读取
`gemm1_clamp_limit`：
`sglang_kunlun/hooks/layers/quantization/unquant.py:122-125`。

因此最小捕获边界是现有调用，不需要新增 helper：

```text
hook: sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul
input: x
non-tensor argument: gemm1_limit
output: output
```

它不保存 expert 权重或 checkpoint。CUDA self-replay 调原有函数；P800 baseline
用同一份 `x` 检查当前 Kunlun SwiGLU 路径是否遗漏 limit。

### 7.4 为什么暂不选择 `gemma_rmsnorm`

Step-3.7 attention 为 q/k 创建 `GemmaRMSNorm`，并在每层调用：

- 创建：`python/sglang/srt/models/step3p5.py:377-378`；
- 调用：`python/sglang/srt/models/step3p5.py:433-440`；
- CUDA leaf call：
  `python/sglang/srt/layers/layernorm.py:648-719`。

Kunlun stub 只绑定 `rmsnorm` 和 `fused_add_rmsnorm`，没有绑定 Gemma 版本：
`sglang_kunlun/bootstrap/sgl_kernel_stub.py:196-207`。同时它已经有可复用的普通
RMSNorm：
`sglang_kunlun/kernels/sgl_kernel_kunlun/layernorm.py:8-29`。
Kunlun platform 明确把 `MultiPlatformOp` 分派到 `forward_cuda`：
`sglang_kunlun/platform/device.py:8-17`，所以这里不能假定会自动走
`forward_native`。

它的捕获边界同样清楚：

```text
hook: sglang.srt.layers.layernorm.gemma_rmsnorm
input: x
direct parameter: weight
non-tensor argument: eps
output: output
```

但它需要额外保存当前 rank 的 1-D `weight`。相比之下，已存在的 clamp SwiGLU
调用没有参数 Tensor，输入输出也都是单 Tensor，因此更小。

## 8. 参数保存规则

Golden Sample 可以保存：

- 活动 kernel 的直接输入；
- 该次调用直接使用的参数 Tensor；
- 必要的标量或枚举；
- CUDA 期望输出；
- shape、dtype、stride、rank 和来源元数据。

Golden Sample 不可以保存：

- 完整 checkpoint；
- module `state_dict`；
- 不参与该次调用的参数；
- 完整 batch、prompt/token、KV cache；
- stream、handle、随机状态或凭证。

“不保存完整权重”不等于禁止一切参数。当前首选不需要保存参数 Tensor；
`gemma_rmsnorm` 的 `weight` 和
`topk_sigmoid` 的 `correction_bias` 都是当前 kernel 的直接参数，可以保存；整层
MLP 或全部 expert 权重不适合作为最小 Demo。

## 9. Golden Sample

当前首选样本建议：

```text
samples/<shape-id>.pt
  schema
  operator_id = sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul
  tp_rank = 0
  signature
    x: shape, dtype, stride
    gemm1_limit
  inputs
    x
  parameters = {}
  non_tensor_args
    gemm1_limit
  outputs
    output
```

同一 shape 只保存第一次真实调用，最多三种。输入和参数保留原 dtype 后复制到 CPU。
P800 actual output 只在内存中参与比较，不落盘。

## 10. 工具职责

| 工具 | 负责 | 不负责 |
|---|---|---|
| `capture_golden.py` | 校验 Scan/Contract、生成采集配置、控制三 shape/rank 0 | 选择算子、修改 Contract |
| 采集插件 | Hook 现有调用、保存允许的输入/参数/输出 | 新增 helper 或自定义算子 |
| `replay_compare.py` | self-replay、P800 replay、结构与精度比较 | 放宽容差、决定修复 |
| `handoff_bundle.py` | self-replay 前记录样本文件摘要，生成与校验 manifest | 自动跨机器复制 |
| `workspace_guard.py` | 固定基线、保存 patch、检查唯一修改 | 自动 commit/push |

当前代码中的 MLP adapter 是 revision 4 历史实现。revision 5 的
`_swiglu_silu_clamp_mul` adapter 已实现：采集插件 Hook 原调用，rank 0 collector
保存最多三个 shape，CUDA self-replay 调 SGLang 原函数，P800 baseline 调
`kunlun_ops.swiglu`。revision 5 CUDA preflight 已在 A800 上通过并回传
`runs/cuda-preflight-r5-001`；它未加载 checkpoint，也未消耗正式 Capture Session。
三个样本也已由 P800 修改版 Torch 读回，固定比较器正常通过并能拒绝有意数值偏差；
证据在 `runs/p800-portability-r5-001`，且没有保存 P800 actual Tensor。

`scan-006` 是不可变证据，所以其中的 `adapter_status=NOT_IMPLEMENTED` 不会被原地
更新。初始实现证据保存在 `runs/adapter-001`；交接包 code review 后的当前源码
摘要与样本/replay 绑定证据保存在 `runs/adapter-002`。该 Run 本身只做本机源码
与元数据校验；后续独立的 `runs/cuda-preflight-r5-002` 已补做当前源码的 CUDA
preflight，且没有消耗正式 Session。

## 11. 运行流程

```text
Contract revision 5
  -> spec-binding-004
  -> scan-005（历史初稿）
  -> scan-006（纠正后当前证据）
  -> 扫描后选择 _swiglu_silu_clamp_mul
  -> 实现并测试该 kernel 的 adapter
  -> CUDA preflight（不消耗正式 Session）
  -> 一次 CUDA Golden Capture，最多三个 shape
  -> 记录 Golden Sample 文件大小与 SHA-256
  -> CUDA self-replay
  -> 构建并校验 Handoff Bundle
  -> 人工复制
  -> P800 manifest 校验
  -> baseline replay
  -> 最多五轮修复
  -> PASS 或有证据的 BLOCKED
```

## 12. 停止规则

立即停止并保留证据的情况：

- Contract 与 Run 绑定不一致；
- 实际 hook 不是 Scan Run 记录的现有调用；
- 需要保存完整 checkpoint 或 module state；
- 唯一 CUDA Session 已消耗但 Golden 不可信；
- P800 baseline 全部通过，说明首选不是实机 correctness gap；
- 修复需要新 C++、自定义 kernel、底层注册或完整模型改动；
- 第五轮仍失败；
- 工作区存在无法解释的已有修改。

性能不属于本 Demo 门槛。首个 Demo 只要求最多三个真实 shape 的正确性通过。

## 13. 当前下一步

`runs/p800-portability-r5-001` 已证明三个 CUDA BF16 样本可由 P800 修改版 Torch
读回，固定比较器正常 PASS 并能拒绝有意数值偏差，且没有保存 P800 actual Tensor。
Ticket 13 已实现 `handoff_bundle.py` 的 record-samples/build/verify：正式
self-replay 前先记录每个 Golden Sample 文件的大小和 SHA-256，并保存独立的
record-samples Run；build 显式核对该 Run、sidecar、当前 kernel、checkpoint、
TP8/rank 0、全部 shape、样本字节、CUDA self-replay 结果与摘要；replay config、
worker result、Golden state 和 wrapper result 还必须绑定同一 sidecar SHA。最后
把 record result/sidecar 摘要和完整允许文件集合写入 manifest，以检测缺失、额外
或篡改。

`runs/cuda-preflight-r5-002` 已验证 adapter-002 当前源码的三个 rank-0 shape、
样本摘要绑定和 CUDA 新进程 self-replay，且未消耗正式 Session。Ticket 14 也已实现
`workspace_guard.py` 的基线检查、baseline 零计数判定、连续且不可复用的 attempt
编号、replay 前完整 patch 固化、patch/replay 联合摘要、失败恢复、通过保留和五轮
上限；证据为
`runs/repair-loop-tool-001`，只来自临时 Git 仓库和 synthetic replay。

Ticket 15 已回传正式 evidence。首次服务 PID `98141` 因模型加载路径错误，在模型
加载、采集 Hook 和样本产生之前退出；人明确该启动不计入 Capture Session。服务
PID `114828` 是唯一实际采集 Session，保存三个 rank-0 shape。Agent 静态核对确认
样本文件与 sidecar 记录的大小和 SHA-256 一致，本机没有反序列化 Tensor；CUDA
self-replay 的 sealed 证据通过。

`runs/cuda-formal-review-r5-002` 已 supersede 前一轮拒绝审计并接受
`runs/cuda-golden-r5-001`。随后 `runs/handoff-build-r5-001` 和
`runs/handoff-verify-cuda-r5-001` 在 CUDA 端通过，manifest 含 13 个允许文件，
SHA-256 为
`c7886622083e8516e335df6946321858ef58dfdec8f33d268aed7140eb13a83a`。
当前 Working State 为 `WAITING / HANDOFF`；下一步只在 P800 校验 manifest 并回传
verification Run，不读取 Golden Tensor、不执行 baseline，也不修改
SGLang-Kunlun。

完整机器可读扫描证据见 `runs/scan-006/result.json`；原始 `scan-005` 保留为历史。

# Step-3.7 P800 Operator Candidate 静态复核

## 结论

本次复核的统计口径是：**固定 Step-3.7 模型调用点所需要的语义能力，以及固定
Kunlun revision 上实际选择该能力的生产路径**。CUDA/Triton/`sgl_kernel` 的具体
符号只作为来源证据，不能因为 P800 上没有同名符号就直接记为算子缺口。

按这个口径，原队列中的五项都能在固定 SGLang 源码中找到 PyTorch native 语义
实现；因此五项的 `implementation_kind` 都是 `NATIVE_IMPLEMENTATION`，不是
`Confirmed Operator Gap`。五条固定 Kunlun 生产路由都还没有 P800 结果，统一记为
`ROUTE_PENDING`。

| 语义算子 | 历史 symbol（只用于追溯） | `implementation_kind` | `route_state` | `confirmed_gap` | 静态路由风险 |
|---|---|---|---|---|---|
| `step3p7.moe.swiglu_clamp` | `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul` | `NATIVE_IMPLEMENTATION` | `ROUTE_PENDING` | `false` | 已有 `kunlun_ops.swiglu` 路径，但固定源码没有装配 `gemm1_clamp_limit` |
| `step3p7.norm.gemma_rmsnorm` | `sgl_kernel.gemma_rmsnorm` | `NATIVE_IMPLEMENTATION` | `ROUTE_PENDING` | `false` | native 完整，但 Kunlun `forward_cuda` 静态上仍进入未绑定的 `sgl_kernel` symbol |
| `step3p7.norm.gemma_fused_add_rmsnorm` | `sgl_kernel.gemma_fused_add_rmsnorm` | `NATIVE_IMPLEMENTATION` | `ROUTE_PENDING` | `false` | 层级 native 完整，但优化 ABI 未绑定，且需核对原地/alias 语义 |
| `step3p7.moe.topk_sigmoid` | `sgl_kernel.topk_sigmoid` | `NATIVE_IMPLEMENTATION` | `ROUTE_PENDING` | `false` | Step-3.7 非 grouped 路径仍进入未绑定的 stub |
| `step3p7.vision.prefill_attention` | `sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel` | `NATIVE_IMPLEMENTATION` | `ROUTE_PENDING` | `false` | 默认生产路由选择 Triton；native SDPA 已存在但没有被默认选择 |

这里的 `NATIVE_IMPLEMENTATION` 只表示语义实现已经存在，**不表示当前 P800 模型
调用已经接通或已经通过精度验证**。`ROUTE_PENDING` 表示必须执行 P800 生产入口；
它不是缺口结论。五项的 `confirmed_gap` 都应保持 `false`，直到当前固定 revision 的
P800 生产调用实际失败。

## 固定范围与分类规则

固定版本来自当前 Contract：

- SGLang：`49e384ce9d304648e9959666ecb8ce8cd98d0deb`
- SGLang-Kunlun：`546ad8c682392922792bbbfe53a8bf575545f118`

检查使用 `git show <revision>:<path>` 和 `git grep <revision>`，没有切换工作树。
本次涉及的 SGLang core 文件在两个固定 revision 中是相同 blob；下文用完整 revision
和 `path:line` 标注来源，Kunlun 专有代码只引用 Kunlun revision。

分类含义：

- `NATIVE_IMPLEMENTATION`：SGLang 中已经存在覆盖模型语义边界的 PyTorch 实现；
  它可以作为算子级 CPU reference，也可以作为允许范围内的 P800 fallback 候选。
- `ROUTE_PENDING`：native 已存在，但固定 Kunlun 生产入口仍需实机确认；静态发现的
  未接通、参数丢失或 backend 选择风险只能记录在 route detail 中。
- `CONFIRMED_OPERATOR_GAP`：只允许在固定 P800 生产入口实际失败后写入。

## 1. MoE SwiGLU clamp

### 语义与模型调用点

语义边界不是必须调用 `_swiglu_silu_clamp_mul` 这个函数名，而是对 MoE gate/up
结果执行：

```text
gate, up = chunk(x, 2)
output = clamp(silu(gate), max=limit) * clamp(up, min=-limit, max=limit)
```

Step-3.7 复用 `Step3p5ForCausalLM`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/models/step3p7.py:70-74`。
`Step3p5MoEMLP` 从模型配置读取每层 `swiglu_limits`，把正值写入
`gemm1_clamp_limit`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/models/step3p5.py:137-161`。

固定 SGLang 中已有两处 PyTorch native 语义：

- shared/dense MLP 直接执行 `F.silu + clamp + multiply`：
  `49e384ce9d304648e9959666ecb8ce8cd98d0deb:python/sglang/srt/models/step3p5.py:95-107`；
- MoE runner 的 `_swiglu_silu_clamp_mul` 也是 PyTorch 组合，并在
  `gemm1_limit is not None` 时调用：
  `49e384ce9d304648e9959666ecb8ce8cd98d0deb:python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:326-332,549-571`。

因此，旧函数名应作为 native reference 的源码锚点，而不是 P800 必须实现的同名
算子。

### 固定 Kunlun 路由

Kunlun 用 REPLACE hook 替换整个 `UnquantizedFusedMoEMethod.apply`：
`546ad8c682392922792bbbfe53a8bf575545f118:sglang-kunlun/sglang_kunlun/hooks/layers/quantization/unquant.py:32-40`。
该路径已有 `kunlun_ops.swiglu`，但固定源码调用只传 `x` 和 `y`，没有读取或传递
`layer.moe_runner_config.gemm1_clamp_limit`：
`546ad8c682392922792bbbfe53a8bf575545f118:sglang-kunlun/sglang_kunlun/hooks/layers/quantization/unquant.py:100-125`。

历史固定 checkpoint 配置摘要显示 MoE 第 43、44 层的 limit 为 `7`，同时
`moe_num_experts=288`、`moe_top_k=8`；这是 SOURCE 侧配置证据，不是当前 P800
结论：`runs/scan-002/checkpoint-config-summary.json:85-134,185-197`。

### 静态分类与 P800 验证

- `implementation_kind`: `NATIVE_IMPLEMENTATION`
- `route_state`: `ROUTE_PENDING`
- route detail：`KUNLUN_CLAMP_NOT_ASSEMBLED`
- `confirmed_gap`: `false`

P800 必须验证实际 checkpoint 的 limit、当前 `kunlun_ops.swiglu` 签名，以及生产
MoE hook 是否把 limit 传到算子。比较对象是上面的 PyTorch 语义，不要求 P800 保留
`_swiglu_silu_clamp_mul` 这个函数名。只有生产路由在相同输入和 limit 下失败或超过
Contract 容差，才能创建 Confirmed Operator Gap。

## 2. Gemma RMSNorm（无 residual）

### Native 实现与模型调用点

`GemmaRMSNorm.forward_native` 已经是完整 PyTorch 实现：先转 FP32 计算方差和
`rsqrt`，乘以 `1 + weight`，再转回输入 dtype：
`49e384ce9d304648e9959666ecb8ce8cd98d0deb:python/sglang/srt/layers/layernorm.py:693-711`。
固定仓库的手工测试也明确把 `forward_native` 当 reference，与模块生产调用比较：
`546ad8c682392922792bbbfe53a8bf575545f118:test/manual/layers/test_layernorm.py:60-109`。

Step-3.7 的语言模型 attention 创建 q/k 两个 `GemmaRMSNorm`，并在 Q/K reshape 后
调用无 residual 分支：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/models/step3p5.py:370-378,433-440`。
Decoder 首层和最终 norm 也可以进入无 residual 分支：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/models/step3p5.py:570-573,706-708,764-772`。

### 为什么不是直接缺口

Kunlun OOT 平台的 dispatch key 是 `cuda`：
`546ad8c682392922792bbbfe53a8bf575545f118:sglang-kunlun/sglang_kunlun/platform/device.py:8-17`。
`MultiPlatformOp` 对 OOT 平台会按这个 key 选择 `forward_cuda`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/layers/utils/multi_platform.py:109-119`。
`GemmaRMSNorm.forward_cuda` 再进入 `_forward_impl`，调用
`sgl_kernel.gemma_rmsnorm`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/layers/layernorm.py:671-691,713-719`。

固定 Kunlun stub 只绑定了普通 `rmsnorm` 和 `fused_add_rmsnorm`，没有绑定 Gemma
两个名字；未绑定的 stub 被调用时会抛 `NotImplementedError`：
`546ad8c682392922792bbbfe53a8bf575545f118:sglang-kunlun/sglang_kunlun/bootstrap/sgl_kernel_stub.py:53-70,196-207`。
同时，Kunlun 已有普通 RMSNorm 实现：
`546ad8c682392922792bbbfe53a8bf575545f118:sglang-kunlun/sglang_kunlun/kernels/sgl_kernel_kunlun/layernorm.py:8-29`；
Gemma 类也预计算了 `weight + 1` buffer：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/layers/layernorm.py:648-669`。

这说明当前静态问题是 native 没接到 Kunlun dispatch，不是 Gemma RMSNorm 数学实现
不存在。

### 静态分类与 P800 验证

- `implementation_kind`: `NATIVE_IMPLEMENTATION`
- `route_state`: `ROUTE_PENDING`
- route detail：`FORWARD_CUDA_SYMBOL_UNBOUND`
- `confirmed_gap`: `false`

P800 先分别调用 `forward_native` 和真实 `GemmaRMSNorm(...)` 模块入口，比较实际
BF16 shape、dtype 和数值。如果 native 能运行而模块入口因 stub 失败，应记录为
Kunlun 路由适配问题，再选择接 native 或复用已有 Kunlun RMSNorm；不能回写为
“GemmaRMSNorm 无算子实现”。

## 3. Gemma add + RMSNorm（有 residual）

### Native 实现与模型调用点

同一个 `forward_native` 已覆盖 residual 语义：必要时先加
`post_residual_addition`，再计算 `x + residual`，返回 `(normalized_x,
new_residual)`：
`49e384ce9d304648e9959666ecb8ce8cd98d0deb:python/sglang/srt/layers/layernorm.py:693-711`。

Decoder 的 `LayerCommunicator` 在已有 residual 时调用 Gemma norm 的 residual 分支：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/layers/communicator.py:543-578,664-678`；
最终 norm 也显式调用 `(hidden_states, residual)`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/models/step3p5.py:764-773`。

### Native 层级语义与 direct ABI 的区别

`forward_native` 覆盖模型所观察的返回值，但它通过 Python 赋值生成新 tensor；
`sgl_kernel.gemma_fused_add_rmsnorm` 的优化接口则在 `_forward_impl` 中被当作原地算子，
随后返回原来的 `x, residual`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/layers/layernorm.py:681-688`。

所以从模型语义统计，它是 `NATIVE_IMPLEMENTATION`；若最终选择保留 direct
`sgl_kernel` ABI，原地修改和 alias 仍是独立的接口验证项。不能因为 native 没有
复制 direct ABI 的实现细节，就把整个语义算子记为缺失。

固定 Kunlun 已有普通 fused add RMSNorm，并写回 `x` 和 `residual`：
`546ad8c682392922792bbbfe53a8bf575545f118:sglang-kunlun/sglang_kunlun/kernels/sgl_kernel_kunlun/layernorm.py:32-56`；
但 stub 没有绑定 Gemma 变体，dispatch 风险与无 residual 分支相同。

### 静态分类与 P800 验证

- `implementation_kind`: `NATIVE_IMPLEMENTATION`
- `route_state`: `ROUTE_PENDING`
- route detail：`FORWARD_CUDA_SYMBOL_UNBOUND_AND_ALIAS_UNVERIFIED`
- `confirmed_gap`: `false`

P800 先以模型入口的返回结构和值为主进行比较；如果修复方案继续使用原地优化 ABI，
再额外检查 `x`、`residual`、返回值、storage alias 和 dtype。只有真实生产入口失败才
升级为 Confirmed Operator Gap。

## 4. Sigmoid TopK

### Native 实现与模型调用点

Step-3.7 MoE 明确配置 `renormalize=True`、`use_grouped_topk=False`、
`scoring_func="sigmoid"`，并传入 router correction bias：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/models/step3p5.py:122-148`。
固定 checkpoint 的 SOURCE 摘要给出 `288` experts 和 `top_k=8`：
`runs/scan-002/checkpoint-config-summary.json:185-197`。

`fused_topk_torch_native` 已完整实现这组语义：sigmoid、bias 只参与选 id、从原
sigmoid score gather 权重、最后 renormalize：
`49e384ce9d304648e9959666ecb8ce8cd98d0deb:python/sglang/srt/layers/moe/topk.py:629-667`。
SGLang CPU 测试直接导入它作为 native reference：
`546ad8c682392922792bbbfe53a8bf575545f118:test/registered/cpu/test_topk.py:5-16`；
`sgl_kernel` 测试也用相同 PyTorch 公式检查 correction bias、归一化权重和精确 ids：
`546ad8c682392922792bbbfe53a8bf575545f118:sgl-kernel/tests/test_moe_topk_sigmoid.py:162-193`。

### 固定 Kunlun 路由

`TopK.forward_native` 会设置 `torch_native=True` 再调用 `select_experts`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/layers/moe/topk.py:434-451`。
但 Kunlun dispatch key 为 `cuda`，生产入口默认执行 `TopK.forward_cuda`；当前
`auto` runner 会使用 standard output，并在 `torch_native=False` 时进入
`fused_topk -> topk_sigmoid`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/layers/moe/topk.py:453-515,733-810`。

Kunlun 的 `select_experts` hook 保留了 `torch_native` 分支：
`546ad8c682392922792bbbfe53a8bf575545f118:sglang-kunlun/sglang_kunlun/hooks/layers/moe/topk.py:60-88,130-167`。
Kunlun 也注册了 `kunlun_ops.moe_sigmoid_group_topk_norm`，但它替换的是 grouped
TopK 路径：
`546ad8c682392922792bbbfe53a8bf575545f118:sglang-kunlun/sglang_kunlun/kernels/kernel_ops.py:370-407`；
Step-3.7 当前设置是 `use_grouped_topk=False`，不能据此宣称生产路由已经接通。
固定 `sgl_kernel` stub 也没有给 `topk_sigmoid` 绑定真实实现；未知符号仍按 stub
规则失败：
`546ad8c682392922792bbbfe53a8bf575545f118:sglang-kunlun/sglang_kunlun/bootstrap/sgl_kernel_stub.py:53-70,158-173`。

### 静态分类与 P800 验证

- `implementation_kind`: `NATIVE_IMPLEMENTATION`
- `route_state`: `ROUTE_PENDING`
- route detail：`NON_GROUPED_PATH_ENTERS_STUB`
- `confirmed_gap`: `false`

P800 用固定模型的 `288 experts / top_k=8 / sigmoid / correction bias /
renormalize` 跑 `TopK.forward_native` 和真实 `TopK(...)` 入口；构造无并列输入，ids
精确相等，weights 使用 Contract 容差。native 通过但生产入口失败时，结论是
Kunlun 路由未接通，不是 sigmoid TopK 语义不存在。

## 5. Vision prefill attention

### 真实模型边界

这项只属于单图请求，不属于 text-only 路径。Step-3.7 创建
`PerceptionEncoder`，单图时执行 vision model：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/models/step3p7.py:47-77,92-110`。
Vision block 创建并调用 `VisionAttention`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/models/step3_vl_10b.py:191-224,246-249`。

Step-3.7 vision config 固定 `width=1536`、`heads=16`，因此生产 head dim 为
`96`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/configs/step3p7.py:6-17,29-35`。
当前模型调用没有传 `qkv_backend`、`cu_seqlens` 或 `num_kv_heads`。因此在这个
固定调用点：

- Q/KV 都是 `16` heads，`num_kv_heads` 默认等于 `num_heads`，不是 GQA；
- `cu_seqlens=None`，backend 生成规则的 batch sequence lengths，不是当前已证明的
  ragged 输入；
- vision attention 是 non-causal。

源码锚点：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/layers/attention/vision.py:888-986,1158-1212,1297-1320`。
所以旧队列中的通用 `causal / GQA / ragged` 用例不能直接当成 Step-3.7 必测生产
语义；只有真实模型运行证明它们被使用时才扩展。

### Native 与 Triton 实现

`VisionSdpaAttention` 已提供 PyTorch native/SDPA 实现，包含 mask、FP32 softmax
分支和 `F.scaled_dot_product_attention`：
`49e384ce9d304648e9959666ecb8ce8cd98d0deb:python/sglang/srt/layers/attention/vision.py:177-334`。
Triton backend 才会调用 `context_attention_fwd`，进而启动旧队列中的
`_fwd_kernel`：
`49e384ce9d304648e9959666ecb8ce8cd98d0deb:python/sglang/srt/layers/attention/vision.py:337-409`；
`49e384ce9d304648e9959666ecb8ce8cd98d0deb:python/sglang/srt/layers/attention/triton_ops/prefill_attention.py:34-57,170-218`。
SGLang 的 Triton attention 测试也用 PyTorch SDPA 作为 reference：
`546ad8c682392922792bbbfe53a8bf575545f118:test/registered/attention/test_triton_attention_kernels.py:453-498`。

backend 选择优先使用 `mm_attention_backend` 或构造参数；该参数源码默认是 `None`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/server_args.py:1380-1396`。
没有 override 时再根据运行时平台和 device capability 决定 `fa3`、`fa4`、
`triton_attn` 或 `sdpa`：
`546ad8c682392922792bbbfe53a8bf575545f118:python/sglang/srt/layers/attention/vision.py:962-986,1065-1108`。
Kunlun pre-shim 把 `torch.cuda.get_device_capability()` 固定为 `(8, 0)`：
`546ad8c682392922792bbbfe53a8bf575545f118:sglang-kunlun/sglang_kunlun/bootstrap/pre_shim.py:202-208`。
因此在当前没有 multimodal override 的默认生产路径上会选择 `triton_attn`，已有的
native SDPA 不会被默认选择。

Kunlun LLM attention backend 的确另有 `kunlun_ops.prefill_attention`：
`546ad8c682392922792bbbfe53a8bf575545f118:sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:1168-1239`。
但该代码消费 `RadixAttention`、KV cache 和 `ForwardBatch` metadata，不是上面的
`PerceptionEncoder -> VisionAttention` 调用点，不能拿它直接证明视觉路径已经适配。

### 静态分类与 P800 验证

- `implementation_kind`: `NATIVE_IMPLEMENTATION`
- `route_state`: `ROUTE_PENDING`
- route detail：`DEFAULT_TRITON_NATIVE_SDPA_NOT_SELECTED`
- `confirmed_gap`: `false`

P800 单图模型运行必须先记录 `type(vision_attention.qkv_backend)`，并验证真实生产
契约：`head_dim=96`、`16/16 MHA`、non-causal、dense sequence：

- 默认预期是 `VisionTritonAttention`；验证它的实际模型输入和输出 buffer，不要先
  扩大成历史通用 causal/GQA/ragged 测试。
- 如果现场显式选择了 `VisionSdpaAttention`，直接按 native 路径验证，并记录导致
  backend 改变的参数；此时旧 `_fwd_kernel` 只保留追溯，不是该 Run 的执行目标。
- 如果选择其他 backend，按实际 backend 重新建立语义调用点和实现证据。

## 建议的后续文档与状态字段

每个语义算子的验证记录至少包含：

```text
semantic_operator_id
model_callsite
native_implementation
kunlun_implementation
production_dispatch
static_classification
runtime_status
confirmed_gap
cpu_reference_command
p800_production_command
precision_result
evidence
```

状态转换应为：

```text
NATIVE_IMPLEMENTATION + ROUTE_PENDING + confirmed_gap=false
    -> P800 production verification
    -> PASS
       或
       Failure Observation
    -> Confirmed Operator Gap（仅真实 P800 生产入口失败后）
```

当前五项都是 `implementation_kind=NATIVE_IMPLEMENTATION`、
`route_state=ROUTE_PENDING`、`confirmed_gap=false`。这份静态复核不能把任何一项
写成 `PASS`，也不能
把任何一项写成 `Confirmed Operator Gap`。

# Step-3.7 实际路径上的 kernel 缺口调研

## revision 5 更新

`runs/scan-006` 已把本调研的 Triton 口径扩展为完整 kernel-call 口径：

- Triton 不再是候选硬条件；CUDA extension、SGLang JIT、第三方 kernel 和真实不
  兼容的 Torch 调用都进入扫描范围。
- Contract 不预选算子，只固定 target-only、文本/单图输入和样本参数规则。
- Golden Sample 可以保存当前 kernel 调用直接使用的参数 Tensor，但不能保存完整
  checkpoint、module `state_dict` 或无关参数。
- 首轮比较已有 `_swiglu_silu_clamp_mul`、`sgl_kernel.gemma_rmsnorm`、
  `sgl_kernel.gemma_fused_add_rmsnorm`、`sgl_kernel.topk_sigmoid` 和视觉
  `prefill_attention._fwd_kernel` 后，选择 `_swiglu_silu_clamp_mul` 作为最小
  Demo。

这个调用是 SGLang 已有的 `@torch.compile` 函数，在文本 MoE 第 43、44 层固定
接收 limit=7。它只需要 `x + limit -> output`，不保存权重，也没有边界内 TP
通信；固定 Kunlun MoE 路径调用普通 `kunlun_ops.swiglu`，没有读取 limit。
Gemma RMSNorm、`topk_sigmoid` 和视觉 attention 保留为后续缺口。

下面保留最初以 Triton 为切入口的详细源码审计。它用于解释 top-k、视觉 attention、
SWA 和上层 Kunlun bypass，不再代表当前 Demo 选择。

## 结论

最初这轮扫描把“仓库中定义了 Triton kernel”和“固定 Demo 实际会调用它”分开处理。

在当前固定的 **Step-3.7、纯文本、target-only、TP8、BF16、eager、无量化、无投机解码、不显式指定 backend** 范围内：

- **Kunlun 已经接通当前 target-only 普通 prefill/decode 所需的 Step-3.7 SWA 路径，窗口值没有在 Python 调用链中丢失。** checkpoint 的 `sliding_window=512` 进入 SWA 层的 `RadixAttention.sliding_window_size`；Kunlun prefill 最终传成 `kunlun_ops.prefill_attention(..., swa_left=512, swa_right=0)`，普通 decode 最终传给 `kunlun_ops.speculative_attention(..., max_window_size=512)`。这里的函数名不表示启用了投机解码。
- 这只能表述为**当前 Contract 的静态接线存在**，不能外推成“Kunlun 完整支持 SWA”。当前仓库没有 Step-3.7/P800 的 SWA 精度结果；chunked/prefix prefill 处还有一处注释与实参不一致。并且 Kunlun 的投机解码 SWA metadata 路径仍内置一个
  `@triton.jit` 的 `_prepare_swa_spec_page_table_kernel`；当前 Contract 因为关闭投机解码而不会触发它。
- 按用户原始条件“Step-3.7 实际算子、SGLang 仓库中有 Triton 实现、Kunlun 没有对应实现”，**首选示范是 MoE sigmoid top-k**：语义边界为 `topk_sigmoid`，SGLang 的 MUSA backend 有同语义显式 Triton kernel
  `topk_sigmoid_triton_kernel`，固定 Kunlun 插件没有给 `topk_sigmoid` 绑定实现。
- `topk_sigmoid` 属于 Step-3.7 的固定纯文本路径：MoE 层 3..44 使用 288 experts、top-k 8、sigmoid、correction bias 和 renormalization。它比整层 MLP 更小，不包含 expert 权重计算或 TP all-reduce。当前证据等级只是 `STATIC_GAP_CANDIDATE`，不能宣称 P800 已经运行失败。
- 必须同时说明另一种更严格口径：固定 CUDA 文本路径实际调用
  `sgl_kernel.topk_sigmoid`，不是 MUSA 的 Triton kernel。如果额外要求“CUDA 本次 golden 运行也必须直接启动同一个 `@triton.jit`”，则首选应改为 Step-3.7 视觉编码器的
  `sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel`；这会把当前纯文本 Contract 扩成最小图像请求。
- 视觉候选同样只是静态候选。CUDA 设备不是 Hopper/Blackwell 且未指定 multimodal backend 时才会选 `triton_attn`；P800 端还需要最小图像请求确认实际失败栈。
- 模型 `forward` 中，源码可确定会进入的显式 Triton kernel 只有 `fused_moe_kernel`。同一个 kernel 在 routed MoE 的第一层和第二层 expert GEMM 各启动一次。
- 文本 attention 的 3 个计算 kernel 只有 CUDA 启动后实际解析为 `attention_backend=triton` 才会调用，不能在没有启动日志时写成无条件可达。
- attention metadata、位置和 request-to-token 写入还会调用 Triton 辅助 kernel；它们发生在模型 `forward` 之前，不应冒充模型算子。
- Kunlun 对 FusedMoE 和文本 attention 都是在更高层改走自己的实现。因此上游 Triton 符号没有一一注册替换，不等于“Kunlun 缺少这个算子”。
- 如果 Demo 必须同时保持纯文本输入，并要求 CUDA 本次运行直接启动显式 Triton kernel，严格候选数仍是 0。

因此，这一阶段曾建议先验证 `topk_sigmoid`；revision 5 的完整 Kernel Call 比较已
由 `scan-006` 取代该建议，并选择更小的 `_swiglu_silu_clamp_mul`。Gemma
RMSNorm、视觉 `prefill_attention._fwd_kernel` 与 top-k 都继续保留在缺口队列。

## 扫描边界

固定输入来自仓库的 Scan Run：

- CUDA revision：`49e384ce9d304648e9959666ecb8ce8cd98d0deb`
- Kunlun revision：`546ad8c682392922792bbbfe53a8bf575545f118`
- checkpoint：`stepfun-ai/Step-3.7-Flash@5f6244077ac62e04eec3f320501ff8c2b293373a`
- TP8、BF16、无量化、无 speculative、decode/prefill CUDA Graph 均关闭、attention/MoE backend 参数均不传
- 当前入口明确是 `general_mm_embed_routine(text-only)`

证据：`runs/scan-004/result.json:9-35,49-67`，checkpoint 摘要见 `runs/scan-004/checkpoint-config-summary.json:1-47`。

为了寻找满足严格条件的示范算子，本文另外扫描了一条只改变输入类型的候选路径：
同一个 Step-3.7 checkpoint、同一 TP8/BF16/eager 配置，输入改为一条最小图像请求。
这条候选路径只用于判断下一版 Demo 是否值得改 Contract，不回写当前 Scan Run 的固定范围。

本文使用四个状态：

- `REACHABLE`：固定范围内源码调用链和激活条件都成立。
- `CONDITIONAL_ON_BACKEND`：模型路径成立，但只有启动后解析为特定 backend 才会调用。
- `NOT_REACHABLE`：固定范围下调用条件不成立，或不在 Step-3.7 调用链上。
- `BYPASSED_ON_KUNLUN`：Kunlun 在更高层换成自己的实现，不会执行该上游 Triton kernel。

清单只统计源码中明确的 `@triton.jit` kernel；`torch.compile` 可能生成的 GPU 代码不按一个源码 Triton kernel 计数。

“eager”在这里仅表示关闭 CUDA Graph，不表示 EAGLE。固定运行参数证据是 `runs/scan-004/result.json:18-40`。

## Step-3.7 根调用链

纯文本入口为：

```text
Step3p7ForConditionalGeneration.forward
  -> general_mm_embed_routine
  -> Step3p5ForCausalLM.forward
  -> Step3p5Model.forward
  -> Step3p5DecoderLayer.forward (0..44)
       -> Step3p5Attention.forward
       -> dense MLP 或 Step3p5MoEMLP.forward
  -> LogitsProcessor
```

代码依据：

- `Step3p7ForConditionalGeneration` 创建 `Step3p5ForCausalLM`，并在 `forward` 中调用通用多模态入口：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p7.py:47-74,136-152`。
- 纯文本请求直接做 token embedding，随后调用 language model：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/managers/mm_utils.py:1023-1055,1130-1145`。
- text model 遍历 decoder layer，最后执行 logits processor：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p5.py:719-773,855-879`。
- decoder layer 内先走 attention，再走 dense MLP 或 routed MoE：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p5.py:594-658`。

## Kunlun 的 Step-3.7 SWA 是否支持

### 结论边界

结论是：**固定 Kunlun revision 已经包含当前 target-only 普通
prefill/decode 的 SWA 模型识别、独立 KV pool、page table 转换以及窗口参数传递；没有发现 `512` 在这条 Python 参数链上被丢弃。** 这只是当前 Contract 的静态源码结论，不等价于 P800 上已经通过精度验证，也不覆盖投机解码。

### 1. 模型把 512 放进每个 SWA attention 层

固定 checkpoint 的 `sliding_window` 是 `512`，45 层中 full attention 为 `0,4,8,...,44`，其余 33 层为 sliding attention：`runs/scan-004/checkpoint-config-summary.json:20-38`。

Step-3.7 文本模型没有把所有层都当成 SWA：

- `Step3p5DecoderLayer` 先用 `layer_types[layer_id] == "sliding_attention"` 判断本层类型，SWA 层取 `config.sliding_window`，full attention 层保留 `-1`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p5.py:507-518`。
- 该值作为 `sliding_window_size` 创建 `Step3p5Attention`，再传入 `RadixAttention`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p5.py:520-540,422-430`。
- `RadixAttention` 把它保存在实例属性中，并在 `forward` 时把整个 `layer` 对象交给当前 attention backend：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/radix_attention.py:81-103,109-153`。

因此窗口值的模型侧链路是：

```text
config.sliding_window = 512
  -> Step3p5DecoderLayer.sliding_window
  -> Step3p5Attention(... sliding_window_size=512)
  -> RadixAttention.sliding_window_size = 512
  -> attention_backend.forward(..., layer=RadixAttention, ...)
```

### 2. ForwardBatch 不保存窗口，但 metadata 保留 SWA 地址信息

`ForwardBatch` 保存请求索引、序列长度和 KV 写入位置，不保存 per-layer 的窗口值：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/model_executor/forward_batch_info.py:321-350`。这不是丢失，因为 full/SWA 层共享同一个 batch，而窗口值继续保存在每个 `RadixAttention` 实例上。

运行时同时建立 SWA 所需的缓存信息：

- SGLang 把 `Step3p7ForConditionalGeneration` 明确识别成 hybrid SWA，并按 `layer_types` 计算 full/SWA layer ids：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/configs/model_config.py:1842-1875,1909-1923`。
- `ModelRunner` 从 model config 得到全局 `sliding_window_size=512`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/model_executor/model_runner.py:1468-1479`。
- Kunlun 的 `_init_pools` AFTER hook 对 hybrid SWA 模型重建 `SWAKVPool` 和 `SWATokenToKVPoolAllocator`，并传入 full/SWA layer ids：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/model_executor/model_runner_kv_cache_mixin.py:19-47,65-99`。
- `KunlunAttentionBackend` 确认 pool 是 `SWAKVPool`，并记录全局窗口存在：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:345-405`。
- 每次请求准备 metadata 时，它把 full page table 转成 `swa_page_table`，并预先转换 SWA 层 KV 写入位置：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:837-851`。

这里 metadata 负责“到哪里读写”，而 `layer.sliding_window_size` 负责“最多看多远”，两者没有合并成一个容易混淆的全局字段。

### 3. Kunlun 算子调用确实收到窗口值

prefill/extend：

- backend 按当前 `layer.sliding_window_size` 得到 SWA 层 `(512, 0)`、full 层 `(-1, -1)`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:948-957`。
- 有 prefix cache 时使用 `swa_page_table`；首次 prefill 使用本次的原始 K/V：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:1019-1049,1172-1224`。
- 两个分支最终都把窗口传给 `kunlun_ops.prefill_attention`，实参为 `swa_left=window_size[0]`、`swa_right=window_size[1]`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:1197-1235`。

decode：

- backend 同样按本层得到 `(512, 0)`，并在 SWA 层切换到 `swa_page_table`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:1478-1484,1559-1569`。
- 最终调用 `kunlun_ops.speculative_attention` 时，`max_window_size` 直接取 `layer.sliding_window_size`，所以 SWA 层传 `512`、full 层传 `-1`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:1597-1618`。函数名虽然叫 `speculative_attention`，这里也是普通单 token decode 的实际 Kunlun 算子，并不表示启用了 EAGLE。

### 4. 还不能说“实机已验证”

现有源码没有给出 Step-3.7/P800 的 SWA 精度测试。并且 prefix-cache prefill 分支的注释写着 translated page table 已裁剪窗口、应传 `swa_left=0`，实际代码仍传 `window_size[0]`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:1186-1214`。这可能只是过期注释，也可能是二次裁剪风险，仅凭 Python 无法确认 `kunlun_ops.prefill_attention` 的内部约定。

还要排除一个容易误读的范围：Kunlun plugin 在投机解码的 SWA
page-table 准备路径中，直接定义并启动
`@triton.jit _prepare_swa_spec_page_table_kernel`：
`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:2998-3038,3052-3148`。P800 不支持上游 Triton，因此不能由普通 prefill/decode 的接线推出“投机 SWA 也支持”。当前
target-only、无投机解码 Contract 不进入这条分支，所以它不阻塞当前 Demo，也不进入本轮候选清单。

所以当前状态应写成：

- 当前 target-only 普通 prefill/decode SWA：`STATIC_IMPLEMENTATION_PRESENT`。
- speculative SWA：`OUT_OF_SCOPE_WITH_STATIC_TRITON_RISK`。
- 窗口值在 Python 链路丢失：`NOT_OBSERVED`。
- P800 长上下文精度：`RUNTIME_CONFIRMATION_REQUIRED`。
- 最小确认样例：至少一条长度大于 512 的文本请求，同时比较 SWA 层与 full-attention 层输出；短于等于 512 的输入无法证明窗口裁剪正确。

## revision 5 首选：已有 `_swiglu_silu_clamp_mul`

这不是重新抽取的 helper，而是固定 CUDA revision 中已经存在的
`@torch.compile` 调用：

```python
@torch.compile
def _swiglu_silu_clamp_mul(x, gemm1_limit):
    gate, up = x.chunk(2, dim=-1)
    gate = F.silu(gate)
    gate = gate.clamp(max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * up
```

Step-3.7 在 routed MoE 第 43、44 层把 `gemm1_clamp_limit=7` 传入 FusedMoE；
CUDA legacy Triton runner 继续把该值传到上面的调用：

- 模型配置传递：
  `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p5.py:118-158`；
- runner 传递：
  `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/moe/moe_runner/triton.py:132-166`；
- 函数定义和实际调用：
  `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:326-333,551-571`；
- checkpoint 中第 43、44 层的 routed limit：
  `runs/scan-004/checkpoint-config-summary.json:34-47`。

固定 Kunlun revision 在 `UnquantizedFusedMoEMethod.apply` 上层替换整段 MoE，
但激活处只执行 `kunlun_ops.swiglu(x=y, y=out1)`，没有读取
`gemm1_clamp_limit`：
`sglang-kunlun/sglang_kunlun/hooks/layers/quantization/unquant.py:100-125`。

因此可比较的现有调用边界是：

```text
CUDA: _swiglu_silu_clamp_mul(x, gemm1_limit) -> output
P800 baseline: 当前 kunlun_ops.swiglu 路径使用同一 x -> actual
```

Golden Sample 只保存 `x`、标量 `gemm1_limit` 和 CUDA output，不需要 expert
权重、checkpoint 或 module state。它比 Gemma RMSNorm 少一个直接参数 Tensor，
比 top-k 少一个输出及排序语义，也比视觉 attention 少 q/k/v metadata，因此成为
revision 5 的最小 Demo。这个结论仍是静态源码结论，必须由 CUDA 真实样本和 P800
baseline 验证。

## 原始条件首选：MoE `topk_sigmoid`

### 为什么它属于 Step-3.7 实际文本路径

Step-3.7 的 MoE 层不是普通 softmax top-k。模型固定创建：

```python
self.topk = TopK(
    top_k=8,
    renormalize=True,
    use_grouped_topk=False,
    scoring_func="sigmoid",
    correction_bias=self.router_bias,
)
```

随后每次 MoE forward 都执行 `self.topk(hidden_states, router_logits)`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p5.py:122-161,208-233`。checkpoint 说明 MoE 层为 3..44、experts 为 288、top-k 为 8、router bias 已开启：`runs/scan-004/checkpoint-config-summary.json:24-31,34-38`。

CUDA 的实际链路是：

```text
Step3p5MoEMLP.forward_normal
  -> TopK.forward_cuda
  -> select_experts
  -> fused_topk(scoring_func="sigmoid")
  -> sgl_kernel.topk_sigmoid
```

证据：

- `TopK.forward_cuda` 进入 `select_experts`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/moe/topk.py:453-510`。
- 非 grouped、无自定义 routing 的分支进入 `fused_topk`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/moe/topk.py:1729-1811`。
- `fused_topk` 的 sigmoid 分支直接调用 `topk_sigmoid`，传入 correction bias 和 renormalize：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/moe/topk.py:733-810`。
- CUDA/XPU 的 `topk_sigmoid` 来自 `sgl_kernel`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/moe/topk.py:176-182`。

### SGLang 已有同语义 Triton kernel

SGLang 的 MUSA backend 提供了显式 `@triton.jit` 实现：

- kernel：`sglang.srt.hardware_backend.musa.kernels.topk.topk_sigmoid_triton_kernel`
- Python launcher：`sglang.srt.hardware_backend.musa.kernels.topk.topk_sigmoid`

它逐行完成 sigmoid、加 correction bias 后选 expert、取未加 bias 的 sigmoid 值作为 weight、再按需 renormalize，和 Step-3.7 的参数完全对应：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/hardware_backend/musa/kernels/topk.py:159-300`。

这份实现支持本模型所需的 288 experts 和 top-k 8：launcher 把 expert 宽度补到下一个 2 的幂，并把 `K` 固定为调用时的 top-k，没有 `num_experts <= 256` 限制。

### Kunlun 当前没有对应绑定

Kunlun 给 `select_experts` 注册了替换，但在 Step-3.7 这组参数上仍调用上游 `fused_topk`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/moe/topk.py:60-72,98-167`。

Kunlun 在 SGLang 导入前把 `sgl_kernel` 换成 stub：

- 未替换的 symbol 被调用时会抛 `NotImplementedError`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/bootstrap/sgl_kernel_stub.py:36-70`。
- stub 只给 top-k 相关的 `fast_topk` 和 `moe_fused_gate` 绑定实现，没有绑定 `topk_sigmoid`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/bootstrap/sgl_kernel_stub.py:158-172`。
- 在整个 Kunlun plugin 中搜索不到 `topk_sigmoid` 的 registry replacement。

因此静态调用链是：

```text
Kunlun Step3p5MoEMLP
  -> hooked select_experts_kunlun
  -> upstream fused_topk
  -> sgl_kernel.topk_sigmoid stub
  -> 若目标镜像没有额外绑定，stub 调用会抛 NotImplementedError
```

最后一步必须用 P800 实机确认，原因是目标机可能带有仓库外的额外 patch 或二进制绑定。没有实机错误栈前，状态是
`STATIC_GAP_CANDIDATE`，不是 `RUNTIME_CONFIRMED_GAP`。

### 捕获和重放边界

- 缺口算子：`topk_sigmoid`。
- 可参考 Triton kernel：`sglang.srt.hardware_backend.musa.kernels.topk.topk_sigmoid_triton_kernel`。
- 捕获 seam：wrap 已有 `TopK.forward`（必要时再落到已有
  `fused_topk`），不要新增自定义算子函数，也不要直接 hook MUSA kernel。固定 CUDA 实际调用的是 `sgl_kernel.topk_sigmoid`，直接 hook MUSA kernel 捕获不到真实样本。
- 每条样本记录 `layer_id`，P800 replay 在同一个 checkpoint 的对应
  `TopK` 实例上执行，避免把不同 MoE 层混在一起。
- 输入保存 `gating_output`；`correction_bias` 是该 kernel 直接使用的参数 Tensor，
  按 revision 5 规则可以进入 Golden Sample。不保存完整 checkpoint、`TopK`
  module state 或其他 MoE 权重。
- 输出：`topk_weights` 和 `topk_ids`。weights 用精度阈值比较，ids 应做精确相等比较。

它比 `Step3p5MoEMLP.forward` 更适合作为最小 Demo：没有两层 expert 权重、没有 TP all-reduce，也不需要保存整层中间 Tensor；同时不需要把输入改成图像请求。

它符合用户原始表述“实际 Step 算子在 SGLang 仓库中有同语义 Triton
实现”。限制也必须写进 Contract：固定 CUDA 路径使用的是 CUDA
`sgl_kernel.topk_sigmoid`，仓库里的 Triton 版本属于 MUSA backend；它不符合额外的“CUDA 本次运行实际启动 Triton”条件。

## 当前纯文本模型 `forward` 内的 Triton kernel

| kernel | CUDA 状态 | 激活条件 | Kunlun 状态 | 是否适合当前 Demo |
|---|---|---|---|---|
| `fused_moe_kernel` | `REACHABLE`（源码已解析，启动日志仍应留证） | MoE 层 3..44；BF16 未量化；MoE runner 默认 `auto` | `BYPASSED_ON_KUNLUN` | 否。Kunlun 已在 method 层替代 |
| `extend_attention._fwd_kernel` | `CONDITIONAL_ON_BACKEND` | 文本 prefill/extend 且 CUDA 最终为 `triton` | `BYPASSED_ON_KUNLUN` | 否 |
| `decode_attention._fwd_grouped_kernel_stage1` | `CONDITIONAL_ON_BACKEND` | 文本 decode、CUDA 为 `triton`、TP8 GQA | `BYPASSED_ON_KUNLUN` | 否 |
| `decode_attention._fwd_kernel_stage2` | `CONDITIONAL_ON_BACKEND` | 与上一项相同，负责合并 split-KV 结果 | `BYPASSED_ON_KUNLUN` | 否 |

### 1. `fused_moe_kernel`

调用链：

```text
Step3p5MoEMLP.forward_normal
  -> FusedMoE.forward
  -> UnquantizedFusedMoEMethod.apply / forward_cuda
  -> MoeRunner(TRITON)
  -> fused_experts_none_to_triton
  -> fused_experts
  -> invoke_fused_moe_kernel
  -> fused_moe_kernel
```

代码依据：

- Step-3.7 的 MoE 层计算 router、top-k，再调用 `self.experts`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p5.py:110-161,180-241`。
- 未量化路径在没有选 FlashInfer、DeepGEMM 或新 Triton-kernels runner 时，选择 legacy `MoeRunnerBackend.TRITON`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/quantization/unquant.py:405-421`。
- BF16 路径构造 `TritonMoeQuantInfo` 并调用 runner：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/quantization/unquant.py:455-563`。
- runner 注册函数进入 `fused_experts`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/moe/moe_runner/triton.py:175-244`。
- kernel 定义和实际 launch 分别在 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py:323-413,714-958`，实际 launch 在 `:907`。
- `fused_experts` 对 w13 和 w2 分别调用一次 launcher：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:358-529,680-731`。这是同一个 kernel 符号的两次启动，不应计成两个算子。

Kunlun 并不是等待这个 kernel 失败。插件直接替换 `UnquantizedFusedMoEMethod.apply`，走 `kunlun_ops.moe_fc -> kunlun_ops.swiglu -> kunlun_ops.moe_fc -> kunlun_ops.moe_post`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/quantization/unquant.py:32-40,89-155`。因此状态是 `BYPASSED_ON_KUNLUN`。

这条路径仍可能存在 clamp 语义差异，但那是已有 Kunlun MoE 实现的语义问题，不是“缺少上游 Triton kernel”。

### 2. 文本 attention 的 3 个计算 kernel

`Step3p5Attention.forward` 经 `RadixAttention` 委托给运行时 attention backend：

- 模型侧 q/k/v、RoPE、`RadixAttention` 调用：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p5.py:413-467`。
- eager 下 `RadixAttention.forward` 直接调用当前 backend：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/radix_attention.py:109-153`。
- backend 再按 decode 或 extend 分派：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/base_attn_backend.py:158-201`。

A800 不是 Hopper/Blackwell，因此 Step-3.7 特例不会固定成 FA3/FA4；通用默认解析是“FlashInfer 可用则 FlashInfer，否则 Triton”：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/server_args.py:3985-4000,4407-4455,4483-4491`。所以在拿到 CUDA 启动日志前，这三项都必须保留为 `CONDITIONAL_ON_BACKEND`。

若最终为 `triton`：

- extend/prefill 调 `extend_attention_fwd`，它 launch `extend_attention._fwd_kernel`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_backend.py:1019-1183`；kernel 与 launcher 见 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_ops/extend_attention.py:239-570,571-676`。
- decode 的 TP8 配置每 rank 为 full attention `q=8, kv=1`，sliding attention `q=12, kv=1`，所以 `kv_group_num != 1`，进入 grouped 分支。head 配置见 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/configs/step3p5.py:10-30`、`runs/scan-002/checkpoint-config-summary.json:192-198`，TP 切分见 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p5.py:353-372`。
- grouped decode 先 launch `_fwd_grouped_kernel_stage1`，再 launch `_fwd_kernel_stage2`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_ops/decode_attention.py:252-519,522-647,745-810`；backend 调用见 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_backend.py:1322-1413`。
- 普通 `_fwd_kernel_stage1` 只用于 `kv_group_num == 1`，在这组 TP8 head 配置下为 `NOT_REACHABLE`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_ops/decode_attention.py:44-249,769-810`。

Kunlun 的默认 backend 是 `kunlun`，由 factory 创建 `KunlunAttentionBackend`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/platform/srt.py:61-74`，`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/attention_registry.py:19-25`。其 extend 和 decode 分别调用 `kunlun_ops.prefill_attention` 与 `kunlun_ops.speculative_attention`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:904-970,1186-1240,1418-1462,1597-1622`。因此这些文本 attention kernel 都是 `BYPASSED_ON_KUNLUN`。

## 严格 CUDA-actual 口径候选：视觉 `prefill_attention._fwd_kernel`

完整 `operator_id`：
`sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel`。

当前纯文本请求在 `general_mm_embed_routine` 中直接做 token embedding，因此视觉分支不执行：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/managers/mm_utils.py:1050-1095,1130-1144`。但 Step-3.7 本身包含视觉编码器；输入一条图像请求后，它就是模型的实际执行路径，而不是从别的模型借来的样例。

含图像输入时，完整链路是：

```text
Step3p7ForConditionalGeneration.forward
  -> general_mm_embed_routine
  -> Step3p7ForConditionalGeneration.get_image_feature
  -> PerceptionEncoder
  -> PerceptionEncoderVisionBlock
  -> VisionAttention
  -> VisionTritonAttention.forward
  -> context_attention_fwd
  -> prefill_attention._fwd_kernel
```

代码依据：

- Step-3.7 创建视觉编码器，并把 `get_image_feature` 传给通用 embedding 入口：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p7.py:56-77,92-110,136-151`。
- `PerceptionEncoder` 逐层调用含 `VisionAttention` 的 block：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3_vl_10b.py:214-224,246-248,271-293,395-420`。
- CUDA 设备不是 Hopper/Blackwell、且未显式指定
  `--mm-attention-backend` 时，视觉 attention 默认选择 `triton_attn`；映射目标是 `VisionTritonAttention`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/vision.py:862-872,962-986,1065-1099`。
- `VisionTritonAttention.forward` 调 `context_attention_fwd`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/vision.py:337-409`。
- `context_attention_fwd` launch 显式
  `@triton.jit` 的 `prefill_attention._fwd_kernel`；视觉调用固定
  `is_causal=False`：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py:34-168,170-219`。

Kunlun 的设备类型对上游报告为 `cuda`，pre-shim 把设备能力报告为 `(8, 0)`，其 utility hook 也返回 `(8, 0)`：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/platform/device.py:8-17`，`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/bootstrap/pre_shim.py:193-208`，`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/utils/common.py:249-255`。按上面的视觉 backend 选择逻辑，它也会选 `triton_attn`。在 Kunlun 插件中未找到 `VisionAttention`、`context_attention_fwd` 或这个 `_fwd_kernel` 的替代注册。

这使它同时满足三个静态筛选条件：

1. 它属于 Step-3.7 的实际图像路径。
2. 符合设备与 backend 条件时，CUDA 本次运行会直接启动该 Triton kernel。
3. 固定 Kunlun revision 中没有视觉 attention 的 native 分派或同层替换，按现有选择逻辑会误入 P800 不支持的 Triton 路径。

### 捕获和修复边界

- 算子身份保持原始 kernel，不新增自定义算子函数。
- 打桩放在现有 Python launcher `context_attention_fwd`。它紧贴 kernel，已经完整接收
  `q/k/v/output`、序列起点、序列长度、最大长度和 softmax scale；直接 hook
  `_fwd_kernel` 本身反而会把 Triton grid、stride 和编译常量混入采集协议。
- Kunlun 已在文本 attention backend 中使用
  `kunlun_ops.prefill_attention`，说明目标侧存在 native prefill attention 能力：
  `/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/hooks/layers/attention/kunlun_backend.py:1197-1235`。因此首选修复方向是让视觉 launcher 在 Kunlun 上分派到该 native op，并适配非因果和序列 LOD 参数；不是移植整个视觉层。两者的参数能否完全对应仍需在 P800 实现阶段验证。

### 证据等级

- 当前纯文本 Contract：`NOT_REACHABLE`。
- 最小图像请求：`STATIC_GAP_CANDIDATE`。
- 升级为 `RUNTIME_CONFIRMED_GAP` 的条件：CUDA 日志确认
  `mm_attention_backend=triton_attn`，P800 请求的失败栈落在
  `VisionTritonAttention/context_attention_fwd`，并排除目标镜像中的仓库外 patch。

## 模型 `forward` 前的 Triton 辅助 kernel

这些 kernel 是一次真实请求路径的一部分，但不是从 `Step3p7ForConditionalGeneration.forward` 往下调用的模型算子。

| kernel | CUDA 状态 | 所在阶段 | Kunlun 处理 |
|---|---|---|---|
| `create_flashinfer_kv_indices_triton` | `REACHABLE`（A800 默认的 FlashInfer 或 Triton backend 都使用） | attention metadata 准备 | attention backend 整体绕行 |
| `get_num_kv_splits_triton` | `CONDITIONAL_ON_BACKEND` | Triton decode metadata | attention backend 整体绕行 |
| `compute_position_kernel` | extend/prefill `REACHABLE` | 构造 token positions | 有精确 registry replacement |
| `write_req_to_token_pool_triton` | extend/prefill `REACHABLE` | scheduler 写 request-to-token 映射 | 有精确 registry replacement |

证据：

- `create_flashinfer_kv_indices_triton` 的定义在 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_ops/kv_indices.py:8-45`。Triton backend 的 normal decode/extend metadata 调用在 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_backend.py:335-353,581-740,1591-1627`；FlashInfer backend 也调用它：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/flashinfer_backend.py:1303-1338,1674-1714`。
- `get_num_kv_splits_triton` 的定义和调用分别在 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_ops/metadata.py:11-60`、`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_backend.py:283-333,592-654`。
- `compute_position_kernel` 的定义和 launcher 在 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/model_executor/triton_ops/position.py:6-59`；extend batch 准备调用在 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/model_executor/forward_batch_info.py:800-829,1496-1512`。Kunlun 精确替代在 `/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/kernels/kernel_ops.py:1364-1383`。
- `write_req_to_token_pool_triton` 定义在 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/mem_cache/triton_ops/common.py:8-57`；Kunlun 精确替代在 `/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/kernels/kernel_ops.py:167-188`。
- Kunlun registry 会替换原模块符号和已经导入的别名：`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/kernels/kernel_ops.py:119-164`。所以这里不能只搜索同名 Python 文件就判断未实现。

## 明确排除的 kernel

### `_flash_attn_fwd_with_block_score_kernel`

状态：`NOT_REACHABLE`。

这个 kernel 的定义和 launcher 只在 MiniMax sparse prefill 模块中：

- `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/minimax_sparse_ops/prefill/flash_with_topk_idx.py:50-75,475-510`
- 它的业务入口明确是 `minimax_sparse_prefill`，文档写的是 MiniMax-M3 sparse prefill：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/minimax_sparse_ops/minimax_sparse.py:9-15,48-86`

Step-3.7 文本路径使用 `Step3p5Attention -> RadixAttention`，视觉路径使用 `PerceptionEncoder -> VisionAttention`；两条链都没有进入 `minimax_sparse_ops`。因此不能因为仓库里存在这个 kernel 就把它列为 Step-3.7 实际候选。

### 其他容易误报的项

- deterministic extend 的 `_copy_unified_indices_kernel` 和 `_fwd_kernel_unified`：server arg 默认关闭 deterministic，backend 仅在开启时进入 unified 分支，当前为 `NOT_REACHABLE`。证据：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_backend.py:1096-1100,1185-1318`，kernel 在 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/triton_ops/extend_attention.py:133-236,716-1081`。
- decode 普通 `_fwd_kernel_stage1`：TP8 下 Step-3.7 是 GQA，进入 grouped stage1，当前为 `NOT_REACHABLE`。
- FlashInfer `merge_state_triton`：TP8 后每 rank 只有 8 或 12 个 q heads，未达到 fallback 阈值，当前为 `NOT_REACHABLE`。选择条件见 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/attention/flashinfer_backend.py:79-116,1019-1071`。
- MoE `act_and_mul_kernel`：Step-3.7 当前 routed expert 路径不进入它；普通层激活使用 CUDA JIT，43/44 层 clamp 使用 `torch.compile`，不是显式 Triton kernel。证据：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:549-632,740-771`。
- logits `softcap_inplace_logits_kernel`：只有 checkpoint 设置 `final_logit_softcapping` 时才调用；固定 revision 的 [Step-3.7 config](https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/5f6244077ac62e04eec3f320501ff8c2b293373a/config.json) 没有该字段，当前为 `NOT_REACHABLE`。条件代码见 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/logits_processor.py:287-294,830-872`，kernel 在 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/triton_ops/softcap.py:71-119`。
- RoPE：Step-3.7 文本 RoPE 调 CUDA C++ JIT `apply_rope_with_cos_sin_cache_inplace`，不是 Triton；Kunlun 有对应 method/JIT replacement。模型调用见 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p5.py:413-440`，Kunlun replacement 见 `/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/kernels/kernel_ops.py:1386-1415`。
- top-k router：固定配置使用 sigmoid+bias 的 `topk_sigmoid`，来源是 `sgl_kernel`，不是 `@triton.jit`。模型配置与调用见 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/models/step3p5.py:137-148,208-233`，top-k 分派见 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/moe/topk.py:733-810,1647-1811`。
- paged allocator 的 `alloc_extend_kernel` / `alloc_decode_kernel`：固定 CUDA 默认 `page_size=1`，不走上游 paged Triton allocator；Kunlun 默认 `page_size=128`，但 platform 直接返回自己的 allocator。证据：`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/server_args.py:4806-4827`，`/tmp/sglang-step3p7-kunlun-original/sglang-kunlun/sglang_kunlun/platform/srt.py:61-68,112-115`。
- `get_last_loc_kernel` / `_get_last_loc_safe_kernel`：上游实际调用点只在 EAGLE 和 dFlash 信息准备中；当前 target-only、无 speculative，因此为 `NOT_REACHABLE`。调用点见 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/speculative/eagle_info_v2.py:89`、`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/speculative/dflash_info_v2.py:235`。
- sampling 的 `murmur_hash32_kernel`：只在模型 `forward` 之后、非 greedy 且请求提供 `sampling_seed` 的 deterministic multinomial 路径触发。kernel 与调用条件见 `/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/utils/hash.py:50-120`、`/tmp/sglang-step3p7-cuda-original/python/sglang/srt/layers/sampler.py:602-663`。当前扫描边界没有固定这些请求条件，所以不列为 Step-3.7 模型算子候选。

## 历史推荐与当前决策

1. SWA 不作为当前缺口候选：只确认 target-only 普通 prefill/decode
   的静态接线，下一步补长上下文 P800 精度证据；不要外推到投机 SWA。
2. 最初按 Triton 切入口推荐 `topk_sigmoid`；当前 `scan-006` 已在更完整的
   Kernel Call 范围内选择已有 `_swiglu_silu_clamp_mul`。
3. `topk_sigmoid` 后续样本允许保存直接参数 `correction_bias`，但不保存完整
   checkpoint 或其他 MoE 权重。CUDA actual 使用 `sgl_kernel.topk_sigmoid`，MUSA
   Triton kernel 只是仓库中的同语义参考。
4. 只有额外要求 CUDA actual 必须启动 Triton 时，才改验视觉
   `prefill_attention._fwd_kernel`；这需要最小图像请求和一次明确的 Contract 变更。

## 仍需运行时确认

- CUDA 启动后文本 attention 最终是 `flashinfer` 还是 `triton`。
- CUDA 启动后 MoE runner 的日志是否与源码解析出的 legacy Triton runner 一致。
- 含图像请求时，两端实际打印的 multimodal attention backend。
- P800 对视觉 `context_attention_fwd` 的具体失败点和错误信息。
- 真实图像请求下最多三种 shape 的选择；本轮只做静态路径扫描，没有采集 tensor 或 shape。

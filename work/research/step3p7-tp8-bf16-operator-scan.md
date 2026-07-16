# Step-3.7-Flash TP8 BF16 算子扫描记录

## 扫描输入

本记录只覆盖已经确认的最小 Demo 分支：

- CUDA SGLang revision：`6274831d9fef7bba04eb59302caac24563a974c9`；其父提交 `49e384ce9d304648e9959666ecb8ce8cd98d0deb` 属于 `kunlun-0.5.14`。
- SGLang-Kunlun revision：`4731f8051b7d0bf2f03cf88e237e7e5fba80a5a9`；其父提交为 `kunlun-0.5.14` 的 `546ad8c682392922792bbbfe53a8bf575545f118`。
- 两侧 revision 都含相同的特殊 SwiGLU helper 补丁；补丁 SHA-256 为 `4fe7dc612b1c59c2eacbbf9ea4e7573d4dc632f79e36c4d4673b574796867ed0`。
- checkpoint：`stepfun-ai/Step-3.7-Flash@5f6244077ac62e04eec3f320501ff8c2b293373a`；`config.json` SHA-256 为 `8d740ba5819e574b7a606ea7aa6d7d381142ed5cc77cab96dde2892c10414042`。
- CUDA 与 P800 使用同一组模型启动参数：TP8、BF16、EAGLE、纯文本；不传量化参数、不额外传 MTP 开关、不显式传 attention backend。
- Precision Gate：`torch.testing.assert_close`，`atol=0.01`、`rtol=0.02`。

`attention_backend` 未显式指定，不代表两端最终使用相同实现。CUDA 端由硬件和已安装能力解析为 FA3、FlashInfer 或 Triton 等实现；P800 是 out-of-tree 平台，默认解析为 `kunlun`。正式运行证据必须记录两端解析后的值，不能把 Contract 中的空值伪装成某个 backend。

## EAGLE 与 draft 路径

本 Demo 不额外传 MTP 开关，但仍必须扫描 `Step3p5MTP`：

1. Step-3.7 检测到 `speculative_algorithm == "EAGLE"` 后，SGLang 自动把 `enable_multi_layer_eagle` 设为 `True`：`python/sglang/srt/server_args.py:3985-4003`。
2. worker 选择随后进入 `MultiLayerEagleWorkerV2`：`python/sglang/srt/speculative/spec_info.py:216-228`。
3. 该 worker 创建 `is_draft_worker=True`、`is_multi_layer_eagle=True` 的 draft worker：`python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py:125-168`。
4. Step-3.7 作为 draft 加载时，配置被改写为文本侧 `Step3p5MTP`：`python/sglang/srt/configs/model_config.py:564-571`。

所以“命令不额外开启 MTP”只描述 CLI；运行时的 EAGLE draft 仍由 `Step3p5MTP` 实现。

## 加载后配置激活结果

固定配置快照确认：

- `torch_dtype=bfloat16`，且没有量化配置；target 与 draft 均走未量化 Linear/MoE 分派。
- target 有 45 层；0-2 为 Dense MLP，3-44 为 MoE。
- EAGLE draft 使用额外的 layer 45，为 Dense MLP。
- target 的 0、4、8、...、44 为 full attention，其余为 sliding attention；draft layer 45 为 sliding attention。
- `use_head_wise_attn_gate=true`。
- MoE 为 288 experts、top-k 8、sigmoid 路由、router bias、FP32 gate。
- target layer 43/44 的 routed expert clamp limit 为 7；shared expert clamp limit 为 16；draft layer 45 的 clamp limit 为 0。
- hidden size 为 4096，vocab size 为 128896，RMSNorm epsilon 为 `1e-5`。

## 实际模型调用链

### Target

```text
Step3p7ForConditionalGeneration.forward
  -> general_mm_embed_routine（纯文本跳过 vision）
  -> Step3p5ForCausalLM.forward
  -> Step3p5Model.forward
  -> Step3p5DecoderLayer.forward（layer 0..44）
  -> LogitsProcessor.forward
```

入口依据：`python/sglang/srt/models/step3p7.py:47-74,136-152`；DecoderLayer 内的 norm、attention、Dense/MoE 和 residual 路径见 `python/sglang/srt/models/step3p5.py:480-665`。

### Draft

```text
Step3p5MTP.forward
  -> Step3p5AMultiTokenPredictor.forward
  -> Step3p5DecoderLayer.forward（固定 layer 45）
  -> SharedHead / LogitsProcessor
```

依据：`python/sglang/srt/models/step3p5_mtp.py:61-127,141-179`。BF16 下外层 `quant_config=None`，因此旧扫描中“draft 未传 INT8 quant_config”的阻塞不再存在。

## 完整 Semantic Operator 清单

表中的边界按可独立理解和替换的计算划分，不把每个 `view`、`chunk` 或逐元素表达式另列成算子。

| operator_id | 路径与激活条件 | 输入输出边界、权重/状态 | CUDA 实现证据 | Kunlun 实现证据 | replay / replace 判断 | verdict |
|---|---|---|---|---|---|---|
| `embedding.vocab_parallel` | target 与 draft；纯文本；TP8 | token ids + embedding weight -> hidden states；含 TP all-reduce | `vocab_parallel_embedding.py:291-339,495-529` | 共用 embedding 逻辑，Kunlun 接管 TP group | 需要权重，当前不采集；实现链完整 | `READY` |
| `linear.bf16` | qkv/o、Dense/shared MLP、router、draft `eh_proj` | activation + BF16 weight -> output | `quantization/unquant.py:110-162` 使用 `F.linear` | P800 魔改 Torch 执行同一 PyTorch 调用 | 需要权重，当前不采集；没有发现专用语义缺口 | `READY` |
| `norm.gemma_rms` | target/draft 的 layer norm、q/k norm、final norm | x、可选 residual、norm weight、epsilon -> normalized x | `layernorm.py:648-705` 调用 `gemma_rmsnorm` / `gemma_fused_add_rmsnorm` | Kunlun dispatch key 仍为 `forward_cuda`：`platform/device.py:8-17`；stub 只安装普通 `rmsnorm`：`bootstrap/sgl_kernel_stub.py:196-207`，没有 Gemma 名称 | 已确认 P800 落点缺失；重放又需要禁止保存的 norm weight | `NEEDS_HUMAN` |
| `rope.step3p5` | target 与 draft 的 full/sliding attention | positions + q/k -> rotated q/k | 调用点 `step3p5.py:410-437` | `hooks/layers/rotary_embedding.py:12-51` 有 replacement | 当前参数组合被 replacement 覆盖 | `READY` |
| `attention.radix` | target full/sliding；draft sliding；EAGLE | q/k/v + KV 状态 + batch metadata -> attention output | 未指定 backend 时走 `server_args.py:4407-4455,4483-4491` 的默认解析 | P800 默认 `kunlun`，draft prefill/decode 工厂见 `platform/srt.py:73-88` 与 `hooks/speculative/draft_utils.py:29-68` | 两端 backend 都有落点，但 CUDA 具体分支必须由启动日志补证；不把它判成已确认 gap | `NEEDS_HUMAN` |
| `activation.silu_and_mul` | Dense 0-2、MoE shared 3-42、draft 45 | gate_up -> activated product；无权重 | `layers/activation.py:90-107` | `kernels/kernel_ops.py:329-358` replacement | 无权重、可独立重放；两侧已有实现 | `READY` |
| `activation.step_swiglu_with_limit` | target shared expert layer 43/44，limit=16 | gate_up + scalar limit -> clamped SwiGLU output；无权重 | helper `models/step3p5_ops.py:5-13`，调用点 `step3p5.py:95-104` | 当前无专用 Kunlun replacement；默认落到 P800 Torch 表达式 | 可采输入和 CUDA 输出，最多三种真实 shape；Demo 首选候选 | `CAPTURE_REQUIRED` |
| `moe.router_fp32` | target MoE layer 3-44 | hidden states + BF16 router weight -> FP32 logits | `step3p5.py:205-221` 的 FP32 `torch.matmul` | P800 魔改 Torch 执行同一表达式 | 需要权重，当前不采集；实现链完整 | `READY` |
| `moe.topk_sigmoid_bias` | target MoE layer 3-44；top-k=8、sigmoid、bias、renormalize | router logits + router bias -> expert ids/weights | 配置与调用 `step3p5.py:119-145,205-221`；upstream `topk.py:1803-1810` 进入 fused top-k | Kunlun `select_experts` replacement 最终仍调用 upstream `fused_topk`：`hooks/layers/moe/topk.py:60-167`；现有 stub 未补 sigmoid+bias 的 CUDA symbol | 已确认 P800 落点缺失；重放需要禁止保存的 router bias | `NEEDS_HUMAN` |
| `moe.bf16_unclamped` | target MoE layer 3-42 | hidden states + top-k + BF16 expert weights -> output | `UnquantizedFusedMoEMethod.forward_cuda` 与 MoE runner：`quantization/unquant.py:455-615` | `hooks/layers/quantization/unquant.py:32-155` 完整替换为 `kunlun_ops` MoE pipeline | 需要权重，当前不采集；无 clamp 时两侧语义链完整 | `READY` |
| `moe.bf16_clamped` | target MoE layer 43/44，expert limit=7 | 同上，且第一段激活必须 clamp | limit 从 `step3p5.py:134-158` 进入 runner，Triton 调用传 `gemm1_limit`：`moe_runner/triton.py:145-165` | Kunlun replacement 在 `hooks/layers/quantization/unquant.py:122-125` 只调用普通 `kunlun_ops.swiglu`，未读取 limit | 已确认语义缺口；现有边界包含禁止保存的 expert weights | `NEEDS_HUMAN` |
| `collective.tp8` | target/draft；TP8 | shard tensor -> all-reduce/all-gather result | `distributed/communication_op.py:18-47`；embedding 调用 `vocab_parallel_embedding.py:520-528` | `hooks/distributed/parallel_state.py:47-92` 接管 group | 运行态通信，不做离线算子采集；两侧有实现 | `READY` |
| `logits.lm_head` | target 与 draft；不返回 logprobs | final hidden + BF16 vocab weight -> TP logits | `logits_processor.py:304-381,830-917` | 共用 BF16 matmul 与 Kunlun TP group | 需要权重，当前不采集；实现链完整 | `READY` |

统计：`READY=8`、`CAPTURE_REQUIRED=1`、`NEEDS_HUMAN=4`，共 13 项。

## 没有单列的模型内胶水

- q/k/v 的 `split`、`reshape` 和 head-wise gate 的 `sigmoid * attention_output` 并入 `linear.bf16 + attention.radix`。
- residual add 并入 `norm.gemma_rms`。
- routed scaling、shared expert 相加并入 top-k、MoE 与 collective。
- draft 的 `cat(enorm, hnorm)` 并入 norm 与 `linear.bf16`。
- logits 的 last-token pruning/indexing 并入 `logits.lm_head`。

这些表达式没有独立 replacement seam，不为凑数量另造 operator_id。

## 缺口与采集计划

| operator_id | 当前结论 | 唯一 CUDA Session | 原因 |
|---|---|---|---|
| `activation.step_swiglu_with_limit` | 候选，是否为 P800 真实 gap 要由 baseline 决定 | 纳入，最多三种去重真实 shape | 无权重、可重放、已有 helper seam；用于证明完整流程 |
| `norm.gemma_rms` | 已确认 P800 落点缺失 | 不纳入 | 现行权限禁止保存 norm weight |
| `moe.topk_sigmoid_bias` | 已确认 P800 落点缺失 | 不纳入 | 现行权限禁止保存 router bias |
| `moe.bf16_clamped` | 已确认 Kunlun 忽略 limit | 不纳入 | 当前可替换边界包含 expert weights |
| `attention.radix` | 不是已确认 gap；实际 CUDA backend 仍待启动日志补证 | 不纳入 | 需要 KV 与 batch runtime state，超出离线 replay 边界 |

按当前 Contract，只要仍有 `NEEDS_HUMAN` 就不能开始 CUDA Capture Session。因此 Scan Run 可以封存完整清单，但 Working State 必须停在 `NEEDS_HUMAN / SCAN`，不能只采特殊 SwiGLU 后假装完整扫描已闭合。

## 特殊 SwiGLU 前置重构证据

- 实现：`python/sglang/srt/models/step3p5_ops.py:5-13`。
- 调用点：`python/sglang/srt/models/step3p5.py:95-104`。
- 三种 shape/dtype 对重构前表达式做 exact compare：`test/registered/unit/models/test_step3p5_ops.py:72-90`。
- HookRegistry replacement 以及 imported binding 传播 smoke：同测试 `:92-123`。
- CUDA 与 Kunlun 两个固定 revision 上均实跑 `2 passed`，两侧 helper patch 完全一致。

这个 helper 只建立采集和替换入口，不算 Operator Gap，也不消耗 repair attempt。

## 扫描结论

BF16 分支已经闭合了旧 INT8 扫描中的 checkpoint/quantization 不确定性，并确认 EAGLE 仍会进入 `Step3p5MTP`。当前完整清单发现三个有明确代码缺口、但因禁止保存权重或 bias 而无法按现行边界重放的算子；另有 attention 的 CUDA 默认 backend 需要实际启动日志补证。特殊 SwiGLU 是唯一可以直接进入现行采集边界的候选，但按既定停止规则，在上述 `NEEDS_HUMAN` 被人处理前不能消耗唯一 CUDA Session。

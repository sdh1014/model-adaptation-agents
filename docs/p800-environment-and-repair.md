# P800 环境问题与 SwiGLU-clamp 修复审查

本文把两类事实分开记录：

- P800 环境中实际遇到的问题；
- revision 5 的数值结果与正式代码审查结论。

旧 Run 保持不可变。`runs/repair-attempt-1-r5-001` 的三个 shape 数值结果仍可说明
`kunlun_ops.swiglu(limit=...)` 能消除当前差异，但其中新增生产 helper 的实现形态
未通过正式审查，不能作为后续方案的代码基线。

## 1. 当前 P800 Kunlun 启动环境策略

所有 P800 服务和 Kernel replay 进程必须先设置：

```bash
export SGLANG_PLATFORM=kunlun
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export PYTHONPATH="$SGLANG_KUNLUN_WORKTREE/python:$SGLANG_KUNLUN_WORKTREE/sglang-kunlun${PYTHONPATH:+:$PYTHONPATH}"
```

前两个变量是不可省略的 Kunlun 基线。`PYTHONPATH` 必须让固定 worktree 中的
SGLang 和 Kunlun plugin 排在已有环境之前。`SGLANG_KUNLUN_WORKTREE` 必须是
Contract 固定 revision 的 Git 根目录，不能用 site-packages 中另一个版本代替。

下面是 Agent 的完整候选清单，不是固定启动模板。Agent 必须结合节点类型、服务
拓扑、实际 backend 和当前活动算子按需选择，不能整表无条件导出。对每个实际设置
的可选变量，要在 P800 环境证据中记录最终值和选择原因。

执行 `replay_compare.py --mode kernel-replay` 时，每个已经导出的可选变量都必须
追加一次 `--p800-environment-reason '变量名=选择原因'`。工具从真实进程环境读取
变量值，只从参数读取原因，并把两者一起写入 replay Run；值和原因缺少任一项都会
在调用算子前停止。例如：

```bash
export MODEL_PATH=/models/step-3.7-flash
# replay_compare.py 的其他参数省略
--p800-environment-reason \
  'MODEL_PATH=加载 Contract 固定的 Step-3.7-Flash checkpoint'
```

| 环境变量 | 建议值 | 选择依据 |
|---|---|---|
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | D 节点：`32`；P 节点：`256` | 使用 DeepEP 时按节点类型选择；更大值需要更多通信缓冲。 |
| `XSHMEM_SYMMETRIC_SIZE` | D 节点：`2147483648`（2GB）；P 节点：`8589934592`（8GB） | 使用 BKCL/RDMA 时按节点类型和通信量选择。 |
| `XSHMEM_QP_NUM_PER_RANK` | `32` | 需要多流 RDMA 并发时设置。 |
| `ENABLE_CONTROL_THINK` | `false` | 需要关闭 Thinking 控制逻辑时设置。 |
| `DEFAULT_ENABLE_THINKING` | `false` | 需要默认关闭 Thinking 时设置。 |
| `XPU_HYBRID_ATTN_USE_GATHER_MULTISTREAM` | `0` | 关闭 Hybrid Attention Gather 多流以规避稳定性问题。 |
| `SGLANG_HEALTH_CHECK_TIMEOUT` | `120` | 模型启动较慢时提高健康检查超时。 |
| `MODEL_PATH` | 实际模型目录 | 启动模板通过环境变量读取权重目录时设置。 |
| `SGLANG_HACK_FLASHMLA_BACKEND` | `kunlun` | 实际路径使用 FlashMLA 且需要强制 Kunlun backend 时设置。 |
| `SGLANG_OPT_USE_TILELANG_MHC_PRE` | `0` | 关闭 TileLang MHC 前处理。 |
| `SGLANG_OPT_DEEPGEMM_HC_PRENORM` | `0` | 关闭 DeepGEMM HeadCache PreNorm。 |
| `SGLANG_OPT_USE_TILELANG_MHC_POST` | `0` | 关闭 TileLang MHC 后处理。 |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` | `0` | 禁用计算通信重叠。 |
| `USE_FAST_ALLOC_EXTEND_KUNLUN` | `False` | 关闭 Kunlun Fast Alloc Extend。 |
| `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` | `False` | 禁用 Torch FP8 Paged MQA Logits。 |
| `SGLANG_OPT_USE_JIT_NORM` | `True` | 需要 JIT LayerNorm/RMSNorm Kernel 时启用。 |
| `SGLANG_OPT_USE_FUSED_STORE_CACHE` | `0` | 关闭 KV Cache Store Fusion。 |
| `SGLANG_OPT_FP8_WO_A_GEMM` | `False` | 不使用 FP8 Weight-Only GEMM 时设置。 |
| `SGLANG_OPT_BF16_FP32_GEMM_ALGO` | `torch` | BF16/FP32 GEMM 需要走 Torch 实现时设置。 |
| `SGLANG_TOPK_TRANSFORM_512_TORCH` | `False` | 禁用 Torch TopK-Transform。 |
| `SGLANG_FIX_DSV4_BASE_MODEL_LOAD` | `1` | 实际模型需要 DeepSeek-V4 Base 加载修复时启用。 |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` | `0` | 关闭 DeepGEMM JIT 预编译。 |
| `SGLANG_DSV4_FP4_EXPERTS` | `0` | 不使用 FP4 Experts 时设置。 |
| `SGLANG_PREP_IN_CUDA_GRAPH` | `False` | 不在 CUDA/XPU Graph 中执行预处理。 |
| `SGLANG_OPT_SWIGLU_CLAMP_FUSION` | `0` | 关闭 SwiGLU Clamp Fusion。 |
| `SGLANG_OPT_USE_FUSED_HASH_TOPK` | `0` | 关闭 Fused Hash TopK。 |
| `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK` | `0` | 关闭 JIT Fused TopK。 |
| `SGLANG_OPT_CP_REARRANGE_TRITON` | `False` | 关闭 Triton CP Rearrange。 |
| `SGLANG_ENABLE_THINKING` | `0` | 禁用 Thinking 推理模式。 |
| `SGLANG_TOOL_STRICT_LEVEL` | `2` | 需要严格 Tool Calling 校验时设置。 |
| `BKCL_RDMA_VERBS` | `1` | 明确使用 RDMA Verbs 而非 TCP 时设置。 |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | `360000` | 大规模 Disaggregation 初始化较慢时设置。 |

如果 DeepEP/BKCL 配置需要区分 D/P 节点而 Agent 无法从现场证据确认节点类型，
进入 `NEEDS_HUMAN` 询问节点类型，不能猜值。未选择的候选变量保持未设置；环境或
导入失败属于工具/环境失败，不能写成 Operator Gap。

## 2. 已遇到的环境问题

这些问题发生在算子执行之前，属于环境或导入链问题，不能记作 Operator Gap。

| 报错 | 原因 | 当时的处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'sglang_kunlun'` | 当前 Python 环境没有安装或暴露 Kunlun plugin | 对固定 worktree 做 editable install |
| `SGLANG_PLATFORM='kunlun' not found` | Kunlun 平台入口点没有在当前环境注册 | editable install 后设置 `SGLANG_PLATFORM=kunlun`、`SGLANG_USE_XPU=1` |
| `No module named 'sglang.srt.plugins'` | site-packages 中的 SGLang 抢先于固定 worktree 被导入 | replay worker 把 `<worktree>/python` 放到 `PYTHONPATH` 前部 |
| `No module named 'triton.tools.tensor_descriptor'` | 环境中的 Triton 与 flashinfer 导入要求不匹配 | 设置 `SGLANG_IS_FLASHINFER_AVAILABLE=false`，避开本轮不需要的 flashinfer |
| `Torch not compiled with XPU enabled` | Kunlun 平台没有正确激活时走到了错误设备路径 | 先解决 plugin 注册与环境变量，再执行 replay |

当时使用的环境准备方式是：

```bash
pip install -e /workspace/baidu/aicapx/sglang/sglang-kunlun \
  --no-deps --no-build-isolation

export SGLANG_PLATFORM=kunlun
export SGLANG_USE_XPU=1
export SGLANG_IS_FLASHINFER_AVAILABLE=false
```

这些命令只用于恢复正确运行环境，不是算子修复。

## 3. 已确认的数值差异

CUDA 的现有调用 `_swiglu_silu_clamp_mul(x, gemm1_limit)` 会：

1. 对 gate 半段执行 SiLU；
2. 把 gate 上界限制为 `gemm1_limit`；
3. 把 linear 半段限制在 `[-gemm1_limit, gemm1_limit]`；
4. 相乘得到输出。

固定 Kunlun revision 中的生产路径原先调用：

```python
kunlun_ops.swiglu(x=y, y=out1)
```

它没有把 `layer.moe_runner_config.gemm1_clamp_limit` 传给已有
`kunlun_ops.swiglu`。P800 baseline 的三个 Golden shape 中，一个通过、两个在固定
`torch.testing.assert_close(atol=0.01, rtol=0.02)` 下失败。

设备探针与旧 attempt 的数值 replay 表明，`kunlun_ops.swiglu` 已支持可选
`limit=<float>`；传入 limit 后三个 shape 均通过。这里保留的是数值结论，不接受旧
attempt 为回放而新增 helper 的做法。

## 4. 正式审查结论

`35aa72e` 相对 `03cb3a1` 的正式 Standards 与 Spec 审查保存在
`runs/code-review-35aa72e-001`。审查拒绝以下设计：

- 在 `unquant.py` 新增 `apply_gemm1_swiglu_clamp`；
- 用通用 `<module>:<callable>` 协议导入任意“修复函数”；
- 让 replay 验证新增 wrapper，而不是原有 `kunlun_ops.swiglu` Kernel Call。

原因很直接：这会改变原始算子边界，也为一个简单参数传递增加不必要的生产函数和
通用协议。

## 5. 修正后的内联补丁

新的建议补丁保存在
`runs/inline-repair-correction-r5-001/candidate.patch`，只修改原调用位置：

```diff
     d = y.shape[-1] // 2
     out1 = torch.empty(y.shape[:-1] + (d,), dtype=y.dtype, device=y.device)
-    kunlun_ops.swiglu(x=y, y=out1)
+    gemm1_limit = layer.moe_runner_config.gemm1_clamp_limit
+    if gemm1_limit is None:
+        kunlun_ops.swiglu(x=y, y=out1)
+    else:
+        kunlun_ops.swiglu(x=y, y=out1, limit=float(gemm1_limit))
```

这份补丁有三个边界：

- 不新增生产函数；
- 保持 `unquantized_fused_moe_apply_kunlun` 原有结构；
- 修复前后都调用已有 `kunlun_ops.swiglu`。

SOURCE 机器没有 P800 环境，因此该 Run 的状态是
`READY_FOR_P800_VALIDATION`，不是硬件 PASS。后续正式候选 replay 必须绑定
`candidate.patch`，但 `invocation_target` 仍然是 `kunlun_ops.swiglu`。

## 6. 对后续队列的影响

revision 6 不会把这一轮旧数据直接当作全队列 Golden。它会：

1. 重新绑定完整 gap queue；
2. 在一个新的 CUDA Session 中一次收齐计划修复项；
3. P800 按顺序逐项 baseline 和修复；
4. 后一项从前一项的累计通过 patch 开始；
5. 简单修复继续内联原调用位置；
6. 只有全部队列项通过才进入 `PASS / DONE`。

旧 revision 5 的环境、baseline 和数值证据继续保留为历史依据，不被改写。

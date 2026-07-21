# Run Evidence: Operator Candidate reanalysis

- Date: `2026-07-21`
- Contract revision: `9`
- Phase: `PREFLIGHT`（SOURCE 侧重分类，未执行 P800）
- Failure category: `none`
- Conclusion: `PASS_SOURCE_ONLY`
- SGLang revision: `49e384ce9d304648e9959666ecb8ce8cd98d0deb`
- SGLang-Kunlun revision: `546ad8c682392922792bbbfe53a8bf575545f118`
- Detailed analysis: `docs/step3p7-p800-operator-gap-analysis.md`

## Question resolved

历史 Operator Candidate 混用了上游 PyTorch helper、CUDA `sgl_kernel` symbol 和通用
Triton kernel，不能直接作为“Kunlun 缺失算子”清单。本轮改为按固定 Kunlun checkout
上的 Step-3.7 生产路径，分别记录：

1. 语义算子；
2. `implementation_kind`；
3. `route_state`；
4. P800 验证 `status`。

## Source findings

| candidate | implementation_kind | fixed-source route finding | P800 status |
|---|---|---|---|
| `step3p7.moe.swiglu_clamp` | `NATIVE_IMPLEMENTATION` | `_swiglu_silu_clamp_mul` 已表达精确语义；Kunlun MoE hook 调 `kunlun_ops.swiglu` 时没有装配 finite limit | `ROUTE_PENDING / PENDING` |
| `step3p7.norm.gemma_rmsnorm` | `NATIVE_IMPLEMENTATION` | `GemmaRMSNorm.forward_native` 已存在；Kunlun OOT dispatch 选择 `forward_cuda`，stub 未绑定 Gemma symbol | `ROUTE_PENDING / PENDING` |
| `step3p7.norm.gemma_fused_add_rmsnorm` | `NATIVE_IMPLEMENTATION` | `forward_native` 覆盖 residual 数学语义；低层 direct fused API 的原地/alias 契约仍需单独验证 | `ROUTE_PENDING / PENDING` |
| `step3p7.moe.topk_sigmoid` | `NATIVE_IMPLEMENTATION` | `fused_topk_torch_native` 已覆盖 sigmoid+bias+renorm；Step 非 grouped Kunlun 路径仍走未绑定 symbol | `ROUTE_PENDING / PENDING` |
| `step3p7.vision.prefill_attention` | `NATIVE_IMPLEMENTATION` | `VisionSdpaAttention` 已存在；默认 CUDA-like Kunlun 路由选择通用 Triton vision attention | `ROUTE_PENDING / PENDING` |

核心源码锚点及完整调用链见详细分析文档。静态重分类没有把任何条目标为 `PASS`，也
没有把任何条目升级为 Confirmed Operator Gap。

## Contract and workflow changes

- Contract revision 增加到 `9`，明确 native 实现不等于 Kunlun 路由已通过，也不等于
  算子缺失。
- Operator Verification Queue 保留全部五个历史 symbol，同时新增语义
  `candidate_id`、`implementation_kind` 和 `route_state`。
- `GemmaRMSNorm` 明确标为 `NATIVE_IMPLEMENTATION`；P800 测试调用模块生产入口，
  residual 模式同时检查返回结构、dtype、数值、alias 和必要的原地语义。
- visual attention 的初始 case 收紧到 Step-3.7 单图真实契约：`head_dim=96`、16/16
  MHA、non-causal、dense equal-length sequence；运行时观测到其他分支再追加。

## Execution

- Source audit: executed with `git show` and `git grep` against the two fixed commits.
- Operator CPU reference: not executed.
- P800 Production Operator: not executed.
- Eager model: not executed.
- Full-model CPU reference: prohibited by the current Contract.

## Local validation

- `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v`：7/7 通过；
- `PYTHONPYCACHEPREFIX=/tmp/model-adaptation-pycache python3 -m py_compile tests/test_run_driven_model_adaptation.py`：通过；
- `git diff --check`：通过；
- Skill frontmatter 的 `name` 和 `description` 本地解析：通过；
- 变更文件 trailing whitespace 检查：无命中。

旧 Run 使用的
`/Users/songdehao/.codex/skills/.system/skill-creator/scripts/quick_validate.py` 在当前机器
不存在，因此没有把该命令写成通过。当前行为测试已覆盖 Skill 必需的阶段、失败分类、
实现/路由字段和停止规则；校验工具路径缺失不是 P800 或算子失败。

## Next action

在 P800 固定 worktree 完成 Preflight。通过后按重分类队列逐项建立 CPU reference，
调用实际 Kunlun 生产入口验证路由和精度；已有实现但 dispatch 不可达时记录
`ADAPTATION`，只有生产调用已到达且无可用实现时才记录 `OPERATOR_MISSING`。

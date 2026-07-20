# Run-driven model adaptation

Status: Done

## Goal

把 `model-adaptation` 从 CUDA Golden capture/replay 流程改成由 Migration Agent
驱动的 P800 eager bring-up：

1. 先检查 P800 环境和固定源码；
2. 对历史 Operator Candidate 建立独立 CPU reference，并直接调用 P800 生产算子比较；
3. 运行真实 eager 模型，按失败类别修复并继续；
4. 模型能运行后验证精度；
5. 只有精度失败时才使用 SGLang dumper/comparator 逐层、多卡定位。

## Required behavior

- 删除自研 capture、replay、handoff bundle、workspace guard 的 Python 实现、
  插件注册、专用 runbook 和只验证这些实现的测试。
- 不保留兼容入口；Git 历史是旧实现的恢复来源。
- `$model-adaptation Step-3.7-Flash` 仍是唯一人类入口。
- 人只固定模型、源码、checkpoint、运行方式、修复范围和精度门槛。CPU reference、
  测试输入、P800 生产调用、比较和修复方案由 Agent 根据固定源码决定。
- 所有浮点结果使用 Contract 固定的 `atol`、`rtol`；结构、shape、约定输出 dtype
  和有限值必须满足，整数与布尔结果精确相等。Agent 不得放宽门槛。
- 局部测试通过后必须回到真实 P800 模型路径验证，不能以 test-only helper 代替
  生产调用。
- 失败至少分为 `ENVIRONMENT`、`ADAPTATION`、`OPERATOR_MISSING`、
  `OPERATOR_CONTRACT`、`DISTRIBUTED_RUNTIME`、`ACCURACY`。
- 当前五个历史 Operator Candidate 必须全部进入初始 Operator Verification Queue：
<!-- REQUIRED-OPERATOR-CANDIDATES: BEGIN -->
  - `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul`
  - `sgl_kernel.gemma_rmsnorm`
  - `sgl_kernel.gemma_fused_add_rmsnorm`
  - `sgl_kernel.topk_sigmoid`
  - `sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel`
<!-- REQUIRED-OPERATOR-CANDIDATES: END -->
- 算子测试必须覆盖各自语义风险：SwiGLU clamp 边界、RMSNorm 非连续输入、
  fused add 的两个原地结果、TopK 的 correction bias/renormalize/无并列输入、
  visual prefill attention 的 ragged sequence/causal/GQA 语义。
- 精度失败优先复用固定 SGLang 版本已有的 dumper、forward hook、comparator；
  不重新建设通用 Tensor replay 格式。

## Evidence

每次 Agent 动作在 `runs/<run-id>/result.md` 或 `result.json` 记录：

- 阶段和失败类别；
- 固定源码位置与 CPU reference 来源；
- 测试命令、输入 shape/dtype/layout；
- 实际生产调用位置；
- 使用的精度门槛和结果；
- 下一动作。

不要求默认保存输入输出 Tensor。

## Non-goals

- 本次 SOURCE 改写不宣称已经在 P800 执行五个算子或模型。
- 不做性能优化。
- 不新增 C++/自定义 kernel/底层注册；确实需要时进入 `BLOCKED`。

## Acceptance

- 活跃 Skill、Context、Spec 和模板只描述 run-driven 流程。
- 当前 Spec 的五个历史 Operator Candidate 全部为待验证项。
- 旧 replay Python 栈及其插件项目不存在。
- 活跃文档不再引用已删除脚本或旧状态阶段。
- 新行为测试和剩余全量测试通过。

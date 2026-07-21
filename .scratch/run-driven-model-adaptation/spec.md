# Run-driven Step-3.7 adaptation

Status: Done

## Goal

让 `$model-adaptation Step-3.7-Flash` 在 P800 上完成一个可持续执行的闭环：

1. 检查固定环境；
2. 验证全部已知算子的 P800 生产入口；
3. 启动真实 TP8 BF16 eager 模型；
4. 自动修复并归档运行中遇到的 BUG；
5. eager 服务正常回答即完成，同一 BUG 3 次不同修复仍失败才阻塞。

## Required behavior

- 唯一 phase 链是 `PREFLIGHT -> OPERATOR_VERIFICATION -> EAGER_BRINGUP -> DONE`。
- 不保留 capture/replay/handoff 工具，也不保留当前目标用不到的整模型 CPU 精度和逐层
  debugger 阶段。
- 算子浮点结果使用 Contract 固定的 `atol`、`rtol`；整数和布尔结果精确相等；同时
  检查结构、shape、声明 dtype、有限值和生产接口语义。
- 局部测试必须直接调用 P800 生产入口；通过后还要回到真实模型路径。
- 当前五个历史 Operator Candidate 必须全部进入初始队列：
<!-- REQUIRED-OPERATOR-CANDIDATES: BEGIN -->
  - `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul`
  - `sgl_kernel.gemma_rmsnorm`
  - `sgl_kernel.gemma_fused_add_rmsnorm`
  - `sgl_kernel.topk_sigmoid`
  - `sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel`
<!-- REQUIRED-OPERATOR-CANDIDATES: END -->
- 历史 symbol 只用于追溯。已有精确 PyTorch/`forward_native` 语义时标为
  `NATIVE_IMPLEMENTATION`，并把生产路由另列为 `ROUTE_PENDING`；不能直接标为算子
  缺失。
- 算子的完整 required cases 保存在 Migration Spec 队列和
  `docs/step3p7-p800-operator-gap-analysis.md`，主 Skill 不复制一套。
- 每个 BUG 使用稳定 `bug_id`。每次 attempt 只验证一个根因假设，修复后重跑聚焦复现
  和原始 P800 路径，并把根因与方案追加到 Bug Ledger。
- 同一 BUG 最多 3 次不同、可验证的修复尝试；3 次失败后 `BLOCKED`。
- 固定文本和单图请求均正常返回并复跑成功后设置 `model_status: PASS`、`status: PASS`、
  `phase: DONE`。

## Non-goals

- 本次 SOURCE 改写不宣称已执行 P800 算子或模型。
- 不验证整模型数值精度，不做性能优化。
- 不新增 C++、自定义 kernel 或底层注册。
- 不自动 SSH、管理凭证或执行 Git 发布动作。

## Acceptance

- 主 Skill 保持单一闭环，不复制算子细节或保留未来精度流程。
- Contract、模板、Context、Bug Loop 和测试使用同一状态机及 3 次上限。
- 初始五个 Candidate 全部保留且状态仍为待 P800 验证。
- eager 正常是 `PASS / DONE`，不是计划内 `BLOCKED`。
- 旧 replay Python 栈不存在，行为测试和全量测试通过。

# 扫描完整代码路径并形成算子缺口清单

Type: task
Status: resolved
Blocked by: 10

## What to build

由 Migration Agent 读取固定 checkpoint 的加载后配置和固定源码 revision，完整扫描 Step-3.7-Flash 实际经过的 target 与 EAGLE draft 模型路径。target 从 `Step3p7ForConditionalGeneration.forward` 进入，draft 从 `Step3p5MTP.forward` 进入；Worker、Runner、调度和 CUDA Graph 只作为入口证据，不进入算子枚举。

扫描的重点不是寻找特殊 SwiGLU，而是找出实际路径上全部 Semantic Operator，并逐项比较 CUDA 与 Kunlun 实现。每条记录至少包含稳定算子标识、target/draft 调用链、激活条件、输入输出边界、权重或状态依赖、CUDA 实现证据、Kunlun 实现证据、可重放性、可替换性，以及带理由的 `READY`、`CAPTURE_REQUIRED` 或 `NEEDS_HUMAN` 结论。

特殊 SwiGLU 作为首选验证候选处理：先完成经人工批准的薄 helper 纯重构，证明重构前后计算一致，并验证普通函数能够被 HookRegistry 替换；然后固定新的 SGLang revision 再形成最终扫描结果。helper 抽取本身不能被记录成 Operator Gap。

扫描判断由 Agent 基于代码、配置和运行证据完成，不增加 `scan.py`，也不靠纯 AST 或一组固定正则替代语义判断。

## Acceptance criteria

- Scan Run 同时覆盖 target 和 draft，且明确记录两条入口到各算子的真实调用链。
- 完整 operator 列表中的每一项都有 CUDA、Kunlun 两侧的代码锚点或明确的缺失证据。
- 分支是否实际激活由加载后配置或运行证据确认，不能仅凭源码中存在条件分支推断。
- 所有发现的缺口都进入 gap queue；所有 `CAPTURE_REQUIRED` 项都进入唯一 CUDA Session 的采集计划，不能静默跳过。
- 无法在禁止权重、内部 Tensor 和运行时对象的边界下重放的项被明确标为 `NEEDS_HUMAN`，并在 CUDA Session 开始前停止。
- 特殊 SwiGLU helper 有重构等价性测试和 HookRegistry replacement smoke；Contract 使用重构后的固定 revision。
- Scan Run 的 `result.json` 保存完整清单和代码证据，Spec Working State 只保存覆盖摘要、gap queue 和 Run 指针。
- 扫描结果足以在 Demo 验证后按“一个真实缺口一个票据”继续拆分，不需要重新扫描代码。

## Comments

特殊 SwiGLU 只是第一个验证样例，不是扫描范围。完整缺口清单是本票据的主要交付。

已按最终运行条件完成：

- Contract revision 1 固定 TP8、BF16、EAGLE、两端同一组启动参数、量化参数为空、attention backend 不显式指定，以及 `torch.testing.assert_close(atol=0.01, rtol=0.02)`。
- 源码确认“不额外传 MTP 开关”不会删除 draft：Step-3.7 的 EAGLE 自动启用 multi-layer EAGLE，并把 draft 架构改写为 `Step3p5MTP`，所以扫描仍同时覆盖 target 和 draft。
- CUDA SGLang revision 为 `6274831d9fef7bba04eb59302caac24563a974c9`，SGLang-Kunlun revision 为 `4731f8051b7d0bf2f03cf88e237e7e5fba80a5a9`；特殊 SwiGLU helper 补丁 SHA-256 在两侧一致，三组等价性与 HookRegistry smoke 均实跑 `2 tests, OK`。
- 完整 BF16 源码记录见 `work/research/step3p7-tp8-bf16-operator-scan.md`；正式不可变清单见 `runs/scan-001/result.json`，共 13 项：`READY=8`、`CAPTURE_REQUIRED=1`、`NEEDS_HUMAN=4`。
- 唯一 CUDA Session 的当前采集计划只包含 `activation.step_swiglu_with_limit`，最多三种去重真实 shape。三个有明确代码证据的其他 P800 缺口因现行边界禁止保存 norm weight、router bias 或 expert weights，不能静默进入采集；CUDA 默认 attention backend 则必须由实机启动日志补证。
- `migration-spec.md` 已绑定 `runs/spec-binding-001` 与 `runs/scan-001`，Working State 如实停在 `NEEDS_HUMAN / SCAN`，没有消耗 CUDA Capture Session。

Ticket 11 的完整扫描与证据封存已经完成。后续是否允许特殊 SwiGLU Demo 先行、将三个不可重放缺口留待后续，是一次新的 Contract 决策，不属于本票据漏扫。

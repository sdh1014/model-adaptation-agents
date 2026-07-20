# Step-3.7-Flash 到 KLX P800 的 Spec 驱动算子迁移工具方案

## Destination

交付一份可直接指导最小 Demo 实现的中文方案设计文档：说明一个 Spec 驱动、单 Agent、少量确定性脚本组成的算子迁移工具，如何完成 Step-3.7-Flash 真实路径扫描、一次性 CUDA Golden 采集、人工跨机器交接，以及 P800 上单算子最多三种 shape 的自动修复与精度验收。

最终文档保存为 `outputs/step3p7-p800-migration-tool-design.md`。设计形成后，本地图
继续记录通过 tickets 实现最小 Demo 的范围变化。唯一实际采集 Session 已封存并
接受 Golden；Handoff Bundle 已在 CUDA 与 P800 两端通过 manifest 校验，当前为
`ACTIVE / P800_REPAIR`。P800 baseline 已确认三个 shape 中两个存在精度缺口且不计
修复轮数；唯一下一步是在 P800 的 Claude Code 中由 Migration Agent 自主开始
attempt 1。

## Notes

- 领域词汇以仓库根目录 `CONTEXT.md` 为准。
- 设计需要参考当前 SGLang、SGLang-Kunlun 与 Step-3.7/Step3p5 源码；所有源码结论必须带稳定路径和行号证据。
- 背景方案页：<https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/vMri-fRViV/G4ag4GvOr4/hro1tO5i3e6uj0>。
- 工具运行时只有 `migration-spec.md + runs/`，不设置 `outputs/`；这里的 `outputs/` 仅是当前 Codex 工作区交付最终方案文档的位置。
- `migration-spec.md` 分为人工只读的 Contract 与 Agent 可更新的 Working State。Agent 每次动作前读取 Spec，动作后立即更新。
- 一个 Migration Agent 负责扫描、判断、生成修复和推进状态；确定性脚本只负责 Golden 采集、replay/compare、交接包校验和工作区保护。
- 一个自然语言 Skill 是首次启动和恢复的唯一人类入口，不设置顶层编排 CLI；Skill 内只保留四个职责单一的确定性脚本。
- `migration-spec.md` 的 Contract 内含专门的 JSON 结构化区块。四个脚本统一通过必选 `--spec` 只读该区块中的固定约束，不解析 Markdown 或 Working State；动作参数仍显式传入，结果绑定 `spec_id + contract_revision + Contract Data SHA-256`。
- Demo 保留完整路径扫描，但只闭环一个真实缺口 kernel，最多覆盖三种实际 shape。
  Contract 不预选算子；`active_operator` 在扫描完成后从 gap queue 选择。
- CUDA 只进行一次 Capture Session，可为同一算子保存最多三个去重样本；输入和
  CUDA 输出 Tensor 必须保存，允许当前 kernel 直接使用的参数 Tensor，但不保存
  完整 checkpoint、module state 或无关参数。
- 工具生成并校验 Handoff Bundle，跨机器复制由人工完成。
- P800 自动修复上限由 Contract 固定为 `max_repair_attempts: 5`，每轮只验证一个假设；允许 Python、P800 可执行的 PyTorch 和已有 xspeedgate，实现需要新 C++/Kernel 注册时进入 BLOCKED。
- 精度门槛位于 Contract，Agent 不得放宽。用户已在 P800 实机验证 `torch.testing` 可用，比较器直接采用 `torch.testing.assert_close`，不再设计多后端适配层。
- 最初 Wayfinder 阶段只负责规划和方案文档；后续实现由明确的 implementation
  tickets 驱动，不把 Wayfinder 调研票据当作实现授权。
- 术语优先复用 `CONTEXT.md`；只添加驱动流程所必需的字段，不为失败原因另造错误码体系、子状态机或新领域概念。

## Decisions so far

<!-- 每关闭一个子票，在这里追加一行简述与链接；详细答案只保留在子票中。 -->

- [确认 Step-3.7 Flash 的真实扫描入口与语义算子边界](issues/01-confirm-step3p7-scan-boundary.md)：按实际配置分别扫描 Step3p7 target 与 Step3p5MTP draft 模型树，Worker/Runner 止步；Agent 做语义判断，薄脚本只抽取和校验证据。特殊 SwiGLU 的 replay 边界成立，但因缺配置证据和独立 replacement seam，当前为 `NEEDS_HUMAN`。
- [原型化单文件 Migration Spec 与状态迁移](issues/02-prototype-migration-spec.md)：一个文件用边界标记分开人工只读 Contract 与 Agent 可写 Working State；`status + phase` 驱动流程，`WAITING` 只表示人工交接，详细历史留在 Run，恢复执行只依赖关键摘要和证据指针。
- [定义一次性 CUDA Golden Run 与交接包契约](issues/03-define-golden-run-contract.md)：保存边界输入与 CUDA 输出、必要标量和布局元数据；一个算子按调用签名最多保留三个样本，并由新 CUDA 进程自回放。交接包只含 Spec、Golden Run 和逐文件 manifest，底层序列化与比较 API 暂不绑定。
- [决定特殊 SwiGLU 的替换入口与 Demo 资格](issues/08-decide-swiglu-seam-and-demo-eligibility.md)：把无权重表达式原样抽成普通函数作为采集和替换入口；只有实际配置命中且 P800 baseline 失败时才有 Demo 资格，否则改选其他已采集的真实缺口。
- [确定 P800 修复的补丁生命周期](issues/04-decide-patch-lifecycle.md)：每轮从同一干净基线生成完整补丁，失败证据留在 Run 后恢复基线；通过补丁作为唯一未提交修改留在 P800 工作区，工具不管理分支、提交或推送。
- [设计基于 torch.testing 的精度比较接口](issues/05-prototype-comparator-interface.md)：比较器从 Contract 读取固定容差，先检查结构、dtype 与有限值，再逐 Tensor 调用 `torch.testing.assert_close`；Run 只保存诊断结果和日志，不重复保存 P800 actual Tensor。
- [走查一个 SwiGLU 三 Shape 的端到端纸面 Demo](issues/06-walkthrough-swiglu-demo.md)：固定源码包含 helper seam；baseline 按全部样本判断真实 gap；Contract 最多五轮且通过轮计数；Working State 只保留 `last_run` 与 `passing_run` 两类修复指针。
- [决定单 Agent 入口与最小脚本表面](issues/09-decide-entrypoint-and-tools.md)：同一个自然语言 Skill 负责初始化和恢复；四个脚本分别处理 Golden 采集、replay/compare、交接校验和工作区保护，并通过 `--spec + --run-dir + result.json` 与 Agent 交接。Spec 的 Contract 内嵌脚本专用 JSON 区块，固定约束不可由命令行覆盖，动态动作参数不写入 Contract。
- [产出 Spec 驱动 P800 算子迁移工具方案设计文档](issues/07-write-solution-design.md)：最终中文方案已保存到 `outputs/step3p7-p800-migration-tool-design.md`，完整合成单 Agent、Spec、四个脚本、一次 CUDA Golden、人工交接、P800 五轮修复、特殊 SwiGLU Demo 和源码证据；未进入工具实现。
- [纠正 eager 运行模式与 Spec 绑定](issues/17-correct-eager-runtime-contract.md)：Contract revision 3 明确为 target-only eager，不启用投机解码、不加载 draft，并同时禁用 decode/prefill CUDA Graph；新结果必须绑定 revision 3，旧 Run 仅作历史证据。
- [按 target-only eager 范围重新封存算子扫描](issues/18-rescan-target-only-eager-operator-gaps.md)：`runs/scan-003` 重新确认 target 路径的 13 个算子与缺口，Spec 已进入 `ACTIVE / CUDA_CAPTURE`，下一步是 revision 3 的真实 N 卡 preflight。
- [恢复 Step3p5MLP 原始算子边界](issues/19-restore-step3p5-mlp-boundary.md)：Contract revision 4 回到原始 CUDA/Kunlun 提交，以 `Step3p5MLP.forward` 覆盖投影、特殊 SwiGLU 和下投影，不再要求 helper 重构。
- [实现 TP8 rank 0 MLP 采集与模型内重放](issues/20-tp8-rank-local-mlp-capture-replay.md)：每种 shape 只保存 rank 0 的 `x/output`；CUDA 与 P800 都在加载同 checkpoint 的 TP8 模型实例内重放，不新增模型算子函数。
- [以 kernel 调用为边界重扫并选择最小 Demo](issues/21-kernel-level-scan-and-demo-selection.md)：Contract revision 5 只固定 kernel-call 扫描范围、文本/单图输入和直接参数保存规则；`scan-005` 保留初次结果，`scan-006` 纠正视觉链并补回 clamp SwiGLU 后选择已有 `_swiglu_silu_clamp_mul`。
- [Gemma RMSNorm kernel adapter（已取消）](issues/22-gemma-rmsnorm-kernel-capture-replay-adapter.md)：保留原票据历史；`scan-006` 改选更小缺口后标记为 `wontfix`。
- [实现 SwiGLU clamp Kernel Call 采集与重放 adapter](issues/23-swiglu-clamp-kernel-capture-replay-adapter.md)：已完成现有 `_swiglu_silu_clamp_mul` Hook、rank 0 三 shape 样本、CUDA 原函数 self-replay 与 P800 `kunlun_ops.swiglu` baseline 入口；adapter-002 的 `cuda-preflight-r5-002` 已回传并通过，未消耗正式 Session。baseline adapter 只证明缺口；P800 Migration Agent 根据源码补齐 repair replay adapter 后再领取 attempt。
- [正式审查内联修复边界](issues/24-review-inline-repair-boundary.md)：拒绝生产 helper 和任意 callable replay；候选补丁必须修改原调用位置，并由真实 worktree diff 与调用参数证据共同约束。
- [单算子通过后继续缺口队列](issues/25-continue-through-operator-gap-queue.md)：revision 6 让后一项继承累计通过 patch，当前项通过并回归旧项后自动进入下一项，全部关闭才进入 `PASS / DONE`。
- [在同一次 Session 采集视觉 attention 缺口](issues/26-capture-vision-gap-in-one-session.md)：`scan-007` 把单图 `prefill_attention._fwd_kernel` 与四个文本缺口放入同一 capture plan；固定本地图像、一个 session config 和五个 rank-0 collector 的 SOURCE 实现已完成，CUDA preflight 待实机执行。
- [确认跨 CUDA/P800 的样本格式与精度比较](issues/12-portable-sample-and-precision-compare.md)：CUDA Torch `2.11.0+cu129` 写出的三个 BF16 样本已由 P800 修改版 Torch `2.5.1+cu118` 读回；固定比较器正常 PASS 并正确拒绝有意数值偏差，Run 不保存 P800 actual Tensor。
- [实现可验证的人工交接包](issues/13-verifiable-manual-handoff-bundle.md)：已完成 pre-replay 样本文件摘要记录、样本摘要与 replay 证据绑定、build/verify、完整允许文件 manifest、CUDA self-replay 结果重算和篡改/错误 Contract 测试；工具不解析 Working State，Agent 按 Skill 校验下一状态临时 Spec 后再原子推进。
- [实现可恢复的五轮修复闭环](issues/14-five-attempt-repair-loop.md)：已完成固定 Kunlun revision/干净工作区检查、baseline 零计数判定、连续且不可复用的单一假设 Run、replay 前完整 patch 固化、patch/replay 联合摘要、失败恢复、通过保留、未知修改保护和第五轮上限；每轮假设及最小文件由 Migration Agent 依据证据自主选择，baseline 文件列表不锁定后续 attempt；synthetic 只允许绑定测试 Spec，不冒充 P800 实机闭环。
- [一次性采集已选择的 kernel 调用](issues/15-one-time-cuda-capture-for-selected-kernel.md)：PID `114828` 是唯一实际采集 Session；三份 Golden Sample 和 Handoff Bundle 已在 CUDA/P800 两端通过校验，Ticket 15 已关闭。
- [使用扫描后选中的 kernel 验证端到端流程](issues/16-validate-flow-with-selected-kernel.md)：P800 baseline 已在固定 revision 干净工作区检查三个 Golden shape，一个通过、两个仅因固定精度门槛失败，已证实为 `Operator Gap`；当前 revision 32、`attempts_used: 0`。下一步由 P800 Claude Code 中的 Migration Agent 先补齐原始 Kernel Call repair replay adapter，再自主选择 attempt 1 与修复实现并完成有限修复循环。

## Not yet specified

- Claude Code 中 Migration Agent 会依据源码和 baseline 证据选择什么 attempt 1
  假设，以及能否在最多五轮内让全部三个 shape 通过。这些属于 Agent 执行结果，
  不在方案或工具中预设。

## Out of scope

- 由 Codex、人工 runbook 或确定性脚本预先指定 attempt 1 的补丁、实现文件或代码
  方案；实际修复由 P800 Claude Code 中的 Migration Agent 决定。
- 重新登录真实 CUDA/P800 环境做其他能力验证；`torch.testing` 可用性直接采用用户已完成的实机验证结论。
- 自动 SSH、远程执行、自动上传或凭证管理。
- 新增 C++/自定义 Kernel、底层算子注册或性能优化。
- 完成 Step-3.7-Flash 的全部算子、全部 shape、DecoderLayer、完整模型组网、E2E 回复或性能验证。
- 从单算子最多三种 shape 扩展到全部缺口和更多 shape；必须等最小闭环真实运行稳定后另行设计。
- 沿用或评审现有 `model-adaptation-agents` 的架构设计。

# Migration Spec

<!-- HUMAN-OWNED CONTRACT: BEGIN -->

## Contract

本区在首次批准前可以根据人类明确提供的值形成草稿。批准后，Migration Agent 必须完整读取但不得修改；人修改本区时必须递增 `contract_revision`，并在 Human Decisions 中记录原因。

### Contract Data

四个 Deterministic Tool 只解析下面两个标记之间的 JSON，不解析 Contract 的其他 Markdown，也不读取或修改 Working State。未决的必填值为 `null` 时，本 Contract 尚不可执行；`model_path.draft_entry`、`runtime.quantization`、`runtime.speculative_algorithm` 与 `runtime.attention_backend` 的 `null` 是本 Demo 有意固定的“命令不传或路径不启用”，不是占位。

脚本启动时只读取一次这个 JSON，并把规范化 JSON 的 SHA-256 与 `spec_id`、`contract_revision` 写入 `result.json`。规范化方式固定为：

```python
json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
```

<!-- CONTRACT-DATA: BEGIN -->
{
  "schema": "migration-spec/v0",
  "spec_id": "step3p7-flash-p800-demo",
  "contract_revision": 5,
  "model": "Step-3.7-Flash",
  "source": {
    "sglang_revision": "49e384ce9d304648e9959666ecb8ce8cd98d0deb",
    "sglang_kunlun_revision": "546ad8c682392922792bbbfe53a8bf575545f118"
  },
  "checkpoint": {
    "id": "stepfun-ai/Step-3.7-Flash@5f6244077ac62e04eec3f320501ff8c2b293373a",
    "config_digest": "8d740ba5819e574b7a606ea7aa6d7d381142ed5cc77cab96dde2892c10414042"
  },
  "model_path": {
    "target_entry": "Step3p7ForConditionalGeneration.forward",
    "draft_entry": null
  },
  "scan_scope": {
    "model_paths": [
      "target"
    ],
    "granularity": "kernel-call",
    "input_modes": [
      "text-only",
      "single-image"
    ]
  },
  "runtime": {
    "tensor_parallel_size": 8,
    "dtype": "bfloat16",
    "quantization": null,
    "speculative_algorithm": null,
    "cuda_graph_backend_decode": "disabled",
    "cuda_graph_backend_prefill": "disabled",
    "attention_backend": null
  },
  "sample_policy": {
    "capture_tp_rank": 0,
    "save_inputs": true,
    "save_expected_outputs": true,
    "save_direct_parameter_tensors": true,
    "parameter_scope": "selected-kernel-call-current-rank",
    "save_full_checkpoint": false,
    "save_module_state_dict": false
  },
  "limits": {
    "max_shapes_per_operator": 3,
    "max_repair_attempts": 5
  },
  "precision_gate": {
    "comparator": "torch.testing.assert_close",
    "atol": 0.01,
    "rtol": 0.02,
    "require_exact_structure": true,
    "require_same_dtype": true,
    "require_finite": true,
    "check_stride": false
  }
}
<!-- CONTRACT-DATA: END -->

运行期信息不进入 Contract Data，例如当前 Run、活动算子、Golden Run、bundle 路径和 worktree 路径。这些信息由 Working State 或当前动作参数保存。

### Identity

- `human_owner`: `songdehao`

`schema`、`spec_id` 与 `contract_revision` 的唯一机器可读值位于 Contract Data，不在 Markdown 正文维护副本。首次批准前必须填写 `human_owner` 和 Contract Data 中全部未决占位，把 `contract_revision` 设为 `1`，并记录初始 Human Decision。不得把有意固定为 `null` 的 draft、量化、投机解码或 attention 参数改成伪造的实现名。

### Goal

在 target-only eager 模式下扫描固定版本 Step-3.7-Flash 的文本与单图实际路径。
扫描边界使用源码中已有的 Kernel Call。每项都要比较 CUDA 与 Kunlun 的实现差异。

扫描完成后，从缺口队列选择最小 Demo。在 CUDA 机器上只采集一次，最多保存三种
真实 shape；交接包由人工复制到 P800。P800 baseline 必须先证明所选调用真实失败，
随后最多进行五轮修复。

### Fixed inputs

- 模型名由 Skill 调用参数写入 Contract Data；本 Demo 的值必须是 `Step-3.7-Flash`。
- SGLang 与 SGLang-Kunlun revision、checkpoint、配置摘要、target 入口、TP8、BF16、target-only eager 和扫描输入模式全部由 Contract Data 固定。CUDA 与 P800 使用同一组启动参数；不传量化或投机解码参数，不额外传 MTP 开关，也不显式传 attention backend。运行后解析出的两端实际 backend 必须分别进入 Scan/Capture 证据。
- eager 固定为 decode 与 prefill 的 CUDA Graph backend 都是 `disabled`。`draft_entry: null` 与 `speculative_algorithm: null` 表示不加载 draft 路径，不得把 eager 解释为 EAGLE。
- 扫描边界是源码中已经存在、可直接调用和替换的 Kernel Call。CUDA extension、Triton、SGLang JIT、第三方 kernel 和真实不兼容的 Torch 调用都在范围内；不得为打桩新增 helper、自定义算子函数或整层 wrapper。
- Contract 不保存或预选 `active_operator`。只有当前 Scan Run 完成 CUDA/Kunlun 证据与候选比较后，Working State 才能从 gap queue 写入一个活动 kernel 调用。
- 扫描同时覆盖 `text-only` 和固定最小 `single-image` 请求。输入模式是扫描范围，不表示唯一 CUDA Session 要采集所有候选；最小 Demo 只采集选中的活动算子。
- CUDA 只允许一个 Capture Session；TP8 中只保存 rank 0，每个算子最多保存三个去重后的真实 shape。Golden Sample 保存重放所需的输入、CUDA 期望输出和必要非 Tensor 参数；可以保存当前 kernel 调用直接使用的当前 rank 参数 Tensor，但不得保存完整 checkpoint、module `state_dict` 或无关参数。
- `max_repair_attempts` 固定为五；baseline 不计数，通过轮计数。
- Precision Gate 本 Demo 固定为 `atol=0.01`、`rtol=0.02`。

### Demo Closure

只有以下条件全部有证据，Working State 才能写为 `PASS / DONE`：

1. Contract 已由人批准，全部必填值已填写，所有工具结果都绑定同一 Contract Data。
2. target-only eager 实际路径扫描完成，每条结论都有源码或运行证据。
3. 所有发现的 Operator Gap 都进入 gap queue；活动算子是在扫描完成后按最小可重放/可修复边界从队列选择，而不是由 Contract 预设。
4. 活动 kernel 调用的 rank 0 输入、必要的直接参数 Tensor 和 CUDA 期望输出已保存，最多三个不同真实 shape；Golden Sample 不包含完整 checkpoint、module state 或无关 Tensor。
5. Golden Run 在 CUDA rank 0 上按该 kernel 的实际调用签名 self-replay 通过，bundle 在 CUDA 与 P800 两端校验通过。
6. 被选作 Demo 的算子在 P800 baseline 中至少有一个样本执行或精度失败。
7. 修复没有超出 Repair Boundary；baseline 不计数，修复不超过五轮。
8. 该算子的全部已保存样本都通过固定 Precision Gate。
9. `passing_run` 保存完整通过 patch，且它是 P800 工作区唯一未提交修改。
10. 所有失败 Run 可追溯，最终 `next_action` 为 `none`。

### Non-goals

- 不要求首个 Demo 关闭其他 Operator Gap、覆盖第四种以后 shape，或证明 TP8 的全部八个 rank 都通过；扩展到全部缺口属于后续工作。
- 不迁移 DecoderLayer、完成模型组网、服务拉起、回复质量或 E2E logits 对齐。
- 不做吞吐、延迟或大 batch 性能优化。
- 不自动跨机器复制、管理凭证或新增 C++、自定义 Kernel、底层算子注册。

### Precision Gate

- 比较器、`atol`、`rtol` 和结构要求从 Contract Data 读取，Agent 和命令行不得覆盖或放宽。
- 输出容器结构、Tensor 叶子路径、shape 和 dtype 必须一致；非 Tensor 叶子按原值相等。
- CUDA expected 与 P800 actual 的所有数字 Tensor 都必须是有限值。
- 对应 Tensor 保持原 dtype 复制到 CPU，然后显式调用 `torch.testing.assert_close`；浮点 Tensor 使用 Contract Data 的容差，整数和布尔 Tensor 必须精确相等。
- `check_stride` 为 `false`；输出 stride 不属于本 Demo 的通过门槛。
- 每个 Golden Sample 必须单独通过。P800 actual output 只在内存中比较，不保存 actual Tensor。

### Authority and environment boundary

- Migration Agent 可以读取源码与证据、更新 Working State、调用四个 Deterministic Tool，并在 Repair Boundary 内生成候选 patch。
- Migration Agent 不得修改已批准 Contract、修改 Precision Gate、自动跨机器复制、取得新凭证或扩大到完整模型。
- Deterministic Tool 只解析 Contract Data，并接收当前动作的显式参数；它们不得读取或修改 Working State。
- 工具负责生成和校验 Handoff Bundle，人工负责复制；P800 必须先校验 manifest 才能读取 Golden Tensor。
- P800 修复只允许活动 kernel 调用的 Python、P800 可执行的 PyTorch、已有 xspeedgate/kunlun_ops 能力和聚焦测试。
- 工具不创建或切换分支，不 commit、不 push，也不清理未知用户修改。

### State model

`phase` 表示当前工作段：`SCAN | CUDA_CAPTURE | HANDOFF | P800_REPAIR | DONE`。

`status` 表示当前 Agent 能否继续：

- `ACTIVE`：只执行 Working State 中唯一的 `next_action`。
- `WAITING`：只用于计划内的人工跨机器复制；Agent 报告动作后停止。
- `PASS`：Demo Closure 已满足；停止且不可恢复。
- `BLOCKED`：原因已明确，但继续会越过范围、权限、环境或次数上限。
- `NEEDS_HUMAN`：缺少人类决定，或无法可靠确定边界、证据或下一动作。

`PASS`、`BLOCKED`、`NEEDS_HUMAN` 终止当前 Agent 执行；`WAITING` 暂停当前执行。`BLOCKED` 或 `NEEDS_HUMAN` 只能在人更新 Contract、递增 revision 并记录解决决定后恢复。`WAITING` 在 P800 校验成功后可以直接恢复。

### Allowed transitions

| From | Guard | To | Working State 必须同步记录 |
|---|---|---|---|
| `ACTIVE / SCAN` | target-only eager 扫描完成；没有未决边界；至少有一个采集候选 | `ACTIVE / CUDA_CAPTURE` | Scan Run、覆盖计数、gap queue、唯一下一动作 |
| `ACTIVE / SCAN` | 实际 kernel 调用、两端等价路径或保存边界不能可靠确定 | `NEEDS_HUMAN / SCAN` | stop reason、证据、一个人类问题 |
| `ACTIVE / SCAN` | 固定范围内没有 Demo 候选 | `BLOCKED / SCAN` | 原因和 Scan Run |
| `ACTIVE / CUDA_CAPTURE` | 唯一 Session 完成；self-replay 和 CUDA 端 bundle 校验通过 | `WAITING / HANDOFF` | Golden Run、bundle、manifest、人工复制动作 |
| `ACTIVE / CUDA_CAPTURE` | Session 已消耗且无法形成有效 Golden | `BLOCKED / CUDA_CAPTURE` | 失败 Run；禁止重开 Session |
| `WAITING / HANDOFF` | bundle 在 P800 通过 manifest 校验 | `ACTIVE / P800_REPAIR` | P800 验证 Run、baseline replay 下一动作 |
| `ACTIVE / P800_REPAIR` | 活动候选 baseline 全部通过 | `BLOCKED / P800_REPAIR` | “首选静态候选不是实机 correctness gap”和 baseline Run；不得重访 CUDA |
| `ACTIVE / P800_REPAIR` | baseline 至少一个样本失败 | 保持 `ACTIVE / P800_REPAIR` | 活动算子、baseline Run、`attempts_used: 0`、单一假设 |
| `ACTIVE / P800_REPAIR` | 本轮失败且尚未达到第五轮 | 保持 `ACTIVE / P800_REPAIR` | 失败 Run、恢复基线、下一条单一假设 |
| `ACTIVE / P800_REPAIR` | 全部样本通过且 Demo Closure 完整 | `PASS / DONE` | `passing_run`、最终 patch、closure 证据 |
| `ACTIVE / P800_REPAIR` | 需要越过 Repair Boundary，或第五轮仍失败 | `BLOCKED / P800_REPAIR` | stop reason、尝试 Run、越界或耗尽证据 |
| 任意 `ACTIVE` | 下一动作需要无法由证据决定的人类选择 | `NEEDS_HUMAN / 当前 phase` | 唯一问题和恢复所需证据 |

### Recovery protocol

新 Agent 或上下文压缩后的 Agent 必须：

1. 完整读取 Contract 与 Working State，不把聊天记录当作状态。
2. 校验 Contract Data 全部必填值和 `contract_revision`。
3. 对比 `contract_revision` 与 `observed_contract_revision`；不一致时先按最新 Human Decision 对齐，不能执行外部动作。
4. 读取 Current、活动算子、最近 Run 和对应证据，不默认加载全部历史 Run。
5. 校验 `status / phase` 和 `next_action`；`ACTIVE` 时只执行一条下一动作。
6. 动作完成后先封存 Run，再更新 Working State。
7. Spec 自相矛盾、证据失效或下一动作不唯一时，进入 `NEEDS_HUMAN` 并停止。

### Human Decisions

| contract_revision | decision | reason |
|---|---|---|
| `1` | 批准 Step-3.7-Flash TP8 BF16 EAGLE Demo Contract | 用户确认两端同命令、不显式指定 backend、不额外启用 MTP，并批准 atol=0.01、rtol=0.02；源码 revision 与 checkpoint config 由固定证据补齐 |
| `2` | 本轮 Demo 只推进 `activation.step_swiglu_with_limit`；`norm.gemma_rms`、`moe.topk_sigmoid_bias`、`moe.bf16_clamped` 保留在 gap queue 作为后续工作 | 用户批准特殊 SwiGLU 先完成 N 卡 Golden 采集验证；唯一 CUDA Session 启动时同时记录实际 CUDA attention 与 MoE backend，其他缺口不阻塞本轮验证 |
| `3` | 纠正为 target-only eager：不启用投机解码，不加载 draft 路径，decode 与 prefill 都禁用 CUDA Graph | 用户澄清此前的 EAGLE 是术语误解；两端仍使用同一组启动参数，其他 TP8、BF16、样本、容差、权限和修复边界不变 |
| `4` | 直接以原始 `Step3p5MLP.forward` 为 Demo 算子，在 TP8 模型中只采集和重放 rank 0；不新增 helper | 用户要求保持模型原始算子边界；源码确认特殊 shared expert 的 `down_proj` 使用 `reduce_results=False`，归约发生在该边界之后，因此 rank 0 足以验证最小流程，完整八 rank 验证留待后续扩展 |
| `5` | Contract 不再预选 MLP；固定 kernel-call 扫描范围、文本/单图输入和直接参数保存规则，扫描完成后再选择活动算子 | 用户要求扫描 CUDA 已实现而 Kunlun 缺失的实际 kernel 调用，Triton 只是一种实现；允许保存当前调用直接使用的参数 Tensor，但禁止完整权重，并要求比较 topk、视觉 attention 与其他真实缺口后选择最小 Demo |

<!-- HUMAN-OWNED CONTRACT: END -->

<!-- AGENT-WRITABLE WORKING STATE: BEGIN -->

## Working State

Agent 每次动作前完整读取 Contract 与本区；每次动作结束后立即更新本区。不得把详细历史堆入本区，完整命令、日志、补丁和结果留在不可变 Run。

### Current

- `observed_contract_revision`: `5`
- `state_revision`: `32`
- `status`: `ACTIVE`
- `phase`: `P800_REPAIR`
- `execution_site`: `SOURCE`
- `active_operator`: `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul`
- `last_completed_action`: `agent_driven_repair_handoff_prepared`
- `last_run`: `runs/repair-loop-tool-002`
- `next_action`: `在 P800 的 Claude Code 中用项目 model-adaptation Skill 恢复：Agent 读取已验收 baseline、失败样本元数据和目标源码，自主确定原始 Kernel Call 重放方式，先补齐并封存 repair replay adapter（不计 attempt），再选择 attempt 1 的单一假设、最小允许文件和修复实现，执行三个 Golden shape 的固定精度比较及有限迭代，直到 PASS、BLOCKED 或 NEEDS_HUMAN`

规则：`ACTIVE` 时 `next_action` 必须恰好一条；`WAITING` 时必须是一条人工动作；`PASS`、`BLOCKED`、`NEEDS_HUMAN` 时必须为 `none`。

### Scan

- `scan_run`: `runs/scan-006`
- `target_coverage`: `COMPLETE`
- `draft_coverage`: `NOT_APPLICABLE`
- `operator_counts`: `{ready: 6, capture_required: 5, needs_human: 0}`

#### Gap queue

| operator_id | scan_verdict | golden | demo_role | repair | evidence |
|---|---|---|---|---|---|
| `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul` | `CAPTURE_REQUIRED` | `SEALED` | 首选最小 Demo | P800 baseline 的三个 shape 中两个精度失败，已证实为 `Operator Gap`；等待 P800 Agent 自主修复 | `runs/p800-baseline-review-r5-001/result.json` |
| `sgl_kernel.gemma_rmsnorm` | `CAPTURE_REQUIRED` | `NOT_PLANNED` | 对比候选 | 缺少 Gemma symbol，且样本需要保存一个直接参数 `weight` | `runs/scan-006/result.json` |
| `sgl_kernel.gemma_fused_add_rmsnorm` | `CAPTURE_REQUIRED` | `NOT_PLANNED` | 后续缺口 | 双 in-place 输出比首选边界复杂 | `runs/scan-006/result.json` |
| `sgl_kernel.topk_sigmoid` | `CAPTURE_REQUIRED` | `NOT_PLANNED` | 对比候选 | 权重与 ids 双输出、排序语义比首选复杂 | `runs/scan-006/result.json` |
| `sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel` | `CAPTURE_REQUIRED` | `NOT_PLANNED` | 图像对比候选 | 只在单图视觉路径激活，metadata 和布局更多 | `runs/scan-006/result.json` |

完整 Kernel Call 清单、两端证据和候选排序保存在 `scan-006`。首选
`_swiglu_silu_clamp_mul` 是因为它是 SGLang 源码中已有的 `torch.compile` 调用，
在固定文本路径的 MoE 第 43、44 层可达，只需 `x`、标量 `gemm1_limit` 和单个输出，
不需要保存权重，也没有边界内 TP 通信。固定 Kunlun 路径调用普通
`kunlun_ops.swiglu`，但没有读取 clamp limit。P800 baseline 已检查全部三个
Golden shape：一个通过、两个仅因固定精度比较失败，因此它已经从静态候选变成
由 P800 baseline 证实的 `Operator Gap`。`scan-004` 及更早 Run 只作历史证据，不再作为当前
修复输入。

### CUDA Capture

- `capture_session_id`: `cuda-formal-session-r5-001`
- `session_status`: `SEALED`
- `golden_run`: `runs/cuda-golden-r5-001`
- `captured_sample_counts`: `{_swiglu_silu_clamp_mul: 3}`

`session_status` 只用 `NOT_STARTED | ACTIVE | SEALED | FAILED`。一旦 Session 为 `SEALED` 或 `FAILED`，不得创建第二个 Session。

### Handoff

- `bundle_status`: `VERIFIED_ON_P800`
- `bundle_path`: `runs/handoff-build-r5-001/bundle`
- `manifest_path`: `runs/handoff-build-r5-001/bundle/manifest.json`
- `p800_verification_run`: `runs/handoff-verify-p800-r5-001`

`bundle_status` 只用 `NOT_BUILT | VALID | COPIED | VERIFIED_ON_P800`。

### P800 Repair

- `baseline_run`: `runs/p800-baseline-assessment-r5-001`
- `attempts_used`: `0`
- `active_hypothesis`: `null`
- `passing_run`: `null`

baseline replay 不算修复尝试。每轮修改源码前递增 `attempts_used`，每轮只有一个 `active_hypothesis`。失败 Run 封存后恢复同一固定基线；通过 patch 是 P800 工作区唯一未提交修改。

### Demo Closure Evidence

| item | status | evidence |
|---|---|---|
| Contract approved and tool bindings match | `PASS` | `runs/spec-binding-004/result.json` |
| target-only eager kernel scan complete | `PASS` | `runs/scan-006/result.json` |
| gap queue and Demo selection complete | `PASS` | `runs/scan-006/result.json` |
| selected Kernel Call capture/CUDA self-replay/P800 baseline adapter implemented and sample-bound | `PASS` | `runs/adapter-003/result.json` |
| repair replay adapter exercises the Agent-selected original Kernel Call boundary | `PENDING` | `null` |
| adapter-001 CUDA preflight passed without consuming formal Session | `PASS` | `runs/cuda-preflight-r5-001/result.json` |
| capture-time adapter-002 CUDA preflight passed without consuming formal Session | `PASS` | `runs/cuda-preflight-r5-002/result.json` |
| CUDA/P800 sample format and fixed comparator validated | `PASS` | `runs/p800-portability-r5-001/result.json` |
| verifiable manual Handoff Bundle tool implemented | `PASS` | `runs/handoff-tool-001/result.json` |
| recoverable five-attempt repair loop tool implemented | `PASS` | `runs/repair-loop-tool-002/result.json` |
| one CUDA Session and at most three samples per operator | `PASS` | `runs/cuda-formal-review-r5-002/result.json` |
| CUDA self-replay passed | `PASS` | `runs/cuda-golden-r5-001/self-replay/result.json` |
| bundle verified on CUDA and P800 | `PASS` | CUDA: `runs/handoff-verify-cuda-r5-001/result.json`; P800: `runs/handoff-verify-p800-r5-001/result.json` |
| selected operator failed P800 baseline | `PASS` | `runs/p800-baseline-review-r5-001/result.json` |
| repair stayed inside boundary and attempt limit | `PENDING` | `null` |
| all selected samples passed Precision Gate | `PENDING` | `null` |
| passing patch is the only workspace change | `PENDING` | `null` |
| failed Runs traceable and next_action none | `PENDING` | `null` |

### Stop reason

- `stop_reason`: `null`
- `human_question`: `null`
- `resume_requires_contract_revision`: `false`

### Decisions

只追加会影响恢复执行的决定；详细过程写入 Run。

| state_revision | decision | evidence |
|---|---|---|
| `1` | 对齐已批准的 Contract revision 1，准备运行 Spec 绑定自检 | `migration-spec.md#human-decisions` |
| `2` | Spec 绑定自检通过，进入 target/draft 扫描 | `runs/spec-binding-001/result.json` |
| `3` | target/draft 完整扫描已封存；因四项 NEEDS_HUMAN 按 Contract 停止 | `runs/scan-001/result.json` |
| `4` | 保留 scan-001，以 scan-002 补齐逐算子调用链、两侧代码锚点、加载后配置证据和默认 MoE 分支解析 | `runs/scan-002/result.json` |
| `5` | 对齐已批准的 Contract revision 2；只推进特殊 SwiGLU，其他三个确认缺口继续保留 | `migration-spec.md#human-decisions`、`runs/scan-002/result.json` |
| `6` | 对齐 Contract revision 3，撤销旧 EAGLE preflight，并回到 SCAN 重新绑定 target-only eager 范围 | `migration-spec.md#human-decisions`、`.scratch/step3p7-p800-migration-tool/issues/17-correct-eager-runtime-contract.md` |
| `7` | revision 3 Spec 绑定自检通过，进入 target-only eager 扫描 | `runs/spec-binding-002/result.json` |
| `8` | `scan-003` 封存 13 个 target Semantic Operator；进入 revision 3 CUDA preflight | `runs/scan-003/result.json` |
| `9` | 对齐 Contract revision 4，恢复原始 `Step3p5MLP.forward` 边界并把 Demo 精度范围收紧到 TP8 rank 0 | `migration-spec.md#human-decisions`、`.scratch/step3p7-p800-migration-tool/issues/19-restore-step3p5-mlp-boundary.md` |
| `10` | revision 4 Spec 绑定自检通过 | `runs/spec-binding-003/result.json` |
| `11` | `scan-004` 以原始 MLP 边界替代 helper，保留完整 target-only eager 算子清单；进入 rank 0 CUDA preflight | `runs/scan-004/result.json` |
| `12` | 对齐 Contract revision 5，取消预选 MLP，回到 kernel-call 扫描范围 | `migration-spec.md#human-decisions`、`.scratch/step3p7-p800-migration-tool/issues/21-kernel-level-scan-and-demo-selection.md` |
| `13` | revision 5 Spec 绑定自检通过 | `runs/spec-binding-004/result.json` |
| `14` | `scan-005` 初稿比较文本与单图 Kernel Call，暂选 `sgl_kernel.gemma_rmsnorm` | `runs/scan-005/result.json` |
| `15` | 保留 `scan-005`，以 `scan-006` 纠正视觉调用链并补回已有 `_swiglu_silu_clamp_mul` 与 Kunlun 普通 SwiGLU 间的 clamp 语义缺口；因其无需权重且只有一个输出，改为最小 Demo | `runs/scan-006/result.json` |
| `16` | Ticket 23 完成 revision 5 Kernel Call adapter；本机只封存 CPU 测试和接口证据，CUDA preflight 由人按 runbook 执行并经 GitHub 回传 | `runs/adapter-001/result.json` |
| `17` | revision 5 CUDA preflight 经 GitHub 回传并通过简单结构校验：rank 0 保存三个 shape、rank 1 不落盘、CUDA 新进程 self-replay 通过，且未消耗正式 Session；下一步只在 P800 验证样本读回和固定比较器 | `runs/cuda-preflight-r5-001/result.json` |
| `18` | 三份 CUDA preflight 样本已由 P800 修改版 Torch 读回；固定比较器正常 PASS 并能拒绝有意数值偏差，且没有保存 P800 actual Tensor；Ticket 12 关闭，下一步实现 Ticket 13 交接包工具 | `runs/p800-portability-r5-001/result.json` |
| `19` | Ticket 13 完成 self-replay 前样本摘要记录、交接包 build/verify、Golden 自重放证据校验和逐文件完整性校验；下一步先实现 Ticket 14 的可恢复修复闭环，正式 CUDA Session 仍留给 Ticket 15 | `runs/handoff-tool-001/result.json` |
| `20` | code review 补齐样本字节与 replay config、worker result、Golden state、wrapper result 的同一摘要绑定，防止替换成另一份同 shape replay；`adapter-002` 只封存当前源码与本机元数据测试，旧 `cuda-preflight-r5-001` 只证明 adapter-001，当前源码 preflight 保持 PENDING 并阻止 Ticket 15 正式 Session | `runs/adapter-002/result.json` |
| `21` | 当前 `adapter-002` 的 CUDA preflight 已经 GitHub 回传并通过结构校验：三个 rank-0 shape、样本摘要绑定和新进程 self-replay 均通过，且未消耗正式 Capture Session | `runs/cuda-preflight-r5-002/result.json` |
| `22` | Ticket 14 完成基线检查、baseline 判定、连续且不可复用的单一假设 Run、replay 前完整 patch 保存、patch/replay 联合摘要、失败恢复、通过保留和五轮上限；正式 Spec 只接受 P800 kernel-replay，下一步进入 Ticket 15 的唯一正式 CUDA Session | `runs/repair-loop-tool-001/result.json` |
| `23` | 在 SOURCE 准备 Ticket 15 正式 runbook，下一动作转到 CUDA；唯一模型 Session 必须先在固定 GitHub evidence 分支原子占位，再运行真实 Golden 采集、自回放和样本摘要，Agent 核验后才生成临时交接 Spec；当前仍未消耗 Session | `model-adaptation/references/formal-cuda-capture.md` |
| `24` | 正式 Session 占位后首次模型启动因 checkpoint 不可用而失败，失败 Run 明确记录 Session 已消耗；随后上传的成功采集使用另一个服务 PID，样本字节虽完整但来源违反一次性 Contract，因此不接受 Golden、不构建 bundle，并进入 `BLOCKED / CUDA_CAPTURE` | `runs/cuda-formal-review-r5-001/result.json` |
| `25` | 人明确澄清模型加载路径错误、未进入采集 Hook 且没有产生样本的启动失败不计入 Capture Session；因此 PID `114828` 是唯一实际采集 Session，三份样本与 sidecar 元数据一致且 CUDA self-replay 通过，接受 Golden 并进入 Handoff Bundle 准备 | `runs/cuda-formal-review-r5-002/result.json` |
| `26` | CUDA 端 Handoff Bundle build 与 verify 通过，bundle 中 Spec、Golden 和 manifest 完整，进入 `WAITING / HANDOFF`；下一动作仅为人工复制到 P800 后先校验 manifest | `runs/handoff-build-r5-001/result.json`、`runs/handoff-verify-cuda-r5-001/result.json` |
| `27` | P800 verification Run 与唯一 evidence 提交核验通过，13 个 bundle 文件匹配 CUDA manifest；未反序列化 Tensor。进入 `ACTIVE / P800_REPAIR`，下一动作只执行零次计数的 baseline replay | `runs/handoff-verify-p800-r5-001/result.json`、`runs/p800-handoff-review-r5-001/result.json` |
| `28` | baseline 审查发现旧 worker 会把依赖、设备或比较器异常误记为 Operator Gap；`adapter-003` 把可信 FAIL 收紧为现有 Kunlun 调用本身的执行失败或明确输出/精度失败，下一动作仍为 P800 baseline | `runs/adapter-003/result.json` |
| `29` | P800 固定 revision 干净工作区完成三个 Golden shape baseline：一个通过、两个只出现 `torch.testing.assert_close` 数值失败；接受它为真实 Operator Gap，baseline 不计修复轮数 | `runs/p800-baseline-review-r5-001/result.json` |
| `30` | 修复实现交由 P800 Migration Agent 自主决定：baseline 检查路径不再锁定 attempt 文件，每轮 Agent 依据单一假设声明最小允许文件，工具只负责轮次、补丁、replay 绑定和恢复；当前尚未领取 attempt 1 | `runs/repair-loop-tool-002/result.json` |
| `31` | 人工明确 baseline 验收后不预填 attempt 假设：`attempts_used: 0`、`active_hypothesis: null` 是合法交接，Agent 在领取 attempt 时写入单一假设 | `runs/repair-loop-tool-002/result.json` |
| `32` | 取消预设 `<module>:<function>` 修复接口：固定 Kunlun 源码的真实入口接收模型层与 dispatch 数据，SwiGLU 只是内部调用，不能假设存在统一 `(x, limit)` 函数。P800 Agent 根据源码自主决定修复与原始 Kernel Call 重放方式；baseline adapter 只证明缺口。Agent 在领取 attempt 前先补齐并封存 repair replay adapter，该准备动作不计 repair attempt | `runs/repair-loop-tool-002/result.json` |

<!-- AGENT-WRITABLE WORKING STATE: END -->

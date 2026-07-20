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
  "contract_revision": 6,
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

扫描完成后按最小可重放边界排列 gap queue。CUDA 机器只启动一个新的 revision 6
Capture Session，为队列中每个计划修复项最多保存三种真实 shape；交接包只人工
复制一次。P800 按队列逐项 baseline 和有限修复，一个算子关闭后自动进入下一个。

### Fixed inputs

- 模型名由 Skill 调用参数写入 Contract Data；本 Demo 的值必须是 `Step-3.7-Flash`。
- SGLang 与 SGLang-Kunlun revision、checkpoint、配置摘要、target 入口、TP8、BF16、target-only eager 和扫描输入模式全部由 Contract Data 固定。CUDA 与 P800 使用同一组启动参数；不传量化或投机解码参数，不额外传 MTP 开关，也不显式传 attention backend。运行后解析出的两端实际 backend 必须分别进入 Scan/Capture 证据。
- eager 固定为 decode 与 prefill 的 CUDA Graph backend 都是 `disabled`。`draft_entry: null` 与 `speculative_algorithm: null` 表示不加载 draft 路径，不得把 eager 解释为 EAGLE。
- 扫描边界是源码中已经存在、可直接调用和替换的 Kernel Call。CUDA extension、Triton、SGLang JIT、第三方 kernel 和真实不兼容的 Torch 调用都在范围内；不得为打桩新增 helper、自定义算子函数或整层 wrapper。
- Contract 不保存或预选 `active_operator`。只有当前 Scan Run 完成 CUDA/Kunlun 证据与候选排序后，Working State 才能从 gap queue 写入一个活动 kernel 调用。
- 扫描同时覆盖 `text-only` 和固定最小 `single-image` 请求。输入模式同时决定唯一 CUDA Session 的请求集合；capture plan 必须覆盖本轮 gap queue 的全部计划修复项。
- revision 6 只允许一个新的 Capture Session；revision 5 的单算子 Session 保留为历史，不能充当 revision 6 Golden。TP8 中只保存 rank 0，每个算子最多保存三个去重后的真实 shape。Golden Sample 保存重放所需的输入、CUDA 期望输出和必要非 Tensor 参数；可以保存当前 kernel 调用直接使用的当前 rank 参数 Tensor，但不得保存完整 checkpoint、module `state_dict` 或无关参数。
- `max_repair_attempts` 固定为五；baseline 不计数，通过轮计数。
- Precision Gate 本 Demo 固定为 `atol=0.01`、`rtol=0.02`。

### Queue Closure

只有以下条件全部有证据，Working State 才能写为 `PASS / DONE`：

1. Contract 已由人批准，全部必填值已填写，所有工具结果都绑定同一 Contract Data。
2. target-only eager 实际路径扫描完成，每条结论都有源码或运行证据。
3. 所有发现的 Operator Gap 都进入 gap queue；活动算子是在扫描完成后按队列顺序选择，而不是由 Contract 预设。
4. 每个计划修复的 kernel 调用都保存 rank 0 输入、必要直接参数和 CUDA 期望输出，且每个算子最多三个不同真实 shape；Golden Sample 不包含完整 checkpoint、module state 或无关 Tensor。
5. 每个 Golden Run 都在 CUDA rank 0 上按其实际调用签名 self-replay 通过，包含全部 Golden 的 bundle 在 CUDA 与 P800 两端校验通过。
6. 每个 gap queue 行都有 baseline；baseline 已等价或经过不超过五轮的 Repair Boundary 内修复后，该行全部样本通过固定 Precision Gate。
7. 后一项修复以此前最近一份非空 `passing_run` 的累计 patch 为起点；baseline 直接通过的行不伪造新 patch，只沿用该指针。候选 replay 前后都证明 P800 工作区逐字节等于该累计 patch；本轮封存时同时校验此前所有通过项的 Golden 回归。任一失败只恢复到上一接受起点。
8. 每行的 `repair_status` 都为 `PASS`，没有 `PENDING` 或 `ACTIVE`。
9. 只要发生过修复，最后一份非空 `passing_run` 就保存完整累计通过 patch，且它是 P800 工作区唯一未提交修改；若全部算子都是 baseline 直接 PASS，则所有 `passing_run` 可以保持 `null`，固定 revision 的干净工作区就是合法终态。
10. 所有失败 Run 可追溯，最终 `next_action` 为 `none`。

### Non-goals

- 不覆盖第四种以后 shape，也不证明 TP8 的全部八个 rank 都通过。
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
- 候选 replay 不能仅凭存在 `candidate.patch` 改变调用参数。它必须同时绑定实际
  worktree diff，并由当前算子的 adapter 从被修改后的原生产调用点确认实际参数
  装配；历史 SwiGLU adapter 还要确认 `x/y` 沿用基线调用、`limit` 来自
  `layer.moe_runner_config.gemm1_clamp_limit`，并保留 `None` 时的原调用。
- 工具不创建或切换分支，不 commit、不 push，也不清理未知用户修改。

### State model

`phase` 表示当前工作段：`SCAN | CUDA_CAPTURE | HANDOFF | P800_REPAIR | DONE`。

`status` 表示当前 Agent 能否继续：

- `ACTIVE`：只执行 Working State 中唯一的 `next_action`。
- `WAITING`：只用于计划内的人工跨机器复制；Agent 报告动作后停止。
- `PASS`：全部 gap queue 已关闭；停止且不可恢复。
- `BLOCKED`：原因已明确，但继续会越过范围、权限、环境或次数上限。
- `NEEDS_HUMAN`：缺少人类决定，或无法可靠确定边界、证据或下一动作。

`PASS`、`BLOCKED`、`NEEDS_HUMAN` 终止当前 Agent 执行；`WAITING` 暂停当前执行。`BLOCKED` 或 `NEEDS_HUMAN` 只能在人更新 Contract、递增 revision 并记录解决决定后恢复。`WAITING` 在 P800 校验成功后可以直接恢复。

### Allowed transitions

| From | Guard | To | Working State 必须同步记录 |
|---|---|---|---|
| `ACTIVE / SCAN` | target-only eager 扫描完成；没有未决边界；至少有一个采集候选 | `ACTIVE / CUDA_CAPTURE` | Scan Run、覆盖计数、gap queue、唯一下一动作 |
| `ACTIVE / SCAN` | 实际 kernel 调用、两端等价路径或保存边界不能可靠确定 | `NEEDS_HUMAN / SCAN` | stop reason、证据、一个人类问题 |
| `ACTIVE / SCAN` | 固定范围内没有 Demo 候选 | `BLOCKED / SCAN` | 原因和 Scan Run |
| `ACTIVE / CUDA_CAPTURE` | 唯一 Session 收齐计划内全部算子；逐算子 self-replay 和 CUDA 端 bundle 校验通过 | `WAITING / HANDOFF` | 每行 Golden Run、bundle、manifest、人工复制动作 |
| `ACTIVE / CUDA_CAPTURE` | Session 已消耗且无法形成有效 Golden | `BLOCKED / CUDA_CAPTURE` | 失败 Run；禁止重开 Session |
| `WAITING / HANDOFF` | bundle 在 P800 通过 manifest 校验 | `ACTIVE / P800_REPAIR` | P800 验证 Run、baseline replay 下一动作 |
| `ACTIVE / P800_REPAIR` | 活动候选 baseline 全部通过 | 保持 `ACTIVE / P800_REPAIR` 或 `PASS / DONE` | 当前行 `PASS`、baseline Run；自动选择下一行，若无下一行则完成 |
| `ACTIVE / P800_REPAIR` | baseline 至少一个样本失败 | 保持 `ACTIVE / P800_REPAIR` | 活动算子、baseline Run、`attempts_used: 0`、单一假设 |
| `ACTIVE / P800_REPAIR` | 本轮失败且尚未达到第五轮 | 保持 `ACTIVE / P800_REPAIR` | 失败 Run、恢复基线、下一条单一假设 |
| `ACTIVE / P800_REPAIR` | 当前行全部样本及先前通过项回归均通过，且仍有下一行 | 保持 `ACTIVE / P800_REPAIR` | 当前行 `PASS`、`passing_run`、下一 `active_operator`、baseline 下一动作 |
| `ACTIVE / P800_REPAIR` | 当前行通过且全部 gap queue 已关闭 | `PASS / DONE` | 每行 passing evidence、最终累计 patch、closure 证据 |
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
| `6` | 从单算子 Demo 扩展为完整 gap queue：一个 revision 6 CUDA Session 一次收齐计划修复项，P800 单算子通过后自动进入下一项；简单修复必须内联原调用位置 | 用户要求审查 `35aa72e`，拒绝为 replay 新增生产 helper，并要求整个 Skill 连续修复队列；旧 revision 5 数值与环境证据保留为历史，但不能冒充 revision 6 Golden |

<!-- HUMAN-OWNED CONTRACT: END -->

<!-- AGENT-WRITABLE WORKING STATE: BEGIN -->

## Working State

Agent 每次动作前完整读取 Contract 与本区；每次动作结束后立即更新本区。不得把详细历史堆入本区，完整命令、日志、补丁和结果留在不可变 Run。

### Current

- `observed_contract_revision`: `6`
- `state_revision`: `51`
- `status`: `ACTIVE`
- `phase`: `CUDA_CAPTURE`
- `execution_site`: `CUDA`
- `active_operator`: `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul`
- `last_completed_action`: `revision_6_cuda_capture_failed_before_sample`
- `last_run`: `runs/cuda-formal-session-r6-001`
- `next_action`: `修正模型加载或服务环境后，只在同一个 reservation、SESSION_RUN 和 evidence 分支继续正式 CUDA Capture；下一次启动必须使用新 state revision 对应的 launch 证据目录`

规则：`ACTIVE` 时 `next_action` 必须恰好一条；`WAITING` 时必须是一条人工动作；`PASS`、`BLOCKED`、`NEEDS_HUMAN` 时必须为 `none`。

### Scan

- `scan_run`: `runs/scan-007`
- `target_coverage`: `PASS`
- `draft_coverage`: `NOT_APPLICABLE`
- `operator_counts`: `{ready: 6, capture_required: 5, needs_human: 0}`

#### Gap queue

| operator_id | scan_verdict | golden_run | repair_status | attempts_used | passing_run | evidence |
|---|---|---|---|---:|---|---|
| `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul` | `CAPTURE_REQUIRED` | `null` | `ACTIVE` | 0 | `null` | `runs/scan-007/result.json` |
| `sgl_kernel.gemma_rmsnorm` | `CAPTURE_REQUIRED` | `null` | `PENDING` | 0 | `null` | `runs/scan-007/result.json` |
| `sgl_kernel.gemma_fused_add_rmsnorm` | `CAPTURE_REQUIRED` | `null` | `PENDING` | 0 | `null` | `runs/scan-007/result.json` |
| `sgl_kernel.topk_sigmoid` | `CAPTURE_REQUIRED` | `null` | `PENDING` | 0 | `null` | `runs/scan-007/result.json` |
| `sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel` | `CAPTURE_REQUIRED` | `null` | `PENDING` | 0 | `null` | `runs/scan-007/result.json` |

`scan-006` 和 revision 5 的 gap queue 只作 `scan-007` 的历史源码线索，没有被
改写。`scan-007` 同时固定文本和单图请求；五项全部进入同一个 capture plan。
`repair_status` 只用
`PENDING | ACTIVE | PASS`；最多一行 `ACTIVE`，且必须等于 `active_operator`。

### CUDA Capture

- `capture_session_id`: `runs/cuda-formal-session-r6-001`
- `session_status`: `ACTIVE`
- `golden_runs`: `{}`
- `captured_sample_counts`: `{}`

`golden_runs` 只保存 `operator_id -> SEALED Golden Run`。`session_status` 只用
`NOT_STARTED | ACTIVE | SEALED | FAILED`。一旦 revision 6
Session 为 `SEALED` 或 `FAILED`，不得创建第二个 revision 6 Session。旧
`cuda-formal-session-r5-001` 保留为历史，不计作 revision 6 Session。

### Handoff

- `bundle_status`: `NOT_BUILT`
- `bundle_path`: `null`
- `manifest_path`: `null`
- `p800_verification_run`: `null`

`bundle_status` 只用 `NOT_BUILT | VALID | COPIED | VERIFIED_ON_P800`。

### P800 Repair

- `baseline_run`: `null`
- `attempts_used`: `0`
- `active_hypothesis`: `null`
- `passing_run`: `null`

P800 启动服务、baseline replay 和 candidate replay 都必须使用：

```bash
export SGLANG_PLATFORM=kunlun
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export PYTHONPATH="$SGLANG_KUNLUN_WORKTREE/python:$SGLANG_KUNLUN_WORKTREE/sglang-kunlun${PYTHONPATH:+:$PYTHONPATH}"
```

完整候选环境变量保存在 `docs/p800-environment-and-repair.md`。Agent 根据 D/P
节点类型、DeepEP/BKCL 拓扑、实际 backend 和当前算子按需选择；不能整表导出。
每个额外设置的变量必须在 P800 环境 Run 中记录最终值和原因。需要节点专属值却
无法确认 D/P 类型时进入 `NEEDS_HUMAN`。环境、插件导入或设备发现失败不能记作
Operator Gap。Kernel replay 对每个实际导出的候选变量使用
`--p800-environment-reason '变量名=选择原因'`，由工具把进程中的真实值和原因
一起封入 replay Run；没有候选变量时不传。

这四项是当前活动行的便捷副本，真实逐算子进度以 gap queue 行为准。baseline
replay 不算修复尝试。失败 Run 恢复到上一项 `passing_run` 的累计 patch；通过
patch 成为下一项的 `--accepted-result`。若一行 baseline 直接 PASS，它的
`passing_run` 沿用此前最近一份非空值；没有历史 patch 时保持 `null`。

领取后续算子 attempt 时，要把此前所有 `PASS` 且有 Golden 的 operator id 封存在
attempt；活动 replay 通过后，每一项都必须在同一累计 patch 下形成 regression
replay。`finish-attempt` 只有收到完整列表且全部通过才保留新 patch；任一回归失败
都把本轮记为失败并恢复上一份接受 patch。工具还会从上一份通过 Run 继承已经封存
的历史列表，后续算子不能只保留最近一项而删掉更早的回归。

### Queue Closure Evidence

| item | status | evidence |
|---|---|---|
| Contract approved and tool bindings match | `PASS` | `runs/spec-binding-005/result.json` |
| revision 5 helper/callable implementation rejected and inline source correction prepared | `PASS` | `runs/code-review-35aa72e-001/result.json`; `runs/inline-repair-correction-r5-001/result.json` |
| cumulative patch, exact original-call argument binding, inherited regression gate and automatic queue continuation implemented at SOURCE | `PASS` | `runs/operator-queue-tool-002/result.json` |
| target-only eager scan complete | `PASS` | `runs/scan-007/result.json` |
| gap queue complete, including fixed single-image vision attention | `PASS` | `runs/scan-007/result.json` |
| all planned CUDA capture/self-replay adapters and one-config multi-collector path implemented at SOURCE | `PASS` | `runs/multimodal-capture-tool-001/result.json` |
| P800 Kunlun required launch environment and Agent-selected optional policy implemented at SOURCE | `PASS` | `runs/p800-launch-environment-tool-001/result.json` |
| gap-driven multi-Golden bundle build and verification implemented and reviewed at SOURCE | `PASS` | `runs/gap-driven-handoff-tool-001/result.json`; `runs/gap-driven-handoff-tool-002/result.json`; `runs/gap-driven-handoff-tool-003/result.json` |
| revision 6 CUDA collector integration validated without consuming the formal Session | `PASS` | `runs/cuda-preflight-r6-002/result.json` |
| revision 6 formal CUDA runbook generated and corrected after SOURCE review | `PASS` | `runs/formal-cuda-runbook-r6-002/result.json` |
| P800 replay adapter resolved from the original Kunlun call site for each active operator | `PENDING` | `null` |
| revision 6 CUDA environment preflight passed without reading Scan or operators | `PENDING` | `null` |
| one revision 6 CUDA Session and at most three samples per operator | `PENDING` | `null` |
| every planned operator has a SEALED Golden Run | `PENDING` | `null` |
| bundle verified on CUDA and P800 | `PENDING` | `null` |
| every gap queue row has baseline evidence | `PENDING` | `null` |
| every repair stayed inside boundary and attempt limit | `PENDING` | `null` |
| every gap queue row passed its Golden Samples | `PENDING` | `null` |
| final cumulative patch is the only workspace change | `PENDING` | `null` |
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
| `33` | repair replay adapter 已在 model-adaptation 流程代码中补齐并封存：`invocation_target` 采用 `repair-kernel-call/v1:<module>:<callable>`，Agent 自选入口且该 callable 必须位于固定 revision 的 SGLang-Kunlun worktree 内，worktree 外的 flow-code 捷径被拒绝；`replay_compare.py` 与 `workspace_guard.py` 可生成并接受经真实修复路径的 P800 replay。此准备不改 SGLang-Kunlun 源码、不计 repair attempt，下一动作为领取 attempt 1 | `runs/repair-replay-adapter-r5-001/result.json` |
| `34` | attempt 1 通过并封存：Agent 选定单一假设，在唯一允许文件 `unquant.py` 内把内部 SwiGLU 提取为生产可调用 `apply_gemm1_swiglu_clamp(x, gemm1_limit)`，向既有 `kunlun_ops.swiglu` 传入 `limit=layer.moe_runner_config.gemm1_clamp_limit` 补上缺失的 clamp。经已封存 repair replay adapter 导入固定 worktree 内该 callable 执行三个 Golden shape 固定精度比较，全部通过（0 失败）。通过 patch 为工作区唯一未提交修改，一轮即 PASS，进入 `PASS`，`next_action: none` | `runs/repair-attempt-1-r5-001/result.json`、`runs/repair-attempt-1-r5-001/replay/result.json` |
| `35` | 正式审查判定 state 34 的三个 shape 数值证据可保留，但新增生产 helper、任意 callable replay 协议和 `PASS / P800_REPAIR` 状态不符合 Kernel Call Contract；用户批准 revision 6，恢复原调用位置内联修复，并把目标扩展为一次采集、P800 连续关闭完整 gap queue | `runs/code-review-35aa72e-001/result.json`、`runs/inline-repair-correction-r5-001/result.json`、`.scratch/step3p7-p800-migration-tool/issues/24-review-inline-repair-boundary.md`、`.scratch/step3p7-p800-migration-tool/issues/25-continue-through-operator-gap-queue.md` |
| `36` | revision 6 Spec 绑定自检通过；旧 revision 5 Scan 与 Golden 只作历史线索，下一动作是生成绑定新 Contract 的完整 Kernel Call Scan Run 和全队列 capture plan | `runs/spec-binding-005/result.json` |
| `37` | 正式审查整改与队列续行流程已落地：候选 replay 只绑定 patch 并保持原 `kunlun_ops.swiglu` 调用；后续算子从上一项累计通过 patch 开始，失败只恢复到该起点，Skill 在当前项通过后自动选择下一项。SOURCE 静态测试通过，P800 修复和 revision 6 Scan/Capture 尚未执行 | `runs/operator-queue-tool-001/result.json` |
| `38` | 复审补齐两条封存门槛：候选 replay 前后逐字节核对真实 worktree，并从原生产调用点确认参数装配；后续 attempt 在领取时固定全部历史 operator，`finish-attempt` 只有活动 replay 和完整回归列表全部通过才保留累计 patch。revision 6 多 adapter、多 collector 和全量 bundle 仍为扫描后的待实现能力，旧 revision 5 runbook 禁止复用 | `runs/operator-queue-tool-001/result.json` |
| `39` | 后继 SOURCE Run 修正复审发现：SwiGLU 候选必须保持基线 `x/y`、从模型配置读取真实 clamp limit 并保留 `None` 分支；新的 accepted Run 继承完整历史回归列表，第三个及以后算子不能删掉更早项；全 baseline 直接 PASS 时允许以干净固定 revision 和空 `passing_run` 闭环。`operator-queue-tool-001` 保留为被替代的历史证据 | `runs/operator-queue-tool-002/result.json` |
| `40` | `scan-007` 绑定 revision 6 并把四个文本缺口与单图视觉 attention 全部放入同一 capture plan；固定 SGLang 图像和摘要。SOURCE 已实现一个 session config、五个现有 CUDA 调用 Hook、逐算子 rank-0 collector 与 CUDA self-replay 编排；除已有 SwiGLU 外不预设 P800 replay 入口，由 Agent 推进到各项时依据 Kunlun 原调用点补齐。未运行 Torch/CUDA，也未消耗正式 Session；下一动作只是在 CUDA 机器执行多算子 preflight | `runs/scan-007/result.json`、`runs/multimodal-capture-tool-001/result.json` |
| `41` | 固定 P800 Kunlun 启动基线：服务与 replay 必须设置 `SGLANG_PLATFORM=kunlun`、`SGLANG_IS_FLASHINFER_AVAILABLE=False`，并优先加载固定 worktree 的 `python` 与 `sglang-kunlun`；其余完整变量表由 Agent 按节点、拓扑、backend 和活动算子选择并记录理由，不得无条件全量导出。该变化不修改 Contract Data，也不使既有 CUDA Scan/Capture 绑定失效 | `runs/p800-launch-environment-tool-001/result.json` |
| `42` | Handoff Manifest v2 改为从不可变 Scan Run 读取完整 `CAPTURE_REQUIRED` 队列，不在 bundle 工具中固定算子名或数量；每项必须恰好匹配一个 SEALED Golden 和 record-samples Run，包内同时保存原 Scan 结果。正式 Session 后由同一 CUDA Agent 连续完成样本审查、临时 WAITING Spec、build 与 verify，只在全部通过后原子更新 Spec 并一次性回传；当前下一动作仍是 CUDA preflight | `runs/gap-driven-handoff-tool-001/result.json` |
| `43` | Ticket 28 正式复审整改完成：Manifest v2 除了从 Scan 得到完整缺口集合，还要求所有 Golden 绑定同一个 Session 配置和同一个采集进程，包内保存原 Session 配置并校验输出 Tensor 元数据；revision 6 禁止省略 Scan 回退到历史 v1。当前仍只有 SOURCE 证据，下一动作保持 CUDA preflight | `runs/gap-driven-handoff-tool-002/result.json` |
| `44` | 后续复审补齐“正式 Session”证据：显式拒绝 preflight 配置，Session 中的文本与单图请求必须逐项匹配 Scan Run，并要求 `formal-result.json` 封存共同进程、Session 配置和全部 Golden state/sidecar 摘要；该结果一同进入 Bundle。下一动作仍保持 CUDA preflight | `runs/gap-driven-handoff-tool-003/result.json` |
| `45` | 首次 revision 6 CUDA preflight（`runs/cuda-preflight-r6-001`）在算子 `sgl_kernel.gemma_rmsnorm` 处 `PREFLIGHT_FAILED`：worker 用被调用对象的定义模块解析固定源，而 `sglang.srt.layers.layernorm` 对该 kernel 是 `from sgl_kernel import …` 的 re-export，`__module__` 落到 `sgl_kernel/elementwise.py`，与固定 seam 源 layernorm.py 不一致。按用户决定改用 SGLang 侧调用它的 layernorm.py：把 `preflight.py` 多 collector 分支的固定源解析改为取 `hook_target` seam 模块的源文件（不改 Contract、不新增 wrapper）。保留失败 Run 为历史，重跑 `runs/cuda-preflight-r6-002` 五算子各三 shape、rank 过滤与逐算子 CUDA self-replay 全部 PASS，未消耗正式 Session；下一动作为唯一 revision 6 正式 Capture Session | `runs/cuda-preflight-r6-002/result.json`、`runs/cuda-preflight-r6-001/result.json`、`.scratch/step3p7-p800-migration-tool/issues/29-preflight-seam-module-source-fixation.md` |
| `46` | 用户纠正 preflight 职责：preflight 只验证环境可用性，不应依赖具体算子。`runs/cuda-preflight-r6-001/002` 保留并重新归类为历史采集集成验证；前者暴露 seam 源解析工具问题，后者证明五个 collector、shape 去重、rank 过滤与 CUDA self-replay 集成通过，均不代表 Operator Gap。当前 `preflight-session` 改为不接收 Scan/operator，只检查 CUDA Torch、TP8 设备数、BF16、固定 SGLang worktree、插件入口和 P800 环境变量污染。正式 runbook 在 Session 占位前运行该环境检查，通过后由同一 CUDA Agent 继续正式采集，不增加中途回传 | `runs/preflight-responsibility-correction-001/result.json`、`runs/cuda-preflight-r6-002/result.json`、`runs/cuda-preflight-r6-001/result.json` |
| `47` | 正式代码与 Spec 复审收口：公开 `--mode preflight` 只运行独立的环境检查模块，不接收 Scan/operator；旧算子 Hook、shape、rank 过滤和 self-replay 入口改名为 `--mode capture-adapter-validation`，且只允许历史 Contract revision 1 至 5，revision 6 会在创建 Run 前拒绝。当前还没有 revision 6 正式 runbook，因此唯一下一步先在 SOURCE 生成并审查它，再交给 CUDA 机器执行 | `runs/preflight-responsibility-review-001/result.json` |
| `48` | revision 6 正式 CUDA runbook 已在 SOURCE 生成并审查：先执行与算子无关的环境 preflight，再从 `scan-007` 动态准备唯一 Session；远端 reservation 成功后，同一 CUDA Agent 启动一次 TP8/BF16/eager 模型，发送固定文本与单图请求，逐项记录并 self-replay 全部 Golden，最后动态构建和验证 Handoff Bundle。SOURCE 未执行 preflight、模型 Session 或 Bundle | `runs/formal-cuda-runbook-r6-001/result.json` |
| `49` | SOURCE 双轴审查整改完成：CUDA self-replay 从每个 Golden 的 capture config 读取五个 Scan 驱动 adapter；环境 preflight、reservation 和 Session 失败都会推进 Working State；每次模型启动使用独立 launch 证据目录；失败判定同时检查严格绑定当前 capture config 的完整 state 与 samples 目录全部文件，只有可信零样本 state 才归档并允许同一 reservation 重试，其余情况进入 BLOCKED；最终回传动态引用实际 ENV Run。当前动作仍在 SOURCE，未执行 CUDA preflight、模型 Session 或 Bundle | `runs/formal-cuda-runbook-r6-002/result.json` |

| `50` | revision 6 正式 CUDA Session 已在 GitHub evidence 分支占位；尚未产生 Golden Sample，只允许在同一个 reservation 内继续 | `runs/cuda-formal-session-r6-001/session-start.json` |

| `51` | revision 6 CUDA 启动失败但 state 与磁盘均无 Golden Sample；保留证据并只允许同一 reservation 内重试 | `runs/cuda-formal-session-r6-001/failure-51.json` |

<!-- AGENT-WRITABLE WORKING STATE: END -->

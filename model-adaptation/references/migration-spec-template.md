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
  "spec_id": null,
  "contract_revision": null,
  "model": null,
  "source": {
    "sglang_revision": null,
    "sglang_kunlun_revision": null
  },
  "checkpoint": {
    "id": null,
    "config_digest": null
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

- `human_owner`: `TBD`

`schema`、`spec_id` 与 `contract_revision` 的唯一机器可读值位于 Contract Data，不在 Markdown 正文维护副本。首次批准前必须填写 `human_owner` 和 Contract Data 中全部未决占位，把 `contract_revision` 设为 `1`，并记录初始 Human Decision。不得把有意固定为 `null` 的 draft、量化、投机解码或 attention 参数改成伪造的实现名。

### Goal

在 target-only eager 模式下，以现有 kernel/device-compute 调用为边界扫描固定版本模型的文本与单图实际路径，记录 CUDA 与 Kunlun 的 Operator Gap。扫描完成后从 gap queue 选择最小 Demo；在一次 CUDA Capture Session 中采集最多三种真实 shape，由人工把 Handoff Bundle 复制到 P800；随后在最多五轮修复内关闭一个 P800 baseline 真实失败的 kernel 调用。

### Fixed inputs

- 模型名由 Skill 调用参数写入 Contract Data；本 Demo 的值必须是 `Step-3.7-Flash`。
- SGLang 与 SGLang-Kunlun revision、checkpoint、配置摘要、target 入口、TP8、BF16、target-only eager 和扫描输入模式全部由 Contract Data 固定。CUDA 与 P800 使用同一组启动参数；不传量化或投机解码参数，不额外传 MTP 开关，也不显式传 attention backend。运行后解析出的两端实际 backend 必须分别进入 Scan/Capture 证据。
- eager 固定为 decode 与 prefill 的 CUDA Graph backend 都是 `disabled`。`draft_entry: null` 与 `speculative_algorithm: null` 表示不加载 draft 路径，不得把 eager 解释为 EAGLE。
- 扫描边界是源码中已经存在、可直接调用和替换的 kernel/device-compute 调用；不得为打桩新增 helper、自定义算子函数或整层 wrapper。
- Contract 不保存或预选 `active_operator`。只有完整 Scan Run 形成后，Working State 才能从 gap queue 写入一个活动 kernel 调用。
- CUDA 只允许一个 Capture Session；TP8 中只保存 rank 0，每个算子最多保存三个去重后的真实 shape。可以保存当前 kernel 调用直接使用的当前 rank 参数 Tensor，但不得保存完整 checkpoint、module `state_dict` 或无关参数。
- `max_repair_attempts` 固定为五；baseline 不计数，通过轮计数。
- Precision Gate 本 Demo 固定为 `atol=0.01`、`rtol=0.02`。

### Demo Closure

只有以下条件全部有证据，Working State 才能写为 `PASS / DONE`：

1. Contract 已由人批准，全部必填值已填写，所有工具结果都绑定同一 Contract Data。
2. target-only eager 实际路径扫描完成，每条结论都有源码或运行证据。
3. 所有发现的 Operator Gap 都进入 gap queue；活动算子是在扫描完成后从队列选择，而不是由 Contract 预设。
4. 活动 kernel 调用的 rank 0 输入、必要直接参数和 CUDA 期望输出已保存，最多三个不同真实 shape；Golden Sample 不包含完整 checkpoint、module state 或无关 Tensor。
5. Golden Run 在 CUDA rank 0 上按该 kernel 的实际调用签名 self-replay 通过，bundle 在 CUDA 与 P800 两端校验通过。
6. 被选作 Demo 的算子在 P800 baseline 中至少有一个样本执行或精度失败。
7. 修复没有超出 Repair Boundary；baseline 不计数，修复不超过五轮。
8. 该算子的全部已保存样本都通过固定 Precision Gate。
9. `passing_run` 保存完整通过 patch，且它是 P800 工作区唯一未提交修改。
10. 所有失败 Run 可追溯，最终 `next_action` 为 `none`。

### Non-goals

- 不要求关闭其他 Operator Gap、覆盖第四种以后 shape，或证明 TP8 的全部八个权重分片都通过。
- 不迁移 DecoderLayer、完成模型组网、服务拉起、回复质量或 E2E logits 对齐。
- 不做吞吐、延迟或大 batch 性能优化。
- 不自动跨机器复制、管理凭证或新增 C++、自定义 Kernel、底层算子注册。

### Precision Gate

- 比较器、`atol`、`rtol` 和结构要求从 Contract Data 读取，Agent 和命令行不得覆盖或放宽。
- 输出容器结构、Tensor 叶子路径、shape 和 dtype 必须一致；非 Tensor 叶子按原值相等。
- CUDA expected 与 P800 actual 的所有数字 Tensor 都必须是有限值。
- 对应 Tensor 保持原 dtype 复制到 CPU，然后显式调用 `torch.testing.assert_close`；浮点 Tensor 使用 Contract Data 容差，整数和布尔 Tensor 精确相等。
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
| `PENDING` | 初始 Contract 尚未批准 | 填写全部必填值，把 revision 设为 1，并记录批准理由 |

<!-- HUMAN-OWNED CONTRACT: END -->

<!-- AGENT-WRITABLE WORKING STATE: BEGIN -->

## Working State

Agent 每次动作前完整读取 Contract 与本区；每次动作结束后立即更新本区。不得把详细历史堆入本区，完整命令、日志、补丁和结果留在不可变 Run。

### Current

- `observed_contract_revision`: `null`
- `state_revision`: `0`
- `status`: `NEEDS_HUMAN`
- `phase`: `SCAN`
- `execution_site`: `SOURCE`
- `active_operator`: `null`
- `last_completed_action`: `spec_template_created`
- `last_run`: `null`
- `next_action`: `none`

规则：`ACTIVE` 时 `next_action` 必须恰好一条；`WAITING` 时必须是一条人工动作；`PASS`、`BLOCKED`、`NEEDS_HUMAN` 时必须为 `none`。

### Scan

- `scan_run`: `null`
- `target_coverage`: `PENDING`
- `draft_coverage`: `NOT_APPLICABLE`
- `operator_counts`: `{ready: 0, capture_required: 0, needs_human: 0}`

#### Gap queue

| operator_id | scan_verdict | golden | demo_role | repair | evidence |
|---|---|---|---|---|---|
| _empty_ |  |  |  |  |  |

完整 operator 列表保存在 Scan Run；Spec 只保留 gap queue 和计数。

### CUDA Capture

- `capture_session_id`: `null`
- `session_status`: `NOT_STARTED`
- `golden_run`: `null`
- `captured_sample_counts`: `{}`

`session_status` 只用 `NOT_STARTED | ACTIVE | SEALED | FAILED`。一旦 Session 为 `SEALED` 或 `FAILED`，不得创建第二个 Session。

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

baseline replay 不算修复尝试。每轮修改源码前递增 `attempts_used`，每轮只有一个 `active_hypothesis`。失败 Run 封存后恢复同一固定基线；通过 patch 是 P800 工作区唯一未提交修改。

### Demo Closure Evidence

| item | status | evidence |
|---|---|---|
| Contract approved and tool bindings match | `PENDING` | `null` |
| target-only eager scan complete | `PENDING` | `null` |
| gap queue complete | `PENDING` | `null` |
| one CUDA Session and at most three samples per operator | `PENDING` | `null` |
| CUDA self-replay passed | `PENDING` | `null` |
| bundle verified on CUDA and P800 | `PENDING` | `null` |
| selected operator failed P800 baseline | `PENDING` | `null` |
| repair stayed inside boundary and attempt limit | `PENDING` | `null` |
| all selected samples passed Precision Gate | `PENDING` | `null` |
| passing patch is the only workspace change | `PENDING` | `null` |
| failed Runs traceable and next_action none | `PENDING` | `null` |

### Stop reason

- `stop_reason`: `Contract 尚未批准，必填值仍未填写。`
- `human_question`: `请提供并批准 human_owner、spec_id、两侧源码 revision、checkpoint id 与 config digest、扫描输入模式、atol 和 rtol，并把 contract_revision 设为 1，可以吗？`
- `resume_requires_contract_revision`: `true`

### Decisions

只追加会影响恢复执行的决定；详细过程写入 Run。

| state_revision | decision | evidence |
|---|---|---|
| `0` | 创建未批准的 Spec 模板，等待人填写并批准 Contract | `migration-spec.md#contract` |

<!-- AGENT-WRITABLE WORKING STATE: END -->

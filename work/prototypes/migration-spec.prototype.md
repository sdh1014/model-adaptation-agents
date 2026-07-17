# PROTOTYPE — 单文件 `migration-spec.md`

> 历史说明（2026-07-17）：本纸面原型中的 EAGLE/draft 范围已被 Contract revision 3 的 target-only eager 决策取代。文件只保留为早期设计推演，不得作为当前运行输入；当前事实以根目录 `migration-spec.md` 和 `runs/scan-003` 为准。
>
> 要验证的问题：一次对话上下文全部丢失后，新 Agent 是否能只读取这一个文件，恢复不变目标、当前阶段、活动算子、证据位置和唯一下一动作；同时，Agent 是否会在权限、一次性 CUDA 采集或 Contract 固定的修复上限处正确停止。
>
> 这是供人工评审的纸面 logic prototype，不是最终 Schema，也不是工具实现。

# Migration Spec

<!-- HUMAN-OWNED CONTRACT: BEGIN
Agent 必须完整读取本区，但不得修改其中任何字符。
人修改本区时必须递增 contract_revision，并在 Human Decisions 中说明原因。
-->

## Contract

### Contract Data

下面的 JSON 区块是 Deterministic Tool 唯一解析的 Spec 内容。它只保存脚本执行必须使用的固定约束；目标说明、权限边界、停止条件和状态迁移仍由本 Contract 的 Markdown 正文表达，Working State 仍由 Migration Agent 读取和更新。

脚本通过 `--spec <path>` 读取本区块，不解析其他 Markdown，也不读取或修改 Working State。任何必填值为 `null` 时，Contract 尚不可执行。脚本启动时只读取一次，并把规范化 JSON 的 SHA-256 与 `spec_id`、`contract_revision` 一起写入 `result.json`；不对整份 Spec 计算绑定摘要，因为 Working State 会在动作完成后继续变化。

```json
{
  "schema": "migration-spec/v0",
  "spec_id": "step3p7-flash-p800-demo",
  "contract_revision": 1,
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
    "draft_entry": "Step3p5MTP.forward"
  },
  "demo_input_mode": null,
  "limits": {
    "max_samples_per_operator": 3,
    "max_repair_attempts": 5
  },
  "precision_gate": {
    "comparator": "torch.testing.assert_close",
    "atol": null,
    "rtol": null,
    "require_exact_structure": true,
    "require_same_dtype": true,
    "require_finite": true,
    "check_stride": false
  }
}
```

`null` 只是未批准模板的合法 JSON 表达；人类批准 Contract 前必须全部替换为实际值。运行期参数，例如当前 Run、Golden Run、执行模式和目标样本路径，不进入 Contract Data，由 Agent 作为脚本的显式参数传入。

### Bootstrap

当 `migration-spec.md` 尚不存在时，用户仍调用同一个自然语言 Skill，不增加 `init` 命令。Skill 可以创建本文件并根据用户输入和源码证据填写 Contract；存在未确定字段时，必须写为 `NEEDS_HUMAN / SCAN` 并停止，不得开始扫描或调用 Deterministic Tool。

人类补齐并明确批准 Contract 后，记录 `contract_revision: 1` 和初始 Human Decision，再由同一个 Skill 校验并初始化为 `ACTIVE / SCAN`。从初次批准开始，Contract 对 Migration Agent 只读；以后只能根据新的人工决定递增 revision。下面的 Working State 示例表示“Contract 已批准且所有 `<...>` 已替换”的运行状态，带占位符的模板本身不可执行。

### Identity

- `human_owner`: `<name>`

`schema`、`spec_id` 与 `contract_revision` 的唯一机器可读值位于 Contract Data。人修改 Contract 时必须同步更新其中的 `contract_revision`，不得在 Markdown 正文再维护一份重复值。

本文件只使用 `CONTEXT.md` 已定义的领域词。`status`、`phase`、`WAITING` 只是三个流程字段，不升级为新的领域概念；失败原因直接写清楚，不另建错误码、子状态机或术语表。

### Goal

扫描固定版本 Step-3.7-Flash 的实际 target 与 EAGLE draft 模型路径，记录所有 Operator Gap；在一次 CUDA Capture Session 中采集所需 Golden Samples，由人工把 Handoff Bundle 复制到 P800；随后由 Migration Agent 在最多五轮修复内关闭一个真实缺口算子，最多验收三种实际 shape。

### Fixed inputs

固定源码版本、checkpoint、target/draft 入口、输入模式以及样本和修复上限统一保存在 Contract Data。Contract Data 中任何必填值仍为 `null`，或者 Markdown Contract 中仍有必须由人决定的 `<...>` 时，Agent 不得开始扫描，必须进入 `NEEDS_HUMAN`。

本 Demo 的 `sglang_revision` 必须已经包含人工批准的特殊 SwiGLU helper 抽取，并在 CUDA 与 P800 使用同一固定源码；helper 抽取不属于 P800 repair attempt。

### Demo closure

只有同时满足以下条件，Working State 才能写为 `PASS`：

1. target 与 draft 实际路径均已扫描完成，全部扫描结论都有源码或运行证据指针；
2. 所有发现的 Operator Gap 都已进入 gap queue，并在唯一一次 CUDA Capture Session 中完成所需采集；
3. 被选作 Demo 的算子在 P800 baseline replay 中确实失败，证明它是真实 correctness gap；
4. 该算子的修复没有超出 Repair Boundary；
5. 最多三种已采集 shape 全部通过 Precision Gate；
6. 修复尝试不超过 Contract 的 `max_repair_attempts`，且最终通过的 patch 位于 `passing_run`。

完整关闭其他缺口、完整模型组网、回复质量和性能均不是 Demo Closure 条件。

### Precision Gate

- 比较器和固定 `atol/rtol` 从 Contract Data 的 `precision_gate` 读取；用户已在 P800 实机验证 `torch.testing.assert_close` 可用；
- 输出结构必须一致；
- 输出 dtype 必须一致；
- CUDA Golden 与 P800 输出都必须为有限值；
- 每个 Golden Sample 都必须单独通过；Agent 不得修改或放宽门槛。

比较器先按 Golden 元数据检查结构、shape、dtype 和有限值，再把 expected 与 actual 保持原 dtype 复制到 CPU，逐 Tensor 显式使用 Contract Data 中的 `atol/rtol` 调用 `torch.testing.assert_close`。输出 stride 不属于本 Demo 的 Precision Gate。P800 actual output 只在内存中比较，Run 保存 `compare.json` 和日志，不额外保存 actual Tensor。

### Authority and environment boundary

- Migration Agent 可以：读取源码与证据；更新 Working State；调用 capture/replay/compare 等 Deterministic Tool；在 Repair Boundary 内生成候选 patch。
- Migration Agent 不可以：修改 Contract；自动跨机器复制；取得或使用新凭证；扩大到 DecoderLayer/完整模型；新增 C++、自定义 Kernel 或底层注册；修改 Precision Gate。
- 同一个自然语言 Skill 是首次启动和恢复的唯一人类入口，不增加顶层编排 CLI。
- Deterministic Tool 只解析 Contract Data JSON 中自己需要的字段，并接收当前动作的显式参数；不解析其余 Markdown，不读取或修改 Working State。Migration Agent 根据工具写入 Run 的结果更新 Working State。
- Deterministic Tool 不提供覆盖固定 Contract 值的参数，例如 `--atol`、`--rtol` 或 `--max-repair-attempts`；其 `result.json` 必须记录 `spec_id`、`contract_revision` 和规范化 Contract Data 的 SHA-256。
- CUDA 机器只允许一个 `capture_session_id`。Session 一旦 `SEALED` 或 `FAILED`，Agent 不得创建第二个 Session。
- 工具负责生成和校验 Handoff Bundle；人工负责复制；P800 端必须先校验 manifest，才能开始 replay。
- P800 修复只允许 Python、P800 可执行的 PyTorch 和已有 xspeedgate/kunlun_ops 能力。
- `max_repair_attempts` 由人工在 Contract Data 中固定为 `5`，Migration Agent 不得修改或绕过。

### State model

`status` 与 `phase` 分开，避免把“正在做什么”和“为什么停止”混成一个字段。

- `status`: `ACTIVE | WAITING | PASS | BLOCKED | NEEDS_HUMAN`
- `phase`: `SCAN | CUDA_CAPTURE | HANDOFF | P800_REPAIR | DONE`

其中：

- `WAITING` 只表示计划内的人工跨机器复制，不代表判断失败；
- `NEEDS_HUMAN` 表示缺少人类决定或无法可靠确定调用/边界；
- `BLOCKED` 表示原因已经明确，但继续会超出 Contract、Repair Boundary、次数或环境权限；
- `PASS`、`BLOCKED`、`NEEDS_HUMAN` 都会终止当前 Agent 执行。

`WAITING` 也会暂停当前 Agent，但它是计划内交接点；P800 校验成功后可以按下表自动恢复，不需要人修改 Contract。

### Allowed transitions

| From | Guard | To | 必须同步写入 Working State |
|---|---|---|---|
| `ACTIVE / SCAN` | target、draft 均扫描完成；没有未决边界；至少有一个采集候选 | `ACTIVE / CUDA_CAPTURE` | Scan Run、覆盖计数、gap queue、唯一下一动作 |
| `ACTIVE / SCAN` | 实际调用或 Semantic Operator 边界不能可靠确定 | `NEEDS_HUMAN / SCAN` | stop reason、证据、一个明确的人类问题 |
| `ACTIVE / SCAN` | 固定范围内没有任何可成为 Demo 的缺口候选 | `BLOCKED / SCAN` | 写清“固定范围内没有候选”并指向 Scan Run |
| `ACTIVE / CUDA_CAPTURE` | 唯一 Session 完成；Golden self-replay 和 bundle 校验通过 | `WAITING / HANDOFF` | Golden Run、bundle、manifest、人工复制命令 |
| `ACTIVE / CUDA_CAPTURE` | Session 已消耗且无法形成有效 Golden | `BLOCKED / CUDA_CAPTURE` | 写清采集失败原因并指向失败 Run；禁止重开 Session |
| `WAITING / HANDOFF` | bundle 已在 P800 上通过 manifest 校验 | `ACTIVE / P800_REPAIR` | 执行位置、P800 校验 Run、baseline replay 作为唯一下一动作 |
| `ACTIVE / P800_REPAIR` | baseline replay 通过 | 保持 `ACTIVE / P800_REPAIR`，换下一个已采集候选 | 证明它不是真实 gap；不消耗 repair attempt |
| `ACTIVE / P800_REPAIR` | 所有已采集候选的 baseline 都通过 | `BLOCKED / P800_REPAIR` | 写清“没有真实 gap”并指向全部 baseline Run；不得把“缺少专用优化”当成修复成果 |
| `ACTIVE / P800_REPAIR` | baseline replay 失败且边界明确 | 保持 `ACTIVE / P800_REPAIR` | 选定 Demo 算子；`attempts_used = 0`；写第一条假设 |
| `ACTIVE / P800_REPAIR` | 本轮比较失败且 `attempts_used < max_repair_attempts` | 保持 `ACTIVE / P800_REPAIR` | 失败 Run、恢复基线、下一轮单一假设 |
| `ACTIVE / P800_REPAIR` | 全部 Demo samples 通过且满足 Demo Closure | `PASS / DONE` | 最终 patch、通过 Run、closure checklist |
| `ACTIVE / P800_REPAIR` | 需要 C++/新 Kernel 注册，或第 `max_repair_attempts` 轮仍失败 | `BLOCKED / P800_REPAIR` | stop reason、全部尝试 Run、越界说明 |
| 任意 `ACTIVE` 状态 | 下一动作需要无法由证据决定的人类选择 | `NEEDS_HUMAN / 当前 phase` | 唯一问题和恢复所需证据 |

`PASS` 不可恢复；需要新目标时新建 Spec。`BLOCKED` 或 `NEEDS_HUMAN` 只能在人类更新 Contract、递增 `contract_revision` 并记录解决决定后恢复为 `ACTIVE`。`WAITING` 在人工复制并通过 P800 校验后可直接恢复，不需要改 Contract。

### Recovery protocol

新 Agent 或上下文压缩后的 Agent 只执行以下步骤：

1. 完整读取 Contract，确认 Contract Data 和 Markdown 中所有固定字段已填写；
2. 对比 `contract_revision` 与 `observed_contract_revision`；不一致时先重新对齐 Working State，不执行外部动作；
3. 读取 Current、active operator、上一 Run 和对应 evidence；不默认加载全部历史 Run；
4. 检查当前 `status / phase` 是否是 Allowed transitions 中的合法组合；
5. `ACTIVE` 时只执行 `next_action`；执行前创建或确定 Run，执行后立即更新 Working State；
6. 发现 Spec 自相矛盾、下一动作不唯一或证据指针失效时，写入 `NEEDS_HUMAN` 并停止。

Recovery protocol 属于 Contract，不另建状态机文件。

### Human Decisions

| contract_revision | decision | reason |
|---|---|---|
| `1` | `<initial contract approved>` | `<human rationale>` |

<!-- HUMAN-OWNED CONTRACT: END -->

<!-- AGENT-WRITABLE WORKING STATE: BEGIN
Agent 每次动作前完整读取 Contract 与本区；每次动作结束后立即更新本区。
不得把聊天记录当作状态来源。Run 是不可变证据，本区只保留恢复执行所需的摘要和指针。
-->

## Working State

### Current

- `observed_contract_revision`: `1`
- `state_revision`: `0`
- `status`: `ACTIVE`
- `phase`: `SCAN`
- `execution_site`: `SOURCE`
- `active_operator`: `null`
- `last_completed_action`: `spec_initialized`
- `last_run`: `null`
- `next_action`: `从固定 target 与 draft 入口扫描实际模型路径，并生成第一个 Scan Run。`

规则：`ACTIVE` 时 `next_action` 必须恰好有一条；`WAITING` 时必须是一条人工动作；终止状态下必须为 `none`。

### Scan

- `scan_run`: `null`
- `target_coverage`: `PENDING`
- `draft_coverage`: `PENDING`
- `operator_counts`: `{ready: 0, capture_required: 0, needs_human: 0}`

#### Gap queue

| operator_id | scan_verdict | golden | demo_role | repair | evidence |
|---|---|---|---|---|---|
| _empty_ |  |  |  |  |  |

完整 operator 列表保存在不可变 Scan Run；Spec 只保留 gap queue 和计数，防止文件无限增长。

### CUDA Capture

- `capture_session_id`: `null`
- `session_status`: `NOT_STARTED` <!-- NOT_STARTED | ACTIVE | SEALED | FAILED -->
- `golden_run`: `null`
- `captured_sample_counts`: `{}`

### Handoff

- `bundle_status`: `NOT_BUILT` <!-- NOT_BUILT | VALID | COPIED | VERIFIED_ON_P800 -->
- `bundle_path`: `null`
- `manifest_path`: `null`
- `p800_verification_run`: `null`

### P800 Repair

- `baseline_run`: `null`
- `attempts_used`: `0`
- `active_hypothesis`: `null`
- `passing_run`: `null`

baseline replay 不算修复尝试。每轮只有一个 `active_hypothesis`；开始修改源码前先将 `attempts_used` 加一并让通用 `last_run` 指向本轮 Run，通过轮也计数。一次失败必须先写入 Run 和 Working State，才可进入下一轮。

修复基线就是 Contract 固定的 SGLang 与 SGLang-Kunlun revisions。开始 baseline replay 和每轮尝试前，相关源码必须与基线一致且没有预先存在的修改。每轮补丁都相对同一基线完整生成并保存在该轮 Run；失败后恢复基线，通过后只把通过补丁作为未提交修改留在 P800 工作区。工具不创建分支、不提交、不推送。

### Stop reason

- `stop_reason`: `null`
- `human_question`: `null`
- `resume_requires_contract_revision`: `null`

### Decisions

只追加对恢复执行有影响的决定；详细过程写入 Run。

| state_revision | decision | evidence |
|---|---|---|
| `0` | 初始化，开始 SCAN | `migration-spec.md#contract` |

<!-- AGENT-WRITABLE WORKING STATE: END -->

# PROTOTYPE-ONLY APPENDIX

以下内容用于本次人工走查，不进入正式运行时 `migration-spec.md`。

## Manual transition walkthrough

下面使用虚构的 `op.demo_gap` 检查状态能否表达完整流程；它不表示特殊 SwiGLU 已被证明是真实缺口。

1. **完成扫描**：`ACTIVE / SCAN -> ACTIVE / CUDA_CAPTURE`；Scan Run 指向完整清单，gap queue 加入 `op.demo_gap`，`next_action` 变为构建一次性 capture plan。
2. **完成唯一 CUDA Session**：`capture_session_id` 第一次且唯一一次赋值，状态变为 `SEALED`；Golden self-replay 与 bundle 校验通过后进入 `WAITING / HANDOFF`。
3. **人工复制**：Agent 停止；人复制 bundle。P800 校验 manifest 后进入 `ACTIVE / P800_REPAIR`，下一动作仅为 baseline replay。
4. **确认真实 gap**：baseline 失败但不消耗 attempt；记录失败 Run，并写入第一条修复假设。
5. **第一至四轮失败**：每轮开始修改前递增 `attempts_used`，从同一基线生成完整补丁；把补丁和结果保存到对应 Run 后恢复基线，下一动作只包含下一条假设。
6. **第五轮通过**：`attempts_used = 5`；最多三个 Golden Samples 全部通过固定 Precision Gate；完整补丁保存在 `passing_run`，并作为唯一未提交修改留在 P800 工作区；closure checklist 完整后进入 `PASS / DONE`。
7. **第五轮仍失败**：进入 `BLOCKED / P800_REPAIR`，写清“五轮已用尽”并指向五次 Run，不得开始第六轮。
8. **任何时刻缺少人类判断**：进入 `NEEDS_HUMAN / 当前 phase`，只保留一个明确问题；Agent 不得自己假设答案后继续。

## Prototype verdict

用户已确认：

- `status + phase` 两个字段足以表达流程；
- 计划内人工复制使用独立 `WAITING`，不滥用 `NEEDS_HUMAN`；
- `BLOCKED/NEEDS_HUMAN` 恢复必须由人递增 Contract revision；
- Scan Run 保存全量结果，Spec 只保存 gap queue、活动算子和关键指针；
- 修复上限由 Contract 的 `max_repair_attempts: 5` 固定；
- 删除与 `last_run`、`passing_run` 重复的 `last_attempt_run`、`candidate_patch`。

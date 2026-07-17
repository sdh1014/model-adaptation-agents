---
name: model-adaptation
description: 按模型参数启动或恢复 KLX P800 的 Spec 驱动模型适配。
argument-hint: "Model name, e.g. Step-3.7-Flash"
disable-model-invocation: true
---

# Input

把用户传给 Skill 的完整参数作为模型名。当前 Demo 只接受精确参数 `Step-3.7-Flash`。

参数缺失或不同于 `Step-3.7-Flash` 时，报告正确调用方式 `$model-adaptation Step-3.7-Flash` 并停止，不修改 Migration Spec。Spec 已存在时，参数必须与 Contract Data 的 `model` 一致；不一致时同样停止，不把调用错误写成迁移状态。

# Outcome

从迁移工作区中的 `migration-spec.md` 启动或恢复工作，只按其中唯一的 `next_action` 推进，直到当前执行到达 `WAITING`、`PASS`、`BLOCKED` 或 `NEEDS_HUMAN`。

`migration-spec.md` 是恢复执行的唯一状态来源：

- `Contract` 保存人类批准的目标、输入、门槛、权限和停止规则；批准后本 Skill 不得修改。
- `Working State` 保存 Agent 可更新的当前阶段、Run 指针和唯一下一动作。
- `runs/` 保存不可变的命令、日志、补丁和结果；聊天记录不能代替这些内容。

每次动作前完整重读 Spec。一次性采集或源码修改按对应分支先记录动作已开始；动作结束后先封存 Run，再更新完成状态。只要状态仍是 `ACTIVE`，就重新读取 Spec 并继续下一轮，不凭记忆接着做。

# Process

## 1. 找到或创建 Spec

默认使用当前迁移工作区的 `migration-spec.md`；用户给出其他路径时使用该路径。存在多个候选且无法从用户输入唯一确定时，停止并询问路径，不自行合并。

如果文件不存在：

1. 完整读取 [references/migration-spec-template.md](references/migration-spec-template.md)。只在这个分支加载模板。
2. 把模板复制为工作区的 `migration-spec.md`，不得改写模板文件。
3. 把已校验的模型参数写入 Contract Data 的 `model`；其他字段只用用户明确提供的值和可引用的源码证据填写，不猜 checkpoint、revision、输入模式或容差。
4. 只要仍有必填值未确定，就保持 `NEEDS_HUMAN / SCAN`、`next_action: none`，在 `human_question` 中集中写一个需要人回答的问题，然后停止。

Contract 首次批准前可以代填草稿；一旦 `contract_revision` 是已批准的正整数，后续 Contract 变化必须由人完成并写入 Human Decisions。本 Skill 只能修改 Working State。

**完成条件：** Spec 已存在；如果 Contract 尚未批准，Working State 准确停在 `NEEDS_HUMAN / SCAN`，且只有一个明确的人类问题。

## 2. 校验并恢复当前状态

完整读取 Contract 和 Working State，然后依次检查：

1. `<!-- CONTRACT-DATA: BEGIN -->` 与 `<!-- CONTRACT-DATA: END -->` 之间只有一个合法 JSON 对象。
2. Contract Data 的 `model` 与调用参数一致，其他必填值和 `human_owner` 均已填写，Contract 中没有未决占位；`contract_revision` 是正整数；同 revision 的人工批准记录存在。
3. Demo 固定约束没有漂移：TP 为 8、权重与计算 dtype 为 BF16、量化参数为空、`speculative_algorithm` 与 `draft_entry` 均为空、decode 和 prefill 的 CUDA Graph backend 均为 `disabled`、CUDA 与 P800 使用同一组启动参数且不显式指定 attention backend；原始算子边界为 `Step3p5MLP.forward`，只在 `self.limit is not None` 时处理，验证 rank 固定为 0；每算子最多三个 shape、最多五次修复、比较器为 `torch.testing.assert_close`，且 `atol=0.01`、`rtol=0.02`。
4. `status / phase` 是 Contract 允许的组合；`ACTIVE` 恰有一条 `next_action`，`WAITING` 恰有一条人工动作，终止状态为 `none`。
5. `last_run`、活动算子和阶段所引用的 Run/证据真实存在。只读取恢复所需的最近 Run，不默认加载全部历史。

如果 `contract_revision != observed_contract_revision`，先读最新 Human Decision 并重新对齐 Working State，不执行源码修改、CUDA 采集或 P800 重放。`BLOCKED` 或 `NEEDS_HUMAN` 只有在更高 Contract revision 明确解决原停止原因后才能恢复为 `ACTIVE`；`WAITING / HANDOFF` 在 P800 校验成功后可直接恢复，不需要修改 Contract。`PASS` 不可恢复，新目标使用新的 Spec。

首次批准有一个固定的对齐动作：当前状态必须仍是模板生成的 `NEEDS_HUMAN / SCAN`，最新 Human Decision 必须批准 revision 1，且所有 Contract 必填值已经通过上述校验。满足这些条件后，只更新 Working State：把 `observed_contract_revision` 设为当前 `contract_revision`，把 `state_revision` 加一，写为 `status: ACTIVE`、`phase: SCAN`、`last_completed_action: contract_approved`，并把唯一动作写为 `next_action: 运行 Spec 绑定自检`。同时清空旧的停止原因和人类问题，将 `resume_requires_contract_revision` 设为 `false`。若任一条件不满足，保持 `NEEDS_HUMAN / SCAN`，不得靠改 Working State 绕过批准。

Spec 绑定自检不做真实算子扫描。首次批准创建 `runs/spec-binding-001` 并写入 `last_run: runs/spec-binding-001`；后续每次 Contract revision 变化创建下一个未使用的 `runs/spec-binding-NNN`，本次 revision 4 使用 `runs/spec-binding-003`。调用 `scripts/replay_compare.py --spec <migration-spec.md> --run-dir <fresh-spec-binding-run> --mode synthetic`；这个模式使用脚本内固定且相等的 JSON 值，只验证 Contract Data 解析、摘要和 Run 写入，不代替后续 `torch.testing` 精度比较。只有退出码为 `0`、`passed: true`，且结果的三项 Spec 绑定与当前 Contract Data 全部一致时，才封存该 Run。随后把 Working State 的 `state_revision` 再加一，写入 `last_completed_action: spec_binding_smoke_passed`、`last_run: <fresh-spec-binding-run>`，并把唯一动作改为 `next_action: 完成 target-only eager 扫描并生成 Scan Run`。更新后重新读取 Spec；新会话必须先校验 `last_run` 的绑定，再执行这条扫描动作。工具未完成、结果不通过或绑定不一致时，不写成功状态，也不把它记为 Operator Gap。

若 Spec 自相矛盾、证据指针失效、下一动作不唯一，更新 Working State 为 `NEEDS_HUMAN / 当前 phase`，保留证据并只问一个问题。

**完成条件：** 当前状态、Contract revision 和最近证据彼此一致，并且要么得到一条可执行的 `next_action`，要么准确停止。

## 3. 按 phase 执行唯一动作

只进入当前 `phase` 对应的分支。不要提前执行后续阶段。

### `SCAN`

用源码搜索、加载后配置和调用关系分析完成扫描，不创建 `scan.py`：

- 只从 `Step3p7ForConditionalGeneration.forward` 扫描 target 路径；`draft_entry: null` 且不启用投机解码，因此不进入 `Step3p5MTP.forward`。
- 只扫描 Contract 固定的 TP8、BF16、target-only eager 分支。decode 和 prefill CUDA Graph 都禁用；CUDA Graph 管理只作为运行边界证据，不进入算子枚举。
- 两端命令都不显式指定 attention backend；不得在 Contract 中猜 backend。扫描或实际启动后分别记录 CUDA 与 P800 的解析结果和代码落点。
- 沿当前 checkpoint、配置和输入实际激活的分支下钻，到可单独采集、离线重放和替换的 Semantic Operator 为止。
- 扫描可以进入 `Step3p5MLP.forward` 内部判断两侧差异，但当前 Demo 的缺口必须归到原始 `Step3p5MLP.forward`；不得新增函数或把内部表达式登记为独立算子。
- 每个算子同时记录 CUDA 与 Kunlun 的实现证据；同名实现不自动算 `READY`，缺少专用 Kunlun Kernel 也不自动算 Operator Gap。
- 每条记录包含 `operator_id`、`model_path`、`activation_guard`、`boundary`、`cuda_impl`、`kunlun_impl`、`replay_replace_check`、`verdict` 和源码锚点。
- verdict 只用 `READY`、`CAPTURE_REQUIRED`、`NEEDS_HUMAN`。全部 `CAPTURE_REQUIRED` 项进入 gap queue 和唯一 Capture Session 的计划。

把完整结果写入新的 `runs/scan-NNN/result.json`，Spec 只保留覆盖计数、gap queue 和 Run 指针。调用或边界无法可靠确定时进入 `NEEDS_HUMAN / SCAN`；固定范围内没有采集候选时进入 `BLOCKED / SCAN`。

**完成条件：** target 有完整覆盖结论，每条结论都有两侧证据，Scan Run 绑定当前 Contract；若存在候选，状态为 `ACTIVE / CUDA_CAPTURE` 且下一动作只有一条。

### `CUDA_CAPTURE`

先确认所有 `CAPTURE_REQUIRED` 项都能在禁止保存权重和运行时对象的前提下形成可重放边界，再消耗唯一 CUDA Session。Capture plan 只作为这次动作的输入，不成为第二个状态文件。开始采集前，先把唯一 `capture_session_id`、`session_status: ACTIVE` 和 Golden Run 路径写入 Working State；恢复时只能检查并继续这个 Session，不能创建第二个。

如果唯一 `next_action` 明确写的是 `capture_golden.py preflight`，先完整读取 [references/cuda-capture-validation.md](references/cuda-capture-validation.md) 并只执行该步骤。preflight 用真实 SGLang Hook 和小型合成 BF16 tensor 验证原始 MLP 接线、rank 0 过滤和样本格式，但不启动模型、不访问 checkpoint，也不消耗正式 Session。无论通过还是失败，都报告 preflight Run 并停止，等待人把实机证据带回确认；不要设置 `capture_session_id`，不要修改 `session_status`，也不要自行进入正式采集。

在 CUDA 机器上：

1. 调用 `scripts/capture_golden.py`，通过 SGLang 的通用插件直接 Hook `sglang.srt.models.step3p5.Step3p5MLP.forward`；只处理 `self.limit is not None` 且 `tp_rank=0` 的实例，按调用签名去重，最多保存前三种真实 shape 的输入 `x`、CUDA `output` 和标量 `limit`。
2. 不新增模型算子函数，不保存 checkpoint 权重、`nn.Parameter`、module state、`gate_up`、`gate`、`up`、完整 batch、prompt/token、KV cache、stream/handle、随机状态或凭证。
3. CUDA 与 P800 的正式 SGLang 启动参数保持一致：TP8、BF16、`--cuda-graph-backend-decode disabled`、`--cuda-graph-backend-prefill disabled`；不传 `--speculative-algorithm`、量化参数、MTP 或 multi-layer EAGLE 开关，也不显式传 attention 或 MoE backend。CUDA 采集只多一个插件配置环境变量，实际解析出的 backend 写入启动日志。
4. 在同一次 CUDA Session 内，由已经加载同一 checkpoint 的 TP8 CUDA 模型在 rank 0 对全部 Golden Samples 做 self-replay；调用 `scripts/replay_compare.py --mode prepare-model-replay` 生成配置，模型内 Hook 完成比较，再用 `--mode finalize-model-replay` 收口结果。不得退回为无权重函数重放。全部通过固定 Precision Gate 后才能把 Golden Run 标为 `SEALED`。
5. 生成下一状态 Spec 临时副本，其中只修改 Working State 为 `WAITING / HANDOFF` 及唯一人工复制动作；调用 `scripts/handoff_bundle.py` 构建并校验 bundle。成功后用同一份 Spec 字节替换工作区 Spec 并停止；构建失败时不替换当前 Spec。

Session 已消耗但无法形成有效 Golden 时，写入失败 Run 并进入 `BLOCKED / CUDA_CAPTURE`；不得再次访问 CUDA 采集。

**完成条件：** 唯一 Session 已 `SEALED`，所有保留样本通过已加载模型内的 rank 0 self-replay，bundle 在 CUDA 端校验通过，状态原子地变为 `WAITING / HANDOFF`。

### `HANDOFF`

工具只生成和校验 bundle；人工负责跨机器复制。如果 bundle 尚未出现在 P800，报告 Spec 中的准确复制动作并停止。

在 P800 上收到 bundle 后，先在未修改的 bundle 上调用 `scripts/handoff_bundle.py` 校验 manifest；任何文件缺失、多出、大小或 SHA-256 不符时都不得读取 Golden Tensor。校验通过后，再用 bundle 中的 `migration-spec.md + runs/` 初始化 P800 迁移工作区，记录验证 Run，并把唯一下一动作设为第一个候选的 baseline replay。

**完成条件：** bundle 已在 P800 校验，Working State 为 `ACTIVE / P800_REPAIR`，`execution_site` 为 `P800`，下一动作只有 baseline replay。

### `P800_REPAIR`

先调用 `scripts/workspace_guard.py` 检查 Contract 固定 revisions 和干净工作区。发现无法解释的已有修改时进入 `NEEDS_HUMAN`，不得替用户清理。

对 gap queue 中已采集的候选逐个调用 `scripts/replay_compare.py` 做 baseline；P800 必须先加载同一 checkpoint 的 TP8 模型，并只在 rank 0 的相同 `Step3p5MLP` 实例内重放：

- 全部样本通过：该候选不是真实 correctness gap，换下一个候选，不增加 `attempts_used`。
- 任一样本执行或精度失败：记录 baseline Run，选择该算子，`attempts_used` 保持 `0`。
- 所有候选都通过：进入 `BLOCKED / P800_REPAIR`，理由写“没有真实 gap”。

确认真实 gap 后，每轮严格执行：

1. 从同一干净基线开始，只写一条 `active_hypothesis`。
2. 修改源码前创建本轮 Run，并把 `attempts_used` 加一；通过轮也计数。
3. 修改只限于活动 Semantic Operator 的 Python、P800 可执行的 PyTorch 或已有 xspeedgate/kunlun_ops 能力及聚焦测试。
4. 调用 `workspace_guard.py` 保存相对固定基线的完整 patch，再用 `replay_compare.py` 重放该算子的全部 Golden Samples。
5. 失败时先封存 Run，再调用 `workspace_guard.py` 恢复基线；形成下一条单一假设后才能继续。
6. 通过时用 `workspace_guard.py` 确认通过 patch 是工作区唯一修改，把 Run 写入 `passing_run`，核对全部 Demo Closure 后进入 `PASS / DONE`。

可信假设需要新 C++、自定义 Kernel、底层注册、DecoderLayer/完整模型改动或放宽 Precision Gate 时，立即恢复基线并进入 `BLOCKED`。第五轮仍失败时同样 `BLOCKED`，不得开始第六轮。

**完成条件：** 要么一个真实 gap 的全部已采集样本通过固定门槛且 closure 证据齐全，要么以可追溯证据停在 `BLOCKED` 或 `NEEDS_HUMAN`。

## 4. 接受结果并更新 Working State

四个脚本都从本 Skill 目录下解析，并至少接收：

```text
--spec <migration-spec.md> --run-dir <fresh-run-directory>
```

调用前确认本阶段所需脚本存在。缺少脚本时报告缺失路径并停止，不把“Skill 包未完成”写成 Operator Gap 或迁移状态。不得传入覆盖 Contract 的 `--atol`、`--rtol`、`--max-shapes` 或 `--max-repair-attempts`。

工具退出码为 `0` 且产生合法 `result.json`，只说明动作已形成可信结果；`passed: false` 可以是有效的 baseline 或精度失败。非零退出不能证明 Operator Gap。

接受结果前，重新解析当前 Contract Data，用规范化 JSON 计算 SHA-256，并核对 `spec_id`、`contract_revision`、`contract_data_sha256` 三项。任一不一致都不得推进状态。

接受后先封存 Run，再更新 `last_completed_action`、`last_run`、阶段字段、`state_revision` 和下一状态。Run 封存后不可修改；`ACTIVE` 只能保留一条 `next_action`。不要创建第二个状态文件，Handoff 构建所需的下一状态 Spec 临时副本除外。

**完成条件：** Working State 可以只依靠当前 Spec 与被引用的 Run 恢复；不存在未封存却被当成成功证据的 Run，也不存在两个下一动作。

## 5. 继续或停止

- `ACTIVE`：回到第 2 步，重新读取 Spec 后继续。
- `WAITING`：报告 bundle、manifest 和人工复制动作，然后停止。
- `PASS`：报告活动算子、样本数、`passing_run` 和最终 patch，然后停止。
- `BLOCKED`：报告停止原因、边界或次数、相关 Run，然后停止。
- `NEEDS_HUMAN`：报告证据和 Working State 中唯一的 `human_question`，然后停止。

不自动 SSH、上传、下载、登录、管理凭证、创建分支、commit 或 push。

**完成条件：** 最终回复与 Spec 的实际 `status / phase` 一致，并给出恢复所需的准确路径或人工动作。

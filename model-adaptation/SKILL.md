---
name: model-adaptation
description: 按模型参数启动或恢复 KLX P800 的 Spec 驱动模型适配。
argument-hint: "Model name, e.g. Step-3.7-Flash"
disable-model-invocation: true
---

# Input

把用户传给 Skill 的完整参数作为模型名。当前 Demo 只接受精确参数
`Step-3.7-Flash`。

参数缺失或不同于 `Step-3.7-Flash` 时，报告正确调用方式
`$model-adaptation Step-3.7-Flash` 并停止。Spec 已存在时，参数还必须与
Contract Data 的 `model` 一致。

# Outcome

从工作区中的 `migration-spec.md` 启动或恢复工作，只执行其中唯一的
`next_action`。一个算子通过不是停止条件：只要 gap queue 还有已封存 Golden 的
待修复项，就自动选择下一项并继续。只有计划内跨机器动作需要 `WAITING`、全部
gap queue 已关闭时达到 `PASS / DONE`，或状态进入 `BLOCKED` /
`NEEDS_HUMAN` 时才停止。

`migration-spec.md` 是恢复执行的唯一状态来源：

- `Contract` 保存人类批准的扫描范围、运行参数、样本规则、精度门槛和权限边界；
  批准后本 Skill 不得修改。
- `Working State` 保存当前阶段、gap queue、扫描完成后选出的
  `active_operator`、每个算子的 `golden_run / repair_status /
  attempts_used / passing_run`、Run 指针和唯一下一动作。
- `runs/` 保存不可变证据；聊天记录不能替代它。

每次动作前完整重读 Spec。动作完成后先封存 Run，再更新 Working State。只要状态
仍是 `ACTIVE`，就重新读取 Spec 后继续，不凭对话记忆恢复。

# Process

## 1. 找到或创建 Spec

默认使用当前工作区的 `migration-spec.md`。如果文件不存在：

1. 完整读取
   [references/migration-spec-template.md](references/migration-spec-template.md)。
2. 复制模板为工作区的 `migration-spec.md`。
3. 只用用户明确提供的值和可引用源码证据填写 Contract；不要猜 checkpoint、
   revision、输入模式或容差。
4. 只要有必填值未确定，就保持 `NEEDS_HUMAN / SCAN`、
   `next_action: none`，只写一个需要人回答的问题，然后停止。

Contract 首次批准前可以形成草稿。批准后的变化必须由人递增
`contract_revision` 并追加 Human Decision；Skill 只能修改 Working State。

## 2. 校验并恢复

完整读取 Contract 和 Working State，依次检查：

1. Contract Data 是唯一合法 JSON，`model` 与 Skill 参数一致，
   `contract_revision` 是正整数，且存在同 revision 的 Human Decision。
2. 固定运行条件没有漂移：target-only、TP8、BF16、无量化、无 draft 和投机解码、
   decode/prefill CUDA Graph 都禁用、两端启动参数一致、不显式指定 attention
   backend。
3. 扫描约束没有漂移：`model_paths=["target"]`、粒度为 `kernel-call`、输入模式为
   `text-only` 和 `single-image`。
4. 样本约束没有漂移：rank 0、每个算子最多三个 shape；保存输入和 CUDA 期望
   输出；允许保存当前 kernel 调用直接使用的当前 rank 参数 Tensor；禁止完整
   checkpoint、module `state_dict` 和无关参数。
5. 比较器为 `torch.testing.assert_close`，固定 `atol=0.01`、
   `rtol=0.02`，结构、dtype 和有限值检查不变。
6. Contract Data 中不存在 `active_operator`、`operator_boundary` 或单数
   `demo_input_mode`。`active_operator` 只能在扫描完成后出现在 Working State。
7. `ACTIVE` 恰有一条 `next_action`；最近 Run、gap queue 和活动算子引用真实存在。
   每个 gap queue 行都有 `golden_run`、`repair_status`、`attempts_used` 和
   `passing_run`；最多一行为 `ACTIVE`，且它必须等于 `active_operator`。
   `passing_run` 只指向带累计 `candidate.patch` 的通过 attempt；baseline 直接
   PASS 的行沿用此前最近一份非空 `passing_run`，此前没有补丁时保持 `null`。
8. revision 5 的单算子 runbook、Golden 和 adapter 都是历史证据。当前
   `contract_revision` 大于 5 时，不得执行这些旧 runbook，也不得把它们当成当前
   多算子 Session 已具备的能力。

如果 `contract_revision != observed_contract_revision`，先读最新 Human Decision，
对齐 Working State，不能直接执行采集或重放。

首次批准时，模板状态必须仍是 `NEEDS_HUMAN / SCAN`。校验通过后，把
`observed_contract_revision` 设为当前 `contract_revision`，把
`state_revision` 加一，并写入 `status: ACTIVE`、`phase: SCAN`、
`last_completed_action: contract_approved`、
`next_action: 运行 Spec 绑定自检`。

每个新 revision 使用下一个未占用的 `runs/spec-binding-NNN`，调用：

```text
scripts/replay_compare.py \
  --spec <migration-spec.md> \
  --run-dir <fresh-spec-binding-run> \
  --mode synthetic
```

只有退出码为 0、`passed: true` 且三项 Spec 绑定完全一致时，才封存绑定 Run 并进入
扫描。首次绑定写入 `last_run: runs/spec-binding-001`、
`last_completed_action: spec_binding_smoke_passed` 和
`next_action: 完成 target-only eager 扫描并生成 Scan Run`。后续 revision 使用
实际的新 Run 路径。这个动作不代表任何算子已扫描或精度已验证。

## 3. 按 phase 执行唯一动作

### `SCAN`

用源码搜索、加载后配置和实际启动证据完成扫描，不创建顶层扫描编排器。

#### 扫描范围

- 从 `Step3p7ForConditionalGeneration.forward` 出发，只进入 target 路径，不进入
  `Step3p5MTP.forward`。
- 分别沿 `text-only` 和固定最小 `single-image` 请求会激活的路径向下扫描。
- Step-3.7 的固定单图输入使用当前固定 SGLang worktree 中的
  `examples/assets/example_image.png`。Scan Run 必须保存其 SHA-256；请求文本包含
  `<im_patch>`。不得在正式 Session 中临时下载 URL 或换图。
- 最小边界是源码中已经存在、能直接描述输入输出并可独立替换的 Kernel Call。
  CUDA extension、Triton、SGLang JIT、第三方 kernel 和实际不兼容的
  Torch 调用都可以进入清单。
- 不把 view、reshape、split 等只改元数据的表达式单独列为缺口。
- 不新增 helper、自定义算子函数、模型方法或整层 wrapper 作为扫描、捕获或重放
  边界。

#### 两端判断

对每个 CUDA 调用沿 Kunlun 路径检查三种情况：

1. 同一调用已有 Kunlun 实现；
2. Kunlun 在更高调用点改走等价实现，因而不会进入 CUDA kernel；
3. 固定参数下仍会进入未绑定或不可执行的调用。

前两种为 `READY`。只有第三种或源码不足以证明等价时才进入 gap queue。不能因为
缺少同名 Kunlun symbol 就直接判缺口，也不能把 “CUDA 使用 Triton” 当作入选硬
条件。

每条记录至少包含：

- `operator_id`、输入模式、激活条件和从模型入口开始的 `call_chain`；
- 现有 `kernel_call.symbol`、实现类型和现有 capture seam；
- 输入、当前调用直接使用的参数、非 Tensor 参数和输出；
- CUDA/Kunlun 源码或运行证据；
- 等价路径判断、可重放/可替换判断和 verdict。

verdict 只用 `READY`、`CAPTURE_REQUIRED`、`NEEDS_HUMAN`。

#### 扫描后形成修复顺序

先完成并封存整个 Scan Run，再从它的 gap queue 比较候选。至少比较
`topk_sigmoid`、视觉 attention 和其他真实缺口。依次优先：

1. 固定输入下必达；
2. 边界输入、输出和直接参数更少；
3. 不依赖 TP 通信、KV cache 或大型运行时对象；
4. 可以用 P800 Torch 或已有 xspeedgate/kunlun_ops 在 Repair Boundary 内修复。

按上述顺序写入 gap queue。第一项写入 Scan Run 的
`selection.active_operator`，并把同一个值写入 Working State 的
`active_operator`；但 `capture_plan` 必须包含本轮准备修复的全部 gap queue
条目，不能只包含第一项。每行初始写
`golden_run: null / repair_status: PENDING / attempts_used: 0 /
passing_run: null`。Contract 保持不变。静态缺口只能写
`STATIC_GAP_CANDIDATE`；P800 baseline 失败后才能称为实机缺口。

完整结果写入新的 `runs/scan-NNN/result.json`。旧 Run 不可修改。

当前 revision 6 Scan Run 是 `runs/scan-007`。它把五个
`CAPTURE_REQUIRED` 项全部写入同一 `capture_plan`，其中包括单图路径上的
`prefill_attention._fwd_kernel`。该项 Hook 现有
`context_attention_fwd`：函数返回 `None`，结果写入参数 `o`，所以 capture
保存调用前的 `q/k/v/b_start_loc/b_seq_len` 和标量参数，并在原调用完成后把 `o`
保存为 `output`。这只固定 CUDA Golden 边界，不预先指定 attention 在 P800 上应
改用哪个实现。

### `CUDA_CAPTURE`

逐项检查 Scan Run 的 `capture_plan.adapter_status`。Scan Run 不可改写；如果某项
是 `NOT_IMPLEMENTED`，只有 Working State 的 Closure Evidence 已把该
`operator_id` 的 adapter 标为 `PASS`，且所指 Run 绑定当前 Contract、该算子、
当前源码摘要和 `passed: true`，才表示正式 Session 可以包含它。
`last_completed_action` 和 `last_run` 随后可以推进到其他动作，不能因此丢失已经
封存的 adapter closure evidence。

如果任一计划项的 adapter 尚未实现：

1. 把实现限制在现有 `capture_golden.py`、`replay_compare.py` 和采集插件中；
2. Hook Scan Run 指定的现有 capture seam，不新增自定义算子函数；
3. 用测试证明只保存 Contract 允许的输入、直接参数、非 Tensor 参数和 CUDA 输出，
   最多三个 shape，且 rank 0 以外不落盘；
4. 用测试证明 CUDA self-replay 调 Scan Run 记录的 CUDA capture seam。P800
   replay 入口不由 CUDA capture plan 猜测：除已有 SwiGLU adapter 外，保持
   `PENDING`，等队列推进到该算子时由 Agent 依据 Kunlun 原调用点补齐；两端都不得
   新增模型 helper 或自定义算子；
5. 把实现与测试证据写入新的 Run，不改写 Scan Run；所有计划项 adapter 都完成
   后，才更新 Working State 的 Closure Evidence、`last_completed_action`、
   `last_run` 和唯一 preflight 下一动作。
6. `capture_golden.py --mode prepare-session` 必须生成一个 config，插件从中加载
   全部 capture plan collector；不能靠逐项 `prepare` 宣称一次 Session。一个
   bundle 包含全部 Golden Run 的能力仍要在正式 Session 前通过测试封存。

当前仓库中的 `Step3p5MLP.forward` capture/replay adapter 只属于 revision 4
历史方案，不能消费当前 kernel-scan Contract。缺少当前活动 kernel 的
capture/replay adapter
时，准确报告缺口并执行对应实现 ticket；不要运行旧 MLP preflight，也不要消耗
CUDA Session。

adapter 完成后，为当前 Contract revision 生成并审查新的 preflight runbook，
先执行不消耗正式 Session 的 preflight。现有
[references/cuda-capture-validation.md](references/cuda-capture-validation.md)
和 [references/formal-cuda-capture.md](references/formal-cuda-capture.md)
只绑定 revision 5，当前 revision 不得直接执行。新 preflight 必须验证：

- Scan Run 和当前 Contract 绑定；
- Hook 的是 Scan Run 记录的现有调用；
- rank 0 和最多三个 shape 的去重；
- 直接参数可保存，但完整 checkpoint 和 module state 会被拒绝；
- 相同样本可以按固定 Precision Gate self-replay。

revision 6 的多算子 preflight 入口是：

```text
python3 model-adaptation/scripts/capture_golden.py \
  --spec migration-spec.md \
  --run-dir runs/cuda-preflight-r6-001 \
  --mode preflight-session \
  --scan-result runs/scan-007/result.json \
  --sglang-worktree "$SGLANG_CUDA_WORKTREE"
```

它在一个配置中预检五个现有调用；每项分别验证三种 shape、rank 过滤和 CUDA
self-replay。这个命令不加载正式 checkpoint，也不消耗唯一 Capture Session。
只能在 CUDA 机器运行；SOURCE 机器上的单元测试不能替代它。

这一步不宣称其余四项已经有 P800 replay adapter。正式 Golden 会保存原 CUDA
Kernel Call 的完整最小边界；P800 队列推进到每一项前，Agent 再根据 Kunlun 源码
补齐该项 baseline/candidate 参数装配。adapter 缺失必须报告为流程能力待补齐，
不能记成 Operator Gap。

如果 adapter Run 记录的任一源码 SHA 在最近一次 PASS preflight 后发生变化，旧
preflight 只能作为历史证据；Closure Evidence 必须把当前源码 preflight 标为
`PENDING`，且不得开始正式 CUDA Session。先按 runbook 生成新的 preflight Run。

如果 Working State 的 `execution_site` 是 `CUDA`，但当前会话不在用户指定的 CUDA
机器，不创建 preflight Run，也不运行合成替代品。只报告 runbook 中的命令和
GitHub evidence 分支回传要求，保持 Working State 不变并停止本次执行。

preflight 通过且唯一下一动作已进入正式 Session 时，完整读取当前 revision 新生成
且绑定新 Scan Run、全部 adapter Run 和全部 capture plan 的正式 runbook。正式
Session 启动前仍先通过 GitHub 提交占位；模型停止后不再为了审查样本做一次中途
回传。同一个 CUDA Agent 继续在本地完成样本审查、临时 Spec、bundle build 和
verify，全部通过后再一次性回传 Session、Golden、record-samples、bundle 和审查
证据。后处理不得重启模型，也不算第二个 Capture Session。

正式模型启动前，必须先把 `session-start.json` 和 prepare 证据提交到 runbook
固定的 GitHub evidence 分支。只有首次 push 成功才消耗并授权该 Session；成功或
失败都只向同一分支追加，不能换分支或 Run 名重试。

不得在真实 Golden 回传前预填 shape 数量或伪造临时 Spec，也不得让人自行修改
Working State。

正式 CUDA Session 一次采集 capture plan 中的全部算子：

1. 启动固定 checkpoint 的真实 TP8/BF16/eager 模型；采集只增加插件配置，不改变
   两端模型启动参数。
2. 同一模型进程依次发送 Scan Run `request_set` 中的固定文本请求和固定单图请求。
   启动前重新计算图像 SHA-256；不一致时停止。启动日志必须确认实际 multimodal
   backend；attention collector 没有命中时不得把 Session 封存为成功。
3. 对每个算子，每种真实 shape 最多保存一份 rank 0 样本。样本包含该调用的输入、
   直接参数、必要标量和 CUDA 期望输出。一次模型进程同时服务所有 collector；
   “每个算子最多三种 shape”不能误计成整个 Session 只有三个样本。
4. 停止采集进程后、CUDA self-replay 前，调用 `handoff_bundle.py
   --mode record-samples --scan-result <当前 Scan result.json>`，把每个 Golden
   Sample 文件的 shape、相对路径、大小和 SHA-256 写入 Golden Run 的
   `sample-files.json`，并把这次动作保存在独立且不可修改的 record-samples Run。
   对 gap queue 每项各执行一次；算子 id 从 capture state 和 Scan Run 核对，不在
   命令或工具里写死。这一步只接受仍为 `ACTIVE`、尚未 self-replay 的 Golden Run；
   失败时不得继续。
5. 在同一次 CUDA Session 内按各自现有 Kernel Call 完成全部算子的 self-replay。
   每个 `golden_run` 都通过才把 Session 标为 `SEALED`。replay config、worker
   result、Golden state 和 wrapper result 必须携带各自
   `sample-files.json` SHA-256；后续 build 会重新计算并要求它们与当前样本字节
   完全一致。
6. Agent 先生成临时 Spec，把其中的 Working State 写成下一状态：
   `WAITING / HANDOFF`，保留第一个 `active_operator`，把每个 gap queue 行的
   `golden_run` 写成对应 SEALED Run，并写入 bundle/manifest 路径；唯一
   `next_action` 是人工复制后在 P800 校验。`handoff_bundle.py` 不解析 Working State，
   所以这些字段必须由 Agent 逐项核对。
7. 用临时 Spec 构建 bundle。build 必须接收当前 Scan Run，并为每项重复传入一个
   `--golden-run` 和一个 `--sample-record-result`。工具从 Scan Run 的
   `gap_queue` 得到唯一有序算子集合，拒绝缺失、额外或重复证据；不要在 runbook
   中维护另一份固定算子列表。全部 Golden 还必须引用同一个 Session 配置、来自
   同一个采集进程，并对应 Session 中按队列排列的目录。Manifest v2 包含原始
   `scan-result.json`、原始 Session 配置、全部 Golden、每项 record
   result/sidecar 摘要和完整文件清单。revision 6 未提供 `--scan-result` 时必须
   停止，不能回退到历史 Manifest v1。
8. 在 CUDA 端立即执行 `--mode verify`。verify 只信任包内 Scan Run，重新得到期望
   算子集合，并用包内 Session 配置交叉核对全部 Golden。只有本地 Agent 的正式
   样本审查、build、verify 和 Contract 绑定全部通过后，才用 bundle 中
   `migration-spec.md` 的相同字节原子替换当前 Spec，并做采集后的唯一一次 GitHub
   回传；不得在 manifest 生成后再次编辑该 Spec。

任何步骤失败都要保留本地证据，不替换当前 Spec；随后把失败证据追加到同一
GitHub evidence 分支。Session 已消耗但不能形成可信 Golden 时进入
`BLOCKED / CUDA_CAPTURE`，不能重开第二次 Session。

### `HANDOFF`

工具只生成和校验 bundle；人工负责跨机器复制。

P800 收到 bundle 后，先校验 manifest。文件缺失、多出、大小或 SHA-256 不符时，
不得读取 Golden Tensor。校验通过后进入 `ACTIVE / P800_REPAIR`，下一动作是活动
kernel 的 baseline replay。

### `P800_REPAIR`

具体修复方案由 Migration Agent 根据当前 baseline、失败样本元数据和目标源码自主
提出。人只提供 P800 环境、仓库和必要权限，不需要提供 attempt 假设、补丁或逐条
修复命令；确定性脚本也不能选择代码改法。

P800 目标仓库由环境变量 `SGLANG_KUNLUN_WORKTREE` 指向。Agent 必须确认它是
Contract 固定 revision 的 Git 根目录。变量缺失或路径无法确认时，才进入
`NEEDS_HUMAN` 询问一个环境问题；不能猜路径。

在 P800 启动服务、baseline replay 或 candidate replay 前，必须先固定 Kunlun
导入环境：

```bash
export SGLANG_PLATFORM=kunlun
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export PYTHONPATH="$SGLANG_KUNLUN_WORKTREE/python:$SGLANG_KUNLUN_WORKTREE/sglang-kunlun${PYTHONPATH:+:$PYTHONPATH}"
```

完整候选变量表位于 `docs/p800-environment-and-repair.md`。Agent 必须完整读取该表，
再根据 D/P 节点类型、DeepEP/BKCL 拓扑、实际 backend 和当前活动算子按需选择；
除上述三个导入约束外，不能整表导出。每个额外设置的变量都要在当前 P800 环境
Run 中记录最终值和选择原因。需要 D/P 专属值但节点类型无法确认时进入
`NEEDS_HUMAN`，不能猜值。环境、插件导入或设备发现失败是工具/环境失败，不是
Operator Gap。

调用 `replay_compare.py --mode kernel-replay` 时，对每个实际导出的候选变量追加
`--p800-environment-reason '变量名=选择原因'`。工具自动读取真实值并写入 Run；
没有选择任何候选变量时不传该参数。缺少原因时先补齐环境证据，不能执行 replay。

先用 `workspace_guard.py` 检查固定 revision。第一个算子要求干净工作区；后续算子
允许工作区恰好等于上一项 `passing_run` 的完整通过 patch。把该 Run 作为
`--accepted-result` 传入，工具必须逐字节核对；除此之外的已有修改进入
`NEEDS_HUMAN`，不得替用户清理。

baseline 直接用 Golden Sample 的输入、直接参数和非 Tensor 参数调用活动 kernel
边界：

- 全部样本通过：该行记录 `repair_status: PASS` 和 baseline Run，说明 Kunlun
  已有等价行为；不伪造失败，也不消耗修复次数。该行 `passing_run` 沿用之前
  PASS 行中最近一份非空值；如果此前没有补丁则保持 `null`，然后自动选择下一项。
- 任一样本执行或精度失败：记录 baseline，`attempts_used` 保持 0，开始有限修复。

baseline 前先调用 `workspace_guard.py --mode check-baseline --operator-id
<active_operator>`；replay 完成后调用 `--mode assess-baseline --operator-id
<active_operator> --replay-result <baseline-replay/result.json>`。只要此前 PASS 行中
存在非空 `passing_run`，两条命令都追加
`--accepted-result <最近非空 passing_run/result.json>`；不能把没有
`candidate.patch` 的 baseline PASS Run 传给该参数。两次都要显式给出检查所覆盖的
文件，但 baseline 的文件列表不预先锁定后续 attempt 的实现位置。工具从 Contract
读取固定 Kunlun revision 和五轮上限，不允许命令行覆盖。
正式 Spec 只接受
`replay_compare.py --mode kernel-replay --execution-site p800` 的结果；测试用
synthetic 结果不能推进正式流程。

每轮修复：

1. Agent 先重读上一结果、失败样本元数据和相关源码，自主选择一个可证伪假设及
   该假设需要的最小文件集合。不同 attempt 可以选择不同文件；第二项以后，声明
   的允许文件还要包含已接受 patch 中的修改文件，使工具能校验完整累计 patch。
   工具只保护声明的文件并拒绝其他改动。不得要求人替 Agent 指定改法。
2. Agent 根据源码追踪 P800 的原始 Kernel Call 边界，再决定修复位置和重放方式。
   流程不预设某个 Python 函数名、函数签名或 attempt 改法。简单改动直接内联在
   原生产函数的既有调用位置，不得为了让 replay import 而新增生产 helper。
   领取 attempt 前，Agent 必须确认 model-adaptation 已有或补齐该 Kernel Call 的
   replay 参数装配，并把源码位置、命令和测试封存在 adapter Run。参数装配只能
   调用原有 P800 Kernel Call，不能复制一份修复算法。adapter 必须验证实际
   `x/y`、直接参数和非 Tensor 参数的数据来源；仅检查同名关键字、常量或不可达
   调用不能作为候选生效证据。
3. 从同一基线开始，只写一个假设；调用
   `workspace_guard.py --mode start-attempt --operator-id <active_operator>
   --attempt N --hypothesis <一句话> --previous-result <上一结果>` 创建 Run 后，
   Agent 才能修改源码。此前存在累计 patch 时同时传
   `--accepted-result <最近非空 passing_run/result.json>`。每个算子的 attempt 1
   都从自己的失败 baseline 开始；同一算子的后续轮必须指向紧邻的上一轮失败结果，
   不能重用编号绕过五轮上限。还要为此前所有 `repair_status: PASS` 且有 Golden
   的行逐个追加 `--regression-operator-id <operator_id>`；这份列表在领取 attempt
   时封存，后续轮不得删减。工具会从 `--accepted-result` 继承已经封存的完整
   历史列表并拒绝缺项，不能只回归最近一份 patch 的所属算子。
4. 修改前把 Working State 的 `attempts_used` 加一；通过轮也计数。
5. 修改只限活动 kernel 调用的 Python、P800 可执行 Torch、已有
   xspeedgate/kunlun_ops 能力和聚焦测试。
6. 修改完成后，先调用 `workspace_guard.py --mode record-candidate` 保存相对固定
   Git revision 的完整累计 `candidate.patch`。然后运行
   `replay_compare.py --mode kernel-replay --execution-site p800
   --candidate-result <attempt-run>/candidate-result.json`，仍以 Scan 记录的原
   P800 Kernel Call 为 `invocation_target`，对活动算子的全部 Golden Samples
   使用固定 `torch.testing.assert_close`。候选 replay 启动前和结束后都必须确认
   当前工作区的完整累计 diff 与 `candidate.patch` 逐字节一致；该算子的 adapter
   必须从被修改后的原生产调用点取得实际参数装配，不能仅以“存在 candidate”作为
   修复生效开关。不得用新生产 wrapper 充当回放入口。
7. 调用 `workspace_guard.py --mode finish-attempt
   --replay-result <attempt-run>/replay/result.json`。若活动 replay 通过，先用同一
   `candidate-result.json` 逐项重放第 3 步封存的历史 operator，把结果写在
   `<attempt-run>/regressions/<operator>/result.json`，并为每项向
   `finish-attempt` 追加一个 `--regression-result`。
   `finish-attempt` 要求工作区仍与已记录 patch 完全一致，并先把 patch 摘要和 replay
   结果摘要、每个历史回归的结果与摘要共同写入 `outcome.json`。活动算子或任一
   历史算子失败，本轮整体都失败并恢复上一份累计 patch。Agent 只有在源码与运行
   日志都能说明重放经过本轮修复边界后，才能接受该结果；否则先修正重放方式，不能
   伪造 PASS。
   失败时只恢复到 `--accepted-result` 指向的上一份通过 patch；通过时保留新的
   累计 patch，并确认它是唯一工作区修改。未知或 staged 修改一律停止且不清理。

需要新增 C++/自定义 kernel/底层注册、改完整模型、放宽精度门槛，或第五轮仍失败，
都进入 `BLOCKED`，不能扩大范围。

`finish-attempt` 已把当前算子和所有先前 `repair_status: PASS` 且有 Golden 的
算子作为同一通过门槛封存。整体通过后：

1. 封存当前 Run，把活动行更新为 `repair_status: PASS`，记录
   `attempts_used`；修复 attempt 通过时 `passing_run` 指向本轮，baseline 直接
   PASS 时沿用之前最近一份非空值；
2. 若存在下一行 `repair_status: PENDING` 且 `golden_run` 为 `SEALED`，按 gap
   queue 表格顺序自动选择下一项，更新 `active_operator`，把该行改为 `ACTIVE`，
   清空当前假设，并把唯一 `next_action` 写成该算子的 P800 baseline；
3. 立即重读 Spec 并继续，不等待人工确认；
4. 若仍有 `PENDING` 但缺少 SEALED Golden，说明唯一 CUDA Session 或 Handoff
   不完整，写入 `BLOCKED`，不得临时重访 CUDA；
5. 只有全部 gap queue 行都是 `PASS`，才写 `PASS / DONE`、
   `active_operator: null` 和 `next_action: none`。若至少发生过一次修复，最后一份
   非空 `passing_run` 必须保存完整累计 patch，且该 patch 是唯一工作区修改；若
   全部 baseline 直接 PASS，则 `passing_run` 可以全部为 `null`，但工作区必须是
   Contract 固定 revision 的干净状态。

同一 P800 Agent 会话中，只要状态仍为 `ACTIVE`，就按上述规则继续当前轮或自动选择
下一项。只有达到全队列 `PASS / DONE`、`BLOCKED`、`NEEDS_HUMAN`，或确实需要计划内
跨机器动作时才停止。

新选中的活动行以 `attempts_used: 0 / active_hypothesis: null` 开始；其中
`active_hypothesis: null` 是人工确认的合法交接状态，不是让人预填修复方案。
Migration Agent 在领取该算子 attempt 1 的同一动作里自主写入单一假设。

## 4. 接受工具结果

这些确定性脚本至少接收：

```text
--spec <migration-spec.md> --run-dir <fresh-run-directory>
```

调用前确认脚本和当前活动算子的 adapter 存在。缺少能力时准确报告代码缺口，不把
工具缺失写成 Operator Gap。不得用命令行覆盖 Contract 的容差、shape 或修复次数。

接受结果前重新计算 Contract Data SHA-256，核对 `spec_id`、
`contract_revision`、`contract_data_sha256`。任一不一致都不得推进状态。

## 5. 停止条件

- `ACTIVE`：重读 Spec 后继续唯一下一动作。
- `WAITING`：报告 bundle、manifest 和人工复制动作。
- `PASS`：只用于全部 gap queue 已关闭；报告逐算子结果和最终累计 patch。
- `BLOCKED`：报告停止原因和相关 Run。
- `NEEDS_HUMAN`：报告唯一问题和恢复所需证据。

不自动 SSH、上传、下载、登录、管理凭证、创建分支、commit 或 push。

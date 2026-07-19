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
`next_action`，直到状态到达 `WAITING`、`PASS`、`BLOCKED` 或
`NEEDS_HUMAN`。

`migration-spec.md` 是恢复执行的唯一状态来源：

- `Contract` 保存人类批准的扫描范围、运行参数、样本规则、精度门槛和权限边界；
  批准后本 Skill 不得修改。
- `Working State` 保存当前阶段、gap queue、扫描完成后选出的
  `active_operator`、Run 指针和唯一下一动作。
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

#### 扫描后选择 Demo

先完成并封存整个 Scan Run，再从它的 gap queue 比较候选。至少比较
`topk_sigmoid`、视觉 attention 和其他真实缺口。依次优先：

1. 固定输入下必达；
2. 边界输入、输出和直接参数更少；
3. 不依赖 TP 通信、KV cache 或大型运行时对象；
4. 可以用 P800 Torch 或已有 xspeedgate/kunlun_ops 在 Repair Boundary 内修复。

只把一个最小候选写入 Scan Run 的 `selection.active_operator` 和
`capture_plan`，然后把同一个值写入 Working State 的 `active_operator`。Contract
保持不变。静态缺口只能写 `STATIC_GAP_CANDIDATE`；P800 baseline 失败后才能称为
实机缺口。

完整结果写入新的 `runs/scan-NNN/result.json`。旧 Run 不可修改。

### `CUDA_CAPTURE`

先检查 Scan Run 的 `capture_plan.adapter_status`。Scan Run 不可改写；如果其中是
`NOT_IMPLEMENTED`，只有 Working State 的 Demo Closure Evidence 已把当前
`active_operator` 的 adapter 标为 `PASS`，且所指 Run 绑定当前 Contract、
`active_operator`、当前源码摘要和 `passed: true`，才表示后续会话可以继续。
`last_completed_action` 和 `last_run` 随后可以推进到其他动作，不能因此丢失已经
封存的 adapter closure evidence。

如果 adapter 尚未实现：

1. 把实现限制在现有 `capture_golden.py`、`replay_compare.py` 和采集插件中；
2. Hook Scan Run 指定的现有 capture seam，不新增自定义算子函数；
3. 用测试证明只保存 Contract 允许的输入、直接参数、非 Tensor 参数和 CUDA 输出，
   最多三个 shape，且 rank 0 以外不落盘；
4. 用测试证明 CUDA self-replay 调 Scan Run 记录的 CUDA capture seam，P800
   compare 调 Scan Run 记录的对应 Kunlun seam；两端都只使用已有调用，不新增模型
   helper 或自定义算子；
5. 把实现与测试证据写入新的 Run，不改写 Scan Run；随后更新 Working State 的
   Demo Closure Evidence、`last_completed_action`、`last_run` 和唯一 preflight
   下一动作。

当前仓库中的 `Step3p5MLP.forward` capture/replay adapter 只属于 revision 4
历史方案，不能消费 revision 5。缺少当前活动 kernel 的 capture/replay adapter
时，准确报告缺口并执行对应实现 ticket；不要运行旧 MLP preflight，也不要消耗
CUDA Session。

adapter 完成后，完整读取
[references/cuda-capture-validation.md](references/cuda-capture-validation.md)，
先执行不消耗正式 Session 的 preflight。preflight 必须验证：

- Scan Run 和当前 Contract 绑定；
- Hook 的是 Scan Run 记录的现有调用；
- rank 0 和最多三个 shape 的去重；
- 直接参数可保存，但完整 checkpoint 和 module state 会被拒绝；
- 相同样本可以按固定 Precision Gate self-replay。

如果 adapter Run 记录的任一源码 SHA 在最近一次 PASS preflight 后发生变化，旧
preflight 只能作为历史证据；Demo Closure Evidence 必须把当前源码 preflight 标为
`PENDING`，且不得开始正式 CUDA Session。先按 runbook 生成新的 preflight Run。

如果 Working State 的 `execution_site` 是 `CUDA`，但当前会话不在用户指定的 CUDA
机器，不创建 preflight Run，也不运行合成替代品。只报告 runbook 中的命令和
GitHub evidence 分支回传要求，保持 Working State 不变并停止本次执行。

preflight 通过且唯一下一动作已进入正式 Session 时，完整读取
[references/formal-cuda-capture.md](references/formal-cuda-capture.md)。跨机器由
GitHub 回传时分成两段：

1. 人在 CUDA 机运行唯一真实模型 Session，完成 `record-samples` 和 CUDA
   self-replay，回传 Session Run、Golden Run 和 record-samples Run 后停止；
2. Agent 核验真实样本与自回放证据，生成包含 `WAITING / HANDOFF` Working State
   的临时 Spec；人再回到同一 CUDA 环境 build + verify bundle。第二段不得重启
   模型，不是第二个 Capture Session。

正式模型启动前，必须先把 `session-start.json` 和 prepare 证据提交到 runbook
固定的 GitHub evidence 分支。只有首次 push 成功才消耗并授权该 Session；成功或
失败都只向同一分支追加，不能换分支或 Run 名重试。

不得在真实 Golden 回传前预填 shape 数量或伪造临时 Spec，也不得让人自行修改
Working State。

正式 CUDA Session 只采集活动算子：

1. 启动固定 checkpoint 的真实 TP8/BF16/eager 模型；采集只增加插件配置，不改变
   两端模型启动参数。
2. 每种真实 shape 最多保存一份 rank 0 样本。样本包含活动调用的输入、直接参数、
   必要标量和 CUDA 期望输出。
3. 停止采集进程后、CUDA self-replay 前，调用 `handoff_bundle.py
   --mode record-samples`，把每个 Golden Sample 文件的 shape、相对路径、大小和
   SHA-256 写入 Golden Run 的 `sample-files.json`，并把这次动作保存在独立且不可
   修改的 record-samples Run。这一步只接受仍为 `ACTIVE`、尚未 self-replay 的
   Golden Run；失败时不得继续。
4. 在同一次 CUDA Session 内用现有 kernel 接口完成全部样本 self-replay。全部
   通过才把 Golden Run 标为 `SEALED`。replay config、worker result、Golden
   state 和 wrapper result 必须携带同一个 `sample-files.json` SHA-256；后续
   build 会重新计算并要求它们与当前样本字节完全一致。
5. Agent 先生成临时 Spec，把其中的 Working State 写成下一状态：
   `WAITING / HANDOFF`，保留同一个 `active_operator`，写入本次
   `golden_run`、bundle/manifest 路径，并把唯一 `next_action` 写成人工复制后在
   P800 校验。`handoff_bundle.py` 按权限边界不解析 Working State，所以这些字段
   必须由 Agent 逐项核对。
6. 用临时 Spec 构建 bundle；build 必须显式接收 record-samples Run 的
   `result.json`，核对它记录的 sidecar SHA，并把 record result 与 sidecar 的
   SHA-256 写入 manifest。随后在 CUDA 端立即 verify。只有 build、verify 和
   Contract 绑定全部通过后，才用 bundle 中 `migration-spec.md` 的相同字节原子
   替换当前 Spec；不得在 manifest 生成后再次编辑该 Spec。

任何步骤失败都要保留证据。Session 已消耗但不能形成可信 Golden 时进入
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

先用 `workspace_guard.py` 检查固定 revision 和干净工作区。无法解释的已有修改
进入 `NEEDS_HUMAN`，不得替用户清理。

baseline 直接用 Golden Sample 的输入、直接参数和非 Tensor 参数调用活动 kernel
边界：

- 全部样本通过：记录它不是由 P800 baseline 证实的 `Operator Gap`，并进入
  `BLOCKED`；不得伪造失败
  或再次访问 CUDA。
- 任一样本执行或精度失败：记录 baseline，`attempts_used` 保持 0，开始有限修复。

baseline 前先调用 `workspace_guard.py --mode check-baseline`；replay 完成后调用
`--mode assess-baseline --replay-result <baseline-replay/result.json>`。两次都要
显式给出检查所覆盖的文件，但 baseline 的文件列表不预先锁定后续 attempt 的实现
位置。工具从 Contract 读取固定 Kunlun revision 和五轮上限，不允许命令行覆盖。
正式 Spec 只接受
`replay_compare.py --mode kernel-replay --execution-site p800` 的结果；测试用
synthetic 结果不能推进正式流程。

每轮修复：

1. Agent 先重读上一结果、失败样本元数据和相关源码，自主选择一个可证伪假设及
   该假设需要的最小文件集合。不同 attempt 可以选择不同文件；工具只保护本轮声明
   的文件并拒绝其他改动。不得要求人替 Agent 指定改法。
2. Agent 根据源码追踪 P800 的原始 Kernel Call 边界，再决定修复位置和重放方式。
   流程不预设某个 Python 函数名、函数签名或 attempt 1 改法。当前仓库只有
   `kunlun_ops.swiglu` baseline adapter，它只证明缺口，不能封存任意候选补丁的
   PASS。领取 attempt 前，Agent 必须先在 model-adaptation 流程代码中补齐并测试
   该边界的 repair replay adapter，使 `replay_compare.py` 和
   `workspace_guard.py` 能生成并接受经过真实修复路径的结果，再把源码位置、命令
   和日志封存在新的 adapter Run。这个准备动作不修改 SGLang-Kunlun、不计
   repair attempt。重放代码可以做参数装配，但不得在 SGLang-Kunlun 中新增只供
   重放使用的 helper、自定义算子或另一份修复逻辑。
3. 从同一基线开始，只写一个假设；调用
   `workspace_guard.py --mode start-attempt --attempt N --hypothesis <一句话>
   --previous-result <上一结果>` 创建 Run 后，Agent 才能修改源码。attempt 1 的
   上一结果是失败的 baseline assessment；后续轮必须指向紧邻的上一轮失败结果。
   每个编号只能使用一次，不能重新从 attempt 1 开始绕过五轮上限。
4. 修改前把 Working State 的 `attempts_used` 加一；通过轮也计数。
5. 修改只限活动 kernel 调用的 Python、P800 可执行 Torch、已有
   xspeedgate/kunlun_ops 能力和聚焦测试。
6. 修改完成后，先调用 `workspace_guard.py --mode record-candidate` 保存相对固定
   基线的完整 `candidate.patch`；然后把本轮原始 Kernel Call 边界的重放 Run 放在
   `<attempt-run>/replay`，对全部 Golden Samples 使用固定
   `torch.testing.assert_close` 门槛。不能因为 standalone baseline adapter
   容易调用，就把它当作候选补丁已生效的证据。
7. 调用 `workspace_guard.py --mode finish-attempt
   --replay-result <attempt-run>/replay/result.json`。
   `finish-attempt` 要求工作区仍与已记录 patch 完全一致，并先把 patch 摘要和 replay
   结果摘要共同写入 `outcome.json`。Agent 只有在源码与运行日志都能说明重放经过
   本轮修复边界后，才能接受该结果；否则先修正重放方式，不能伪造 PASS。失败时
   才恢复已声明的候选文件；通过时保留 patch，并确认它是唯一工作区修改。未知或
   staged 修改一律停止且不清理。

需要新增 C++/自定义 kernel/底层注册、改完整模型、放宽精度门槛，或第五轮仍失败，
都进入 `BLOCKED`，不能扩大范围。

同一 P800 Agent 会话中，只要状态仍为 `ACTIVE`，就按上述规则继续下一轮，不在每轮
之间等待人选择补丁。只有达到 `PASS`、`BLOCKED`、`NEEDS_HUMAN`，或确实需要计划内
跨机器动作时才停止。

当前 baseline 已失败、`attempts_used: 0` 且 `active_hypothesis: null` 是人工确认
的交接状态，不是 Spec 冲突。Agent 在领取 attempt 的同一动作里先选择单一假设，
再把 `attempts_used` 和 `active_hypothesis` 写入 Working State；不能在 baseline
验收时由其他人预填假设。

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
- `PASS`：报告活动算子、样本数、passing Run 和最终 patch。
- `BLOCKED`：报告停止原因和相关 Run。
- `NEEDS_HUMAN`：报告唯一问题和恢复所需证据。

不自动 SSH、上传、下载、登录、管理凭证、创建分支、commit 或 push。

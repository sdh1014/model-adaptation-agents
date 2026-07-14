# 决定单 Agent 入口与最小脚本表面

Type: grilling
Status: resolved
Blocked by: 06

## Question

端到端纸面 Demo 已证明每个阶段需要哪些判断和确定性动作。为了保持“更多依靠模型能力、工具足够简洁”，用户应通过一个自然语言 Skill、一个顶层 CLI，还是“Skill 作为唯一入口并在内部调用少量脚本”来启动和恢复 Migration Agent？同时确定 capture、bundle 校验、P800 replay/compare 和基线保护所需的最小脚本边界，避免把 Agent 的语义判断重新写成工作流代码。

## Comments

- 用户确认：一个自然语言 Skill 是唯一的人类入口；首次启动、CUDA 执行、P800 恢复和上下文压缩后继续都调用同一个 Skill。Skill 每次先读取 Migration Spec，再按唯一 `next_action` 行动；不增加顶层编排 CLI。
- 用户确认四个 Deterministic Tool 边界：`capture_golden.py` 负责 CUDA 打桩与去重采集，`replay_compare.py` 负责 CUDA 自回放和 P800 replay/compare，`handoff_bundle.py` 负责 manifest 生成与两端校验，`workspace_guard.py` 负责固定基线、完整补丁和失败恢复保护。Demo 不新增 `scan.py`，源码扫描和语义判断由 Migration Agent 完成。
- 用户确认统一工具接口：Agent 为当前动作创建 Run，并向脚本传入 `--spec migration-spec.md --run-dir runs/<run-id>`；脚本只读取所需 Contract 字段，不读写 Working State，只向该 Run 写结构化结果与日志。退出码 `0` 表示成功形成可信 `result.json`，业务上的 PASS/FAIL 写入结果文件；非零只表示工具未能完成，不能直接判为 Operator Gap。
- 用户进一步确认：`migration-spec.md` 的 Contract 内含一个专门的 JSON 结构化区块。`--spec` 是所有脚本的必选只读参数，但脚本只解析该 JSON，不解析 Markdown 或 Working State；当前 Golden Run、执行模式等动态信息继续通过脚本专有参数传入。固定容差和上限不提供命令行覆盖。结果记录 `spec_id`、`contract_revision` 和规范化 Contract Data 的 SHA-256，不绑定会持续变化的整份 Spec 摘要。
- 用户确认首次启动：如果 `migration-spec.md` 不存在，同一个 Skill 生成骨架并尽量填写 Contract；存在占位时写为 `NEEDS_HUMAN / SCAN` 并停止。人类补齐并批准后记录 `contract_revision: 1` 和初始 Human Decision，再次调用同一 Skill 校验并进入 `ACTIVE / SCAN`；不增加 `init` 命令。

## Answer

唯一的人类入口是一个自然语言 Skill，不提供顶层编排 CLI。首次启动、CUDA 采集、人工交接后的 P800 恢复，以及上下文压缩后的继续执行，都调用同一个 Skill；它每次先完整读取 Migration Spec，只执行唯一 `next_action`。如果 Spec 不存在，Skill 生成骨架并尽量填写 Contract；存在未决字段时进入 `NEEDS_HUMAN / SCAN`，获得人工初次批准并记录 `contract_revision: 1` 后才进入 `ACTIVE / SCAN`。

Demo 只实现四个 Deterministic Tool：

| 脚本 | 唯一职责 |
|---|---|
| `capture_golden.py` | CUDA 打桩、签名去重，并为一个算子最多保存三个真实样本 |
| `replay_compare.py` | CUDA 新进程 self-replay，以及 P800 baseline/repair replay 和 `torch.testing.assert_close` 比较 |
| `handoff_bundle.py` | 生成 manifest，并在 CUDA、P800 两端校验交接包 |
| `workspace_guard.py` | 校验固定干净基线、保存完整候选补丁、失败后恢复基线，并确认通过补丁是唯一修改 |

不新增 `scan.py`：Migration Agent 使用现有源码搜索能力完成调用链和 Semantic Operator 判断，避免把模型判断固化成脆弱规则。脚本也不选择算子、假设、状态迁移或容差。

四个脚本共享必选参数 `--spec migration-spec.md --run-dir runs/<run-id>`，并可按职责接收最少的动作参数，例如 Golden Run 路径或 replay 模式。Spec 的 Contract 内包含专门的 JSON 结构化区块；脚本只从该区块读取自己需要的固定值，不解析其余 Markdown，不读写 Working State，也不提供 `--atol`、`--rtol` 等覆盖固定 Contract 的参数。这样 `--spec` 用于锁定人工确认的约束，而不是让脚本根据 Spec 决定工作流。

每个可信 `result.json` 都记录 `spec_id`、`contract_revision` 和规范化 Contract Data 的 SHA-256；不记录整份 Spec 的绑定摘要，因为 Working State 会在动作后变化。所有结构化结果和日志写入当前 Run。退出码 `0` 只表示成功形成可信 `result.json`，精度或校验是否通过由结果文件表达；非零表示工具自身未完成，不能直接判为 Operator Gap。Migration Agent 读取结果、作出语义判断，再立即更新 Working State。

Skill 包内部只需 `SKILL.md + scripts/`；运行工作区仍只有 `migration-spec.md + runs/`。这保持了模型负责判断、脚本负责确定性动作的职责边界，也没有引入新的领域词。

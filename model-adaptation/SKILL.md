---
name: model-adaptation
description: 在 KLX P800 上验证 Step-3.7-Flash 算子、自动修复运行报错并跑通 eager 服务。
argument-hint: "Step-3.7-Flash"
disable-model-invocation: true
---

# 输入和目标

只接受精确调用：

```text
$model-adaptation Step-3.7-Flash
```

参数缺失或不同就说明正确调用并停止，不猜测其他模型。

本 Skill 只完成一个闭环：

```text
PREFLIGHT -> OPERATOR_VERIFICATION -> EAGER_BRINGUP -> DONE
                  ^                         |
                  +--- 新算子问题 ----------+
```

失败留在当前 phase，进入 BUG Repair Loop；修复后从原失败点继续。只有两个正常退出：

- `PASS / DONE`：固定 P800 环境、全部算子和固定 eager 请求通过；
- `BLOCKED`：同一个 BUG 的 3 次不同、可验证的修复尝试都失败。

这里验证的是“服务能够稳定启动并正常回答”，不执行整模型 CPU reference，不宣称完成
整模型精度验证。

# 唯一状态和按需资料

每次动作前完整读取工作区根目录的 `migration-spec.md`。其中 Contract 固定模型、源码、
checkpoint、运行参数、容差和修复边界；Working State 保存唯一当前状态和
`next_action`。不要从旧 Run 推断当前进度。

只在对应阶段读取以下资料：

- P800 环境失败：`docs/p800-environment-and-repair.md`；
- 算子分类和测试用例：`docs/step3p7-p800-operator-gap-analysis.md`；
- 第一次真实失败：完整读取 `references/bug-repair-loop.md`；
- BUG 历史：追加到 `docs/step3p7-p800-bug-log.md`。

本流程不使用通用 Tensor capture、pickle/JSON replay、handoff bundle 或统一 adapter。
局部测试直接调用固定源码的 P800 生产入口。

# 不可放宽的规则

- 不修改 Contract 固定的 commit、checkpoint、TP8、BF16、eager 请求和精度门槛。
- 不用 test-only helper 代替真实生产调用。
- 不把导入、设备、依赖或路由错误记成算子缺失。
- 已有精确 PyTorch/`forward_native` 语义时标为 `NATIVE_IMPLEMENTATION`；实现存在与
  P800 生产路由是否可达分开记录。
- 浮点结果使用 Contract 的 `torch.testing.assert_close`；整数和布尔结果精确相等；
  同时检查结构、shape、声明 dtype、有限值，以及生产接口要求的 stride、alias、输出
  buffer 和原地语义。
- 允许自主修改 Repair Scope 内的 Python、P800 PyTorch 和已有 Kunlun 算子装配；不
  新增 C++、自定义 kernel 或底层注册。
- 不自动登录远端、取得凭证、创建分支、commit 或 push；只使用用户已经提供的 P800
  会话和工作树。

# 执行流程

## 1. PREFLIGHT

在判断算子前验证并记录：

1. SGLang、SGLang-Kunlun 和 checkpoint 与 Contract 一致；
2. 实际导入路径来自固定 worktree，Kunlun platform 已激活；
3. 8 张目标设备可见，BF16 Tensor 创建、搬运和简单运算正常；
4. decode 和 prefill graph backend 都关闭；
5. 启动所需依赖、模型配置和固定请求资源可读。

全部通过后写 Run Evidence，设置 `environment_status: PASS`，进入
`OPERATOR_VERIFICATION`。任何失败都记录为 `ENVIRONMENT` BUG 并进入修复循环；环境
通过前不得产生 `OPERATOR_MISSING` 结论。

## 2. OPERATOR_VERIFICATION

按 Spec 中 Operator Verification Queue 的顺序处理每一行，包括初始五个历史
Candidate；静态复核或旧结果不能将它们直接设为 `PASS`。

对每项执行：

1. 从固定源码确认模型调用点、语义边界、已有 native/Kunlun 实现和实际 P800 路由；
2. 从固定 native 实现、社区测试或独立公式建立 CPU reference，不能调用待验证实现；
3. 按队列的 `required cases` 生成确定性输入，并直接调用 P800 Production Operator；
4. 使用 Contract Precision Gate 比较全部语义输出；
5. 保存源码锚点、命令、shape、dtype、layout、stride、seed 和逐项结果。

一项只有同时满足以下条件才算“算子可用”：

- `route_state: PRODUCTION_REACHABLE`；
- 所有 required cases 满足数值和接口契约；
- 聚焦测试直接覆盖真实生产入口。

通过后将该项设为 `PASS` 并继续下一项。失败按首个根因分类为 `ADAPTATION`、
`OPERATOR_MISSING`、`OPERATOR_CONTRACT` 或 `DISTRIBUTED_RUNTIME`，进入修复循环。
队列全部通过后进入 `EAGER_BRINGUP`。

## 3. EAGER_BRINGUP

用 Contract 固定的 TP8、BF16、target-only eager 配置启动真实 Step-3.7-Flash，依次
执行固定文本和单图请求。

“回答正常”必须同时满足：

- engine 或服务到达 ready；
- 两个固定请求都成功返回，响应结构有效且生成内容非空；
- 日志没有未处理 traceback，能观察到的数值结果均为有限值；
- 相同确定性请求至少复跑一次仍成功；
- 已验证算子的生产路径在真实模型中可达。

遇到失败，只处理日志中的首个根因：

- 环境或依赖问题回到 `PREFLIGHT`；
- 新的生产算子或新 shape/dtype/layout 问题追加到队列，回到
  `OPERATOR_VERIFICATION`；
- 模型适配或多卡运行问题留在 `EAGER_BRINGUP` 修复。

修复后必须重新执行原失败请求。两个请求都正常后写最终 Run Evidence，一次性设置：

```text
model_status: PASS
status: PASS
phase: DONE
next_action: null
```

然后停止。不要追加整模型 CPU 精度比较或逐层 debugger。

# BUG Repair Loop

任何真实命令、测试或请求失败时，读取 `references/bug-repair-loop.md` 并执行：

1. 分配稳定 `bug_id`，保存可复现失败并追加 Bug Ledger；
2. 每个 attempt 只验证一个有证据的根因假设，做最小范围修改；
3. 先重跑聚焦复现，再重跑最初失败的 P800 命令或请求；
4. 修复成功时记录根因、解决方法和回归证据，关闭 BUG 并继续当前流程；
5. 同一 BUG 第 3 次不同尝试仍失败时，记录所有尝试，设置 `BLOCKED` 并停止。

不得重复相同命令和补丁来凑 attempt；只有证据证明根因不同才创建新 `bug_id`。
普通 Repair Scope 内修复不等待用户选择方案。

`auto_repair_status` 只使用：

- `NOT_EXERCISED`：尚未遇到真实 BUG；
- `ACTIVE`：正在修复；
- `PASS`：至少一个真实 BUG 已完成失败复现、修复和原路径复验；
- `BLOCKED`：当前 BUG 的 3 次修复尝试都失败。

不得主动注入故障来制造 `PASS`。

# Evidence 和状态更新

每次检查或 attempt 都先创建新的 `runs/<run-id>/result.md` 或 `result.json`，至少记录：

- Contract revision、phase、结论和完整命令；
- 固定源码 commit、checkpoint、实际导入路径和环境；
- 算子 reference/生产入口的源码锚点，或 eager 请求与日志；
- 使用的输入特征、Precision Gate、预期与实际结果；
- BUG 的 `bug_id`、attempt 序号、假设、修改文件和验证结果；
- 唯一下一动作。

关闭 BUG 时还要记录确认根因、最终解决方法、聚焦回归、原始 P800 路径复验和残余
风险。随后再更新 Working State；不要覆盖旧 evidence 或旧 attempt。

每次更新后重读 Spec：状态仍为 `ACTIVE` 就立即执行唯一 `next_action`；达到 `PASS /
DONE` 或 `BLOCKED` 才结束本次循环。

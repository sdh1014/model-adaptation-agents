# 实现可恢复的五轮修复闭环

Type: task
Status: resolved
Blocked by: 12

## What to build

在一个小型测试仓库中实现并验证 `workspace_guard.py` 与 `replay_compare.py` 协作的 P800 Repair Loop。每个候选补丁都从同一个干净基线生成，失败补丁不能污染下一轮；下一轮可以读取失败证据，但必须重新形成相对基线的完整补丁。

baseline replay 只用于证明真实执行或精度缺口，不计入最多五轮 repair attempt。进入修复后，每轮只允许一个明确假设；比较完成后，必须先封存 Run、记录补丁和结果，再决定保留或恢复。

## Acceptance criteria

- 开始 baseline 和每轮修复前都会校验允许修改范围；未知改动会停止执行，不能被自动覆盖。
- baseline PASS 会把候选标记为非 correctness gap，且 `attempts_used` 保持为 0。
- baseline FAIL 会记录可信证据并进入修复，第一次候选补丁才计为 attempt 1。
- 每轮只有一个 `active_hypothesis`，候选补丁是相对同一基线的完整补丁。
- 失败 Run 被完整写入后，工作区恢复到基线；通过补丁保留在工作区但不提交。
- 第五轮通过和第五轮失败两种路径都能正确结束；不能出现第六轮。
- 工具不创建分支、不提交、不推送，也不修改 Repair Boundary 之外的文件。
- 有端到端测试覆盖 baseline PASS、若干轮后 PASS、五轮耗尽和检测到未知改动。

## Comments

本票据使用测试仓库验证控制逻辑，不依赖真实 Step-3.7-Flash 缺口。

已实现 `model-adaptation/scripts/workspace_guard.py`：

- `check-baseline` 在 baseline replay 前确认 Contract 固定的 Kunlun revision 和
  干净工作区；
- `assess-baseline` 读取独立的 `replay_compare.py` Run，固定
  `attempts_used: 0` 并区分“不是 gap”与“已观察到 gap”；
- `start-attempt` 在修改前写入唯一假设、轮数、允许文件、固定基线和上一结果摘要；
  每轮编号只能使用一次，后续轮必须接在紧邻的失败结果之后；
- `record-candidate` 在 replay 前保存相对同一基线的完整 `candidate.patch`；
- `finish-attempt` 只接受正式 P800 kernel replay；测试 synthetic 只能绑定测试
  Spec。它会确认 replay 晚于 candidate、工作区仍是同一 patch，并先封存
  `outcome.json`，再按失败恢复或通过保留；
- staged 或未知修改会停止，工具不会替用户清理；第五轮失败标记达到上限，第六轮
  以及重复从第一轮开始都会被拒绝。

11 个端到端测试覆盖 baseline PASS、baseline FAIL 后第二轮通过、第五轮通过、第五轮
失败、第六轮/重新 assessment 后重复第一轮拒绝、candidate 必须先于 replay、
patch/replay 绑定、正式 Spec 拒绝 synthetic 和伪造 kernel replay 元数据、未知修改
保护，以及新增聚焦测试文件进入完整 patch 后的失败恢复。测试只使用
临时 Git 仓库和 synthetic replay，没有运行 P800、没有保存 actual Tensor，也没有
创建分支、commit 或 push。实现证据见 `runs/repair-loop-tool-001/result.json`。

2026-07-19 根据真实 P800 baseline 交接方式补充 Agent 自主修复边界：

- baseline 的检查文件只说明当时工作区干净，不得把该列表当作后续 attempt 的固定
  实现位置；
- 每轮由 Migration Agent 根据上一轮证据和目标源码选择一个假设，并声明该轮所需
  的最小文件；不同轮次允许选择不同文件；
- `workspace_guard.py` 仍逐轮校验路径安全、未知修改、连续 claim、完整 patch 与
  replay 绑定，但不替 Agent 选择代码方案；
- 新端到端测试覆盖 baseline、attempt 1 和 attempt 2 使用不同允许文件集合。更新
  证据见 `runs/repair-loop-tool-002/result.json`。

后续源码审查确认不能把 attempt replay 固定成
`<module>:<function>(x, limit)`：真实 Kunlun 入口接收模型层和 dispatch 数据，
SwiGLU 只是内部调用。该预设已取消。Migration Agent 根据每轮源码决定修复位置和
原始 Kernel Call 重放方式；baseline adapter 不能自动证明候选补丁生效。
`workspace_guard.py` 仍负责本轮允许文件、完整 `candidate.patch`、replay 顺序、
结果封存和失败恢复，不替 Agent 决定代码方案。当前正式验证器仍只绑定 baseline
adapter，因此 P800 Agent 在领取 attempt 前先补齐并封存原始边界的 repair replay
adapter；这一步不修改 SGLang-Kunlun、不计 repair attempt。

## Answer

可恢复五轮修复闭环已经实现。Agent 仍负责假设和 Working State；脚本只保护基线、
封存 patch、消费 replay 结果并执行受限恢复，没有新增工作流引擎。

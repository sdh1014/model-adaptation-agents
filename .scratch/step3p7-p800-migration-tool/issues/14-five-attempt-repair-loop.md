# 实现可恢复的五轮修复闭环

Type: task
Status: ready-for-agent
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

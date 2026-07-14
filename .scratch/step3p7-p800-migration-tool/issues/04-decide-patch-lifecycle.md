# 确定 P800 修复的补丁生命周期

Type: grilling
Status: resolved
Blocked by: 02

## Question

P800 Repair Loop 中，基线代码、每轮单一假设、候选补丁、失败后的代码状态、下一轮修改起点和最终通过补丁之间应是什么关系，才能保证 Contract 限定次数内的自动修复可追溯、失败尝试不污染最终代码，同时不引入复杂分支管理或恢复系统？

## Comments

- 用户确认：每轮修复都从同一个干净基线开始，不在上一轮失败代码上累积修改。下一轮可以读取失败 Run 的证据，但必须相对基线重新生成一份完整候选补丁。
- 用户确认：通过补丁自动保留在 P800 工作区，但保持未提交；工具不创建分支、不提交、不推送。
- 后续端到端走查中，用户把 Contract 的最大修复轮数确定为五轮，并确认通过轮也计数。

## Answer

P800 Repair Loop 只认 Contract 中固定的 SGLang 与 SGLang-Kunlun revisions 为修复基线。开始 baseline replay 和每轮修复前，工具都要确认相关源码与该基线一致且没有预先存在的修改；否则进入 `NEEDS_HUMAN`，不能覆盖未知代码，也不消耗 repair attempt。

baseline replay 只证明真实 Operator Gap，不修改代码，也不计入 Contract 的 `max_repair_attempts: 5`。确认 baseline 失败后，每轮按以下关系执行：

1. Migration Agent 先在 Working State 写入本轮唯一假设、创建新的 Run、递增 `attempts_used` 并让通用 `last_run` 指向本轮 Run，再修改源码；通过轮也计数。
2. 本轮产生一份相对同一基线的完整候选补丁，而不是只保存相对上一轮的增量。
3. Run 保存基线 revisions、假设、完整补丁、修改文件、replay/compare 命令、日志和逐样本结果。
4. 若比较失败，先完成 Run，随后把源码恢复到基线；Working State 只保留通用 `last_run` 和下一条假设。
5. 下一轮可以根据以前的失败证据重写方案，但仍从基线生成一份自包含补丁，不继承失败工作区。

某轮通过全部 Golden Samples 后，该轮完整补丁保存在 `passing_run`，并自动保留为 P800 工作区中唯一的未提交修改。Working State 不再重复保存 `candidate_patch` 或 `last_attempt_run`。工具不创建或切换分支，不 commit、不 push。后续是否提交由人工决定。

这里不要求实现复杂恢复系统，也不把 `git worktree`、临时副本或原地恢复写死为契约。确定性脚本只需保证三个外部可检查的事实：每轮开始时是同一干净基线，失败后回到该基线，通过后工作区只包含通过补丁。若恢复后仍有无法解释的修改，Agent 必须停止为 `NEEDS_HUMAN`，不能猜测哪些文件可以删除。

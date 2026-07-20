# Ticket 25：单算子通过后继续缺口队列

Status: Done

## Goal

让同一次 `model-adaptation` Skill 执行在当前算子通过后，自动选择 gap queue 中
下一个已有 SEALED Golden 的待修复算子，并继续 baseline 与有限修复。只有全部
队列项关闭、明确阻塞、需要人决定，或计划内跨机器交接时才停止。

## Acceptance

- gap queue 每行直接记录 `golden_run`、`repair_status`、`attempts_used` 和
  `passing_run`；不增加编排器或新的顶层 CLI。
- 任一算子通过后，Migration Agent 先封存 Run、更新该行，再按表格顺序选择下一项。
- 下一项已有 Golden 时，`active_operator` 自动切换并执行 baseline，不等待人工确认。
- 仍有待修复项但缺 Golden 时，不伪造自动化；进入唯一一次人工 Handoff 所需状态。
- 工作区允许以前一项通过的完整 patch 作为下一项的已接受起点；失败尝试只能回退
  到这个起点，不能抹掉已通过修复。
- 后续 attempt 在领取时封存此前全部通过 operator；活动 replay 通过后，
  `finish-attempt` 必须校验完整回归列表。任一旧算子失败时，本轮不能封存 PASS。
- 某行 baseline 直接 PASS 时不产生 patch；该行沿用此前最近一份非空
  `passing_run`，若此前没有 patch 则保持 `null`。下一项不能把 baseline Run
  误传给 `--accepted-result`。
- 只有所有队列项均完成时写 `PASS / DONE`。

## Blocked by

- None

## Comments

- CUDA 与 P800 不在同一机器。要实现无中断的 P800 队列循环，正式 CUDA Session
  必须一次收齐计划修复项的 Golden Sample。
- SOURCE 实现证据见 `runs/operator-queue-tool-002`；第一版
  `runs/operator-queue-tool-001` 保留为历史。revision 6 实际 Scan、
  多算子 adapter、单进程多 collector、全量 bundle、CUDA Capture 和 P800 执行
  仍按 `migration-spec.md` 的唯一 `next_action` 分阶段完成，不属于本 ticket 的
  硬件 PASS。

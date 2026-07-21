# BUG Repair Loop

只在 P800 检查、算子测试或 eager 请求真实失败时读取本文件。一个 BUG 最多允许 3 次
不同、可验证的修复尝试。记录格式以
`docs/step3p7-p800-bug-log.md` 为准。

## 1. 固定失败

为首个根因分配 `BUG-<PHASE>-<NNN>`，在 Bug Ledger 建立 `OPEN` 记录，并把
`auto_repair_status` 设为 `ACTIVE`。保存完整命令、返回码、最短有效 traceback、实际
环境、固定源码锚点和当前 diff。

建立最小但真实的红色复现：算子问题直接调用 P800 Production Operator；多卡问题保留
必要 rank/collective；模型问题保留原始请求作为最终回归。环境或生产路由尚未确认时，
不能把症状写成算子缺失。

## 2. 尝试修复

`attempt_id` 从 1 递增到 3。每次 attempt 必须在修改前写明：

- 一个根因假设及支持证据；
- 预计修改的最小生产代码面；
- 聚焦复现和原始 P800 路径的验证命令。

只修改 Contract Repair Scope 内的代码。执行后立即记录 diff 和结果：

- 聚焦复现仍失败：该 attempt 为 `FAILED`，下一次必须使用新增证据和不同假设；
- 聚焦复现通过：继续运行相邻回归和原始 P800 路径；
- 证据证明是另一个独立根因：新建 `bug_id`，不能借此清零原 BUG 的失败次数；
- 没有可行的范围内补丁：记录为什么当前假设失败，继续寻找下一种范围内方案。

不得原样重复命令、补丁或假设来凑满次数，也不得在一个 attempt 中堆叠多个猜测。

## 3. 关闭或阻塞

聚焦复现、相邻回归和最初失败的 P800 命令或请求全部通过后：

1. 将 BUG 改为 `RESOLVED`；
2. 记录确认根因、最终解决方法、修改文件、验证命令和残余风险；
3. 将 `auto_repair_status` 设为 `PASS`；
4. 清空 `active_bug` 和 `active_attempt`，继续原 phase 的下一动作。

若第 3 次不同 attempt 后原始失败仍存在：

1. 将 BUG 和 `auto_repair_status` 设为 `BLOCKED`；
2. 保存三次尝试、尚未满足的条件，以及继续所需的具体信息或能力；
3. 保持当前 phase，设置 `status: BLOCKED` 并停止。

没有真实 BUG 时保持 `NOT_EXERCISED`；不能仅凭存在本流程就宣称自动修复能力已通过。

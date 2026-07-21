# KLX P800 Run-Driven Model Adaptation

本项目用一个 Migration Agent 把固定版本的 Step-3.7-Flash 在 KLX P800 上以 eager
模式跑起来。主线只有环境检查、算子验证、真实模型运行和遇错修复。

## Language

**Migration Spec**：
唯一执行状态。Contract 保存人固定的目标和边界，Working State 保存 Agent 当前进度。

**Contract**：
固定模型、源码、checkpoint、TP、dtype、请求、算子精度门槛、修复范围和每个 BUG 的
最大尝试次数。Agent 只能读取，不能在运行中自行放宽。

**Working State**：
保存当前 phase、算子队列、模型状态、活动 BUG、证据路径与唯一 `next_action`。

**Eager Mode**：
decode 和 prefill graph backend 都关闭的 SGLang 执行方式。

**P800 Preflight**：
算子判断前检查固定 worktree、导入路径、Kunlun platform、设备、checkpoint 和基础
BF16 运算。这里的失败只能先记为 `ENVIRONMENT`。

**Operator Candidate**：
静态源码或历史证据提示需要验证的语义算子。缺少同名 Kunlun/CUDA symbol 不能单独
证明算子缺失。

**Implementation Classification**：

- `NATIVE_IMPLEMENTATION`：固定 SGLang 已有精确 PyTorch/`forward_native` 语义；
- `KUNLUN_IMPLEMENTATION`：固定 SGLang-Kunlun 已有对应生产实现；
- `NO_IMPLEMENTATION_FOUND`：静态检查尚未找到精确实现，仍需 P800 证据。

实现可以同时存在。分类与生产路由分开记录。

**Route State**：

- `ROUTE_PENDING`：尚未在 P800 验证；
- `PRODUCTION_REACHABLE`：真实生产入口已到达；
- `ROUTE_BLOCKED`：已有能力但生产 dispatch 未接通，先记为 `ADAPTATION`。

**Confirmed Operator Gap**：
P800 生产调用已到达后，确认没有可用实现或接口/数值不满足 Contract。环境、工具和
模型路由失败不属于算子缺口。

**CPU Reference**：
来自固定 native 源码、社区测试或独立公式的局部算子实现。它不能调用待验证的 P800
实现，也不表示运行整模型 CPU reference。

**P800 Production Operator**：
真实模型最终使用的 Kunlun、xspeedgate 或 P800 PyTorch 入口。聚焦测试必须直接调用
它，不能使用 test-only helper。

**Operator Verification**：
比较 CPU Reference 与 P800 Production Operator，并检查结构、shape、dtype、有限值、
stride、alias、输出 buffer 和原地语义。只有生产路由及全部 required cases 通过才算
算子可用。

**Failure Observation**：
一次失败的最小事实记录，category 只使用：

- `ENVIRONMENT`：导入、设备、依赖、checkpoint 或平台问题；
- `ADAPTATION`：模型类、plugin、backend、dispatch 或调用路由问题；
- `OPERATOR_MISSING`：已到达生产调用，但实现不存在；
- `OPERATOR_CONTRACT`：数值、shape、dtype、layout 或原地接口不符；
- `DISTRIBUTED_RUNTIME`：TP/EP、collective、rank、通信、内存或 stream 问题。

**BUG Repair Loop**：
一个 `bug_id` 最多执行 3 次不同、由证据支持的修复尝试。修复后必须通过聚焦复现和
最初失败的 P800 路径；成功就归档并继续，3 次仍失败才 `BLOCKED`。

**Eager Bring-up**：
在固定 TP8 BF16 配置启动真实模型，完成固定文本和单图请求并复跑。服务 ready、响应
结构有效、生成非空且无未处理错误即完成；它不等于整模型精度验证。

**Run Evidence**：
`runs/<run-id>/result.md` 或 `result.json` 中的一次动作记录，保存命令、源码锚点、环境、
输入特征、门槛、结果和下一动作。

**Closure**：
初始和运行中新发现的算子全部通过，固定 eager 请求正常后进入 `PASS / DONE`；同一
BUG 的 3 次不同修复尝试均失败后进入 `BLOCKED`。除此之外不设置计划内停止点。

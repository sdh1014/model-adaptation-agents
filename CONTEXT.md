# KLX P800 Run-Driven Model Adaptation

本项目用一个 Migration Agent 把固定版本的 SGLang 模型适配到 KLX P800。主线是：
先验证环境和已知算子，再运行真实 eager 模型；模型能运行后验证精度，只有精度失败
时才进入逐层、多卡定位。

## Language

**Migration Spec**：
人和 Agent 共享的唯一执行状态。它同时保存不可由 Agent 放宽的 Contract，以及
Agent 每轮更新的 Working State。

**Contract**：
Migration Spec 中由人确认的部分。它固定模型、源码、checkpoint、运行方式、请求、
精度门槛、允许修改范围和停止规则。Agent 可以完整读取，但不能自行修改。

**Working State**：
Migration Spec 中由 Agent 维护的部分。它保存当前 phase、Operator Verification
Queue、模型和精度状态、Failure Observation、证据路径与唯一下一动作。

**Migration Agent**：
唯一做判断和修改的执行者。它读取源码和现场证据，选择 CPU reference、生成聚焦
测试、调用 P800 生产路径、分类失败、修改允许范围内的代码，并更新 Working State。

**Eager Mode**：
decode 和 prefill 的 graph backend 都关闭的 SGLang 执行方式。它不表示启用 draft
模型或投机解码。

**P800 Preflight**：
任何算子判断之前的环境检查，包括固定 worktree、Python 导入、Kunlun plugin、设备、
checkpoint 和基础设备操作。这里的失败类别只能是 `ENVIRONMENT`。

**Operator Candidate**：
静态源码或历史证据提示可能需要适配的生产算子。缺少同名 Kunlun symbol 只能形成
候选，不能单独证明算子缺失。

**Confirmed Operator Gap**：
实际 P800 生产调用已经到达，并出现实现缺失、接口不兼容或数值不满足 Contract 的
算子。测试工具、环境和模型路由失败不属于 Confirmed Operator Gap。

**CPU Reference**：
Agent 从固定版本 SGLang 源码、社区测试或算子数学定义中建立的独立 PyTorch 语义
实现。它不能调用或复制待验证的 P800 实现。

**P800 Production Operator**：
真实模型调用链最终使用的 Kunlun、xspeedgate 或 P800 PyTorch 实现。局部测试必须
调用这个入口，不能用仅为测试新增的 helper 代替。

**Operator Verification**：
使用确定性输入比较 CPU Reference 和 P800 Production Operator，同时检查结构、
shape、约定输出 dtype、有限值、stride、输出 buffer 和原地修改语义。局部通过后
仍需真实模型路径确认可达。

**Precision Gate**：
Contract 固定的比较规则。浮点结果使用指定的 `atol`、`rtol`；整数和布尔结果精确
相等；结构、shape、约定输出 dtype 和有限值必须满足。Agent 不得放宽门槛。

**Failure Observation**：
一次失败的最小事实记录，至少包含 phase、category、命令、错误位置、源码锚点、
证据路径和下一动作。category 只使用：

- `ENVIRONMENT`：导入、设备、依赖、checkpoint 或平台激活问题；
- `ADAPTATION`：模型类、plugin、backend、dispatch 或调用路由不正确；
- `OPERATOR_MISSING`：已到达生产调用，但实现或底层能力不存在；
- `OPERATOR_CONTRACT`：数值、shape、dtype、stride、alias、输出 buffer 或原地语义不符；
- `DISTRIBUTED_RUNTIME`：TP/EP、collective、rank、通信、内存或 stream 问题；
- `ACCURACY`：模型能够完成请求，但最终或逐层结果不满足 Precision Gate。

**Eager Bring-up**：
在 Contract 固定拓扑上启动真实模型并完成固定文本与单图请求的修复循环。它要求
生产路径命中、结果有限且请求可重复执行。

**Model Accuracy Gate**：
模型能够运行后，对固定输入的 logits、离散输出和必要中间结果执行 Contract
规定的比较。服务健康或生成了一段文本都不能单独代表精度通过。

**Accuracy Debug**：
仅在 `ACCURACY` 失败后启用。Agent 优先使用固定 SGLang 版本已有的 dumper、
forward hook 和 comparator，对齐输入与 rank，定位第一个发散层，再缩小到生产算子。

**Run Evidence**：
`runs/<run-id>/result.md` 或 `result.json` 中的一次有界动作记录。它保存结论所依赖
的命令、源码位置、输入特征、门槛、结果与下一动作；默认不保存完整 Tensor。

**Repair Scope**：
允许修改 P800 生产调用附近的 Python、P800 可执行 PyTorch，以及已有
xspeedgate/kunlun_ops 的组合和参数装配。需要新增 C++、自定义 kernel 或底层注册时
进入 `BLOCKED`。

**Closure**：
环境通过、初始 Operator Verification Queue 全部通过、运行中新增的 Confirmed
Operator Gap 全部关闭、固定 eager 模型请求通过、Model Accuracy Gate 通过，并且
没有未处理 Failure Observation。

# 走查一个 SwiGLU 三 Shape 的端到端纸面 Demo

Type: prototype
Status: resolved
Blocked by: 04, 05, 08

## Question

使用已经确定的源码边界、Spec、Golden Run、交接包、补丁生命周期和比较接口，逐步模拟一个特殊 SwiGLU 算子、最多三种 shape 从扫描到 `PASS` 或终止状态的完整过程。这个纸面 Demo 是否暴露了缺失字段、隐式依赖、重复状态或超出 Repair Boundary 的步骤？

## Comments

- 纸面走查与可运行状态原型见 [特殊 SwiGLU 三 Shape 端到端纸面走查](../../../work/prototypes/swiglu-three-shape-walkthrough.prototype.md)。
- 主路径、没有真实 gap、需要新 Kernel、五轮耗尽四类状态轨迹均已在本地运行。
- 用户确认 helper 固定基线、baseline 全样本判定、轮次计数和重复字段删除；最大修复轮数由 Contract 固定为五轮。

## Answer

特殊 SwiGLU 的纸面闭环可以只使用现有 `status + phase` 和 `migration-spec.md + runs/` 完整表达，不需要第二个状态文件或工作流引擎。固定 `sglang_revision` 必须已经包含人工批准的无权重 helper 抽取；只有加载后配置命中该分支、唯一 CUDA Session 已采集最多三个真实 shape，且 P800 baseline 任一样本执行或比较失败时，它才成为 Demo 的真实 Operator Gap。全部 baseline 样本通过时必须换候选。

Contract 新增且固定 `max_repair_attempts: 5`。baseline 不计数；每轮在修改源码前递增 `attempts_used`，通过轮也计数。每轮仍从同一干净基线生成完整补丁，失败后保留 Run 并恢复基线；第五轮仍失败时进入 `BLOCKED`，不得开始第六轮。需要新 C++、自定义 Kernel 或底层注册时立即 `BLOCKED`，不必等五轮用尽。

Working State 删除重复的 `candidate_patch` 与 `last_attempt_run`：通用 `last_run` 指向最近尝试，`passing_run` 同时定位通过 Run 及其中的完整补丁。P800 actual Tensor 只在内存中比较，每轮 Run 保存命令、完整补丁、日志和逐样本 `compare.json`。

经人工确认的完整流程、代码依据和停止分支见 [特殊 SwiGLU 三 Shape 端到端纸面走查](../../../work/prototypes/swiglu-three-shape-walkthrough.prototype.md)。

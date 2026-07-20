---
name: model-adaptation
description: 按固定 Contract 在 KLX P800 上验证算子、跑通 eager 模型并检查精度。
argument-hint: "Model name, e.g. Step-3.7-Flash"
disable-model-invocation: true
---

# Input

把用户传给 Skill 的完整参数作为模型名。当前工作区只接受精确参数
`Step-3.7-Flash`。参数缺失或不同于 Contract Data 的 `model` 时，报告正确调用：

```text
$model-adaptation Step-3.7-Flash
```

然后停止，不猜测其他模型。

# Outcome

从工作区根目录的 `migration-spec.md` 启动或恢复同一个 Migration Agent。目标不是
只关闭局部算子，而是依次完成：

```text
PREFLIGHT
-> OPERATOR_VERIFICATION
-> EAGER_BRINGUP
-> MODEL_ACCURACY
-> ACCURACY_DEBUG（仅精度失败时）
-> DONE
```

初始 Operator Verification Queue 中的五个历史 gap 必须全部实际测试。一个局部算子
通过不能跳过其余条目；局部队列全部通过也不能替代真实模型和精度验收。

本 Skill 不依赖本仓库内的通用 Tensor 采集、序列化、跨机交接或重放程序。Agent
直接读取固定源码，在目标 SGLang-Kunlun 仓库创建聚焦测试、调用 P800 生产入口，并
用 Contract 固定门槛比较。

# State and authority

`migration-spec.md` 是唯一恢复状态：

- `Contract` 由人确认，固定模型、源码、checkpoint、运行方式、请求、Precision
  Gate、Repair Scope 和停止规则；Agent 不得修改。
- `Working State` 由 Agent 更新，保存 phase、队列、模型状态、Failure Observation、
  Run Evidence 和唯一 `next_action`。
- 每次动作前重读整个 Spec；动作后先写新 Run Evidence，再更新 Working State。
- 旧 Contract revision 的 Run 只能提供源码线索，不能替当前 P800 测试或模型结果。

允许 Agent：

- 读取固定 SGLang、SGLang-Kunlun 和 checkpoint 配置；
- 在固定 Kunlun worktree 内新增或修改聚焦测试和 Repair Scope 内的生产代码；
- 运行 CPU、P800、单进程或 Contract 固定多卡命令；
- 使用 SGLang 自带的 dumper、forward hook 和 comparator；
- 按证据更新 Working State。

不允许 Agent：

- 放宽 Precision Gate；
- 用 test-only helper 代替实际生产调用；
- 把环境、工具或路由失败写成算子缺失；
- 自动取得新凭证、猜 checkpoint 或猜节点类型；
- 新增 C++、自定义 kernel 或底层注册。确实需要时进入 `BLOCKED`。

# Precision Gate

所有 Operator Verification 和模型张量比较都从 Contract 读取门槛：

1. CPU reference 可以用 FP32 完成归约，但比较前转换到声明的输出 dtype。
2. 输出容器结构和 shape 必须一致。
3. P800 生产输出 dtype 必须等于源码声明的 dtype。
4. 两侧数字 Tensor 必须都是有限值。
5. 浮点 Tensor 使用 `torch.testing.assert_close` 和固定 `atol`、`rtol`。
6. 整数和布尔 Tensor 精确相等。
7. 原地算子分别比较每个被修改的 buffer；返回 `None` 不代表没有输出。
8. 根据生产接口检查必要的 stride、非连续输入、alias 和输出 buffer 语义。
9. Agent 不得因为失败而改变门槛；只能修实现、修输入构造错误或报告 `BLOCKED`。

# Process

## 1. Validate Contract and workspace

读取 Contract Data，确认：

- `schema` 为 `model-adaptation/v1`；
- 模型参数与 Skill 输入相同；
- SGLang、SGLang-Kunlun revision 和 checkpoint 都已固定；
- TP、dtype、target-only eager、文本与单图请求已固定；
- operator 和 model 两组 Precision Gate 完整；
- Repair Scope 明确禁止新 C++、自定义 kernel 和底层注册。

读取 `docs/p800-environment-and-repair.md`。检查当前工作区已有修改，只修改本轮相关
文件，不清理或覆盖无法解释的用户改动。

## 2. `PREFLIGHT`

在任何算子判断之前，在 P800 现场验证：

- `SGLANG_KUNLUN_WORKTREE` 指向固定 revision；
- `sglang`、`sglang_kunlun` 从固定 worktree 导入；
- Kunlun platform 和 Contract 要求的设备数量可用；
- BF16 Tensor 创建、设备搬运和简单 PyTorch 运算正常；
- checkpoint 可读且配置摘要一致；
- decode 和 prefill graph backend 都关闭；
- 实际环境变量及选择原因已经记录。

把命令、版本、导入路径、设备结果和日志写入新的 Run Evidence。

- 全部通过：`environment_status: PASS`，进入 `OPERATOR_VERIFICATION`。
- 任一失败：记录 `ENVIRONMENT`；修复后重复本 phase。
- 缺少人才能提供的路径、权限或节点类型：进入 `NEEDS_HUMAN`。

环境通过前不得把任何条目标为 `OPERATOR_MISSING` 或
`OPERATOR_CONTRACT`。

## 3. `OPERATOR_VERIFICATION`

按 Spec 表格顺序选择第一项 `PENDING`，改为 `ACTIVE`。初始五项必须全部完成：

1. `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul`
2. `sgl_kernel.gemma_rmsnorm`
3. `sgl_kernel.gemma_fused_add_rmsnorm`
4. `sgl_kernel.topk_sigmoid`
5. `sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel`

### 3.1 Establish the source contract

Agent 从固定版本源码确认：

- 从模型入口到 P800 Production Operator 的实际调用链；
- 输入、输出、直接参数和非 Tensor 参数；
- 输出 dtype、输出 buffer、原地修改和 layout 要求；
- 社区测试或数学定义中的独立 CPU reference。

静态缺少同名 symbol 只能记录为 Operator Candidate。只有实际 P800 生产调用已经
到达，才能判为 `OPERATOR_MISSING` 或 `OPERATOR_CONTRACT`。

### 3.2 Create focused tests

优先扩展目标仓库已有测试；没有合适 seam 时，在目标仓库按其测试约定新增一个聚焦
测试。测试必须直接调用 P800 生产入口，不能在本仓库建设统一 adapter。

输入由代码确定性构造，不保存任意活体 Tensor。涉及非连续布局时，在 CPU 和 P800
分别从 base Tensor 创建 view，显式断言预期 stride。

每项至少覆盖 Spec 的 required cases：

- SwiGLU clamp：`gemm1_limit=None` 和有限 limit；输入包含超过正负 clamp 边界的值，
  同时验证实际 Kunlun 调用取得模型配置中的 limit。
- Gemma RMSNorm：BF16、生产 hidden size、较小诊断 shape、连续和
  `base[:, :hidden]` 非连续输入；参考计算用 FP32 归约后转回输入 dtype。
- Gemma fused add RMSNorm：除 RMSNorm 条件外，分别比较修改后的 `x` 和
  `residual`，确认函数返回 `None` 时两个输出仍被验证。
- TopK sigmoid：有/无 `correction_bias`、`renormalize` 分支、生产 top-k 和专家数；
  构造无并列分数，weights 按浮点门槛比较，ids 精确相等。
- Visual prefill attention：ragged sequence、causal 分支、生产 head dim、Q/KV head
  数和 GQA 映射；按 `b_start_loc`、`b_seq_len` 分段，用
  `torch.nn.functional.scaled_dot_product_attention` 建立 CPU reference。

历史 shape 可以作为 case 线索，但不能单独替代当前 P800 生产测试。运行时出现的新
shape 要加入当前算子的聚焦回归测试。

### 3.3 Compare and decide

运行 CPU reference 和 P800 生产算子，按 Precision Gate 比较并保存 Run Evidence。

- 直接通过：队列行改为 `PASS`。
- 实现不存在：记录 `OPERATOR_MISSING`，在 Repair Scope 内修复并重测。
- 执行成功但数值、dtype、layout 或原地语义失败：记录
  `OPERATOR_CONTRACT`，修复并重测。
- 生产调用没有到达：记录 `ADAPTATION`，先修实际路由。
- 多卡通信、rank、内存或 stream 失败：记录 `DISTRIBUTED_RUNTIME`。

修复通过后保留聚焦回归测试，并继续下一条，不等待人工选择方案。五项全部
`PASS` 后进入 `EAGER_BRINGUP`。

## 4. `EAGER_BRINGUP`

使用 Contract 固定的 TP8、BF16、target-only eager 配置启动真实模型，依次执行
固定文本和单图请求。至少验证：

- 服务或离线 engine 完成启动；
- 实际 backend 和调用路径符合固定源码；
- 两种请求都完成，输出和必要 logits 有限；
- 相同输入在确定性设置下可以重复；
- 初始五个算子的 P800 Production Operator 在真实模型路径可达。

遇到失败时按以下规则处理：

| category | rule | action |
|---|---|---|
| `ENVIRONMENT` | 生产算子前的导入、设备、依赖、checkpoint 问题 | 回到 `PREFLIGHT` |
| `ADAPTATION` | 模型类、plugin、backend、dispatch 或调用路由错误 | 修适配代码后重启 |
| `OPERATOR_MISSING` | 实际到达的新生产算子没有实现 | 追加队列并回到 `OPERATOR_VERIFICATION` |
| `OPERATOR_CONTRACT` | 实际 shape/dtype/layout/原地语义暴露新问题 | 扩充对应聚焦测试并回到算子验证 |
| `DISTRIBUTED_RUNTIME` | TP/EP、collective、rank、通信、内存或 stream 问题 | 建立最小多卡复现后修复 |
| `ACCURACY` | 请求完成但结果不满足门槛 | 进入 `MODEL_ACCURACY` |

每轮只处理有证据的首个根因。修复后重新运行失败请求；两个固定请求都通过后设置
`model_status: PASS`，进入 `MODEL_ACCURACY`。

## 5. `MODEL_ACCURACY`

使用 Contract 固定的同版本 SGLang reference runtime 和相同 checkpoint、输入、
tokenizer、TP 拓扑、dtype 与 eager 设置进行比较。不要只比较自然语言观感。

至少检查：

- 输入 token、图像预处理结果和生成配置一致；
- 第一个生成位置和需要的后续位置 logits 满足 model Precision Gate；
- 离散 token、router ids 等整数结果精确相等；
- 两种固定请求都通过；
- P800 结果全部有限。

通过后设置 `accuracy_status: PASS` 并进入 `DONE`。失败时记录 `ACCURACY`，进入
`ACCURACY_DEBUG`，不得回头放宽门槛。

## 6. `ACCURACY_DEBUG`

优先复用固定 SGLang 版本已有的：

- `python/sglang/srt/debug_utils/dumper.py`
- `python/sglang/srt/debug_utils/tensor_dump_forward_hook.py`
- `python/sglang/srt/debug_utils/comparator/`

保持两端输入、tokenizer、checkpoint、拓扑、dtype 和 eager 设置一致。先比较
embedding、每个 decoder layer 输出、final norm 和 logits，定位第一个发散层；多卡
Tensor 必须记录 rank 和切分维度，必要时给 comparator 提供 dims override。

找到第一个发散层后：

1. 对比该层输入，确认误差不是上游传播；
2. 缩小到模块和 P800 Production Operator；
3. 将新算子或新 case 加入 Operator Verification Queue；
4. 修复并依次重跑局部测试、真实模型和 Model Accuracy Gate。

函数级 logger 只可辅助确认调用和崩溃；对返回 `None` 的原地算子，必须显式保存并
比较修改后的参数，不能只查看返回值。

# Run Evidence

每次动作创建新的 `runs/<run-id>/result.md` 或 `result.json`，至少记录：

- Contract revision、phase、category 和结论；
- SGLang、SGLang-Kunlun commit、checkpoint 和完整命令；
- CPU reference 的固定源码位置；
- P800 Production Operator 的固定源码位置和真实调用链；
- 输入 shape、dtype、layout、stride、seed 和关键标量；
- 使用的 Precision Gate；
- 通过项、失败项、traceback 和下一动作。

默认不保存完整输入输出 Tensor。只有 `ACCURACY_DEBUG` 无法由统计和局部测试定位时，
Agent 才保存必要的定点张量，并在 Run Evidence 中说明原因和范围。

# State transitions and stop conditions

- `ACTIVE`：保存证据、更新 Working State、重读 Spec 并继续唯一下一动作。
- `NEEDS_HUMAN`：只用于缺少人才能提供的路径、权限、节点类型、baseline 或 Contract
  决定；写一个明确问题后停止。
- `BLOCKED`：需要新增 C++、自定义 kernel、底层注册或越过 Repair Scope；保存证据后
  停止。
- `PASS / DONE`：环境、全部队列项、两个真实模型请求和 Model Accuracy Gate 全部
  通过，且没有未处理 Failure Observation。

不自动 SSH、登录、管理凭证、创建分支、commit 或 push。用户明确提供远端会话或
要求相应 Git 动作时再执行。

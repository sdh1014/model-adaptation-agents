# 一次性采集完整缺口清单中的候选算子

Type: task
Status: ready-for-agent
Blocked by: 13, 20

## What to build

实现并在 CUDA 机器上运行 `capture_golden.py`，消费完整 Scan Run 和 Agent 生成的采集计划，在唯一一次 CUDA Capture Session 中采集所有 `CAPTURE_REQUIRED` 且满足 Contract 的候选算子，而不是只采集特殊 SwiGLU。

采集发生在 SGLang 已确认的 Semantic Operator 边界。每个算子按稳定签名去重，最多保留三个真实调用形态；重复调用和第四个以后的新形态只计数，不再保存数据。每个样本保存边界输入、CUDA 期望输出、必要标量和重放元数据，不保存权重或内部 Tensor。

Session 结束后，在已加载同一 checkpoint 的 CUDA TP8 模型 rank 0 上对全部 Golden Samples 自回放。只有全部通过，才封存 Golden Run、构建交接包并完成 CUDA 端校验。

## Acceptance criteria

- Session 开始前，target-only eager 扫描完成，所有采集候选都有 hook target、运行命令、边界说明和代码证据。
- 所有无法安全采集或仍为 `NEEDS_HUMAN` 的项都在 Session 前处理；Session 开始后不能为了补遗漏再创建第二次 Session。
- 每个候选算子最多保存三个不同签名，重复次数和被跳过的新形态数量有记录。
- 去重签名不包含 Tensor 数值或输出，但包含算子、执行路径、输入结构、shape、dtype、layout、stride 和必要非 Tensor 参数。
- Golden Sample 不包含权重、内部中间 Tensor、完整 batch、KV cache 或不可重建运行时对象。
- 新进程自回放覆盖全部已保存样本；任一失败都会阻止 Golden Run 封存和 bundle 构建。
- bundle 构建及 CUDA 端 verify 通过后，Spec 原子进入 `WAITING / HANDOFF`，并记录唯一 Session id、样本计数和 Golden Run 指针。
- 完成后不再需要访问 CUDA 机器；后续特殊 SwiGLU 和其他算子的处理都使用本次数据。

## Comments

特殊 SwiGLU 是首选验证样例，但唯一 CUDA Session 必须覆盖完整扫描中所有符合 Contract 的采集候选。

2026-07-17 纠正：本票据只消费 Contract revision 3 的 target-only eager 扫描结果 `runs/scan-003`；不得使用旧 EAGLE 范围的 `scan-002` 启动唯一 CUDA Capture Session。

2026-07-17 边界更新：正式 Session 改为消费 revision 4 的 `scan-004`，在 TP8 模型 rank 0 的原始 `Step3p5MLP.forward` 保存每种 shape 的一份 `x/output`；权重由同一 checkpoint 提供，revision 3 的旧方案只保留为历史。

# 设计基于 torch.testing 的精度比较接口

Type: prototype
Status: resolved
Blocked by: 03

## Question

用户已在 P800 实机确认 `torch.testing` 可用。精度比较器应怎样直接基于 `torch.testing.assert_close` 接收 CUDA Golden 与 P800 输出、严格执行 Contract 固定的结构、dtype、有限值和 `atol`/`rtol` 规则，并输出足够支持下一轮修复的最小诊断证据？

## Comments

- 用户提供新的实机结论：P800 修改版 Torch 的 `torch.testing` 已验证可用，方案应直接按该实现设计，不再保留 Torch/NumPy 多后端适配层。
- 逻辑原型见 [基于 torch.testing 的最小精度比较接口](../../../work/prototypes/comparator-interface.prototype.md)。
- 用户确认：P800 actual output 只在内存中参与比较，Run 保存 `compare.json` 和日志，不为每轮尝试额外保存 actual Tensor。

## Answer

比较器直接绑定 `torch.testing.assert_close`，不再增加 Torch/NumPy Adapter。它是 replay 工具内部调用的 Deterministic Tool：接收 `spec_path`、Golden Sample 和内存中的 P800 actual output；自己从 Contract 读取 `atol/rtol`，接口不允许 Agent 传入覆盖值。

`assert_close` 之前必须按固定顺序检查 Golden 中保存的输出结构、Tensor shape、dtype 和两端有限值。结构检查不能完全委托给 `assert_close`，因为 PyTorch 允许不同 Sequence 类型逐元素比较；有限值也要单独检查，因为相同位置的 `inf` 可能被视为接近。随后将 expected 与 actual Tensor 保持原 dtype 复制到 CPU，逐叶调用：显式传入 Contract 的 `atol/rtol`，使用 `equal_nan=False`、`check_dtype=True`、`check_layout=True`、`check_stride=False`。

每个样本生成一个 `compare.json`，最少记录 sample、Torch 版本、实际使用的 Precision Gate、结构/dtype/有限值检查、逐 Tensor 的 shape/dtype、PASS、最大绝对差、不匹配数量和原始 `AssertionError` 文本。诊断字段只帮助形成下一轮假设，不构成第二套门槛；最多三个 Golden Samples 必须全部通过，不允许 skip。

P800 actual output 只在本次 replay 进程内存中使用，不额外落盘。Run 已保存 Golden 指针、命令、补丁、`compare.json` 和日志；需要完整 actual 时可以在同一台 P800 上再次 replay，从而避免五轮重复保存大 Tensor。

完整接口、判定顺序、结果示例、官方语义和 SGLang 代码依据见 [逻辑原型](../../../work/prototypes/comparator-interface.prototype.md)。

# 确认跨 CUDA/P800 的样本格式与精度比较

Type: task
Status: ready-for-agent
Blocked by: 10

## What to build

使用一个无权重的合成算子样本，在实际 CUDA Torch 和 P800 修改版 Torch 环境中验证可移植的数据格式，再把验证通过的格式用于 Golden Sample。

Golden Sample 只保存边界输入 Tensor、CUDA 期望输出 Tensor、容器结构和重放必需的非 Tensor 参数。默认不保存权重、内部中间 Tensor、完整 batch、KV cache 或运行时对象。

完成 `replay_compare.py` 的最小可信比较路径：先检查容器结构、shape、dtype 和有限值，再使用 Contract 固定的 `torch.testing.assert_close`、`atol` 和 `rtol` 判定。P800 actual output 只在当前进程内存中参与比较，不额外保存。

## Acceptance criteria

- 候选格式在实际 CUDA 和 P800 环境中完成写入、人工复制、读取和数值 round-trip，并记录两端 Torch 版本。
- 只有跨端验证通过后，才把序列化格式写入实现约束；不能根据标准 PyTorch 行为直接假定 P800 兼容。
- 一个包含嵌套容器、输入 Tensor、期望输出和必要标量的无权重样本可以在两端恢复。
- 比较器严格执行结构、shape、dtype、有限值和 `torch.testing.assert_close`；容差不能被命令行覆盖。
- `compare.json` 同时支持 PASS 和有意制造的 FAIL，并保存定位所需的最小误差诊断与异常信息。
- actual output 不写入 Run；Run 保存 Golden Sample 指针、比较参数、结果和日志。
- 结果沿用 Ticket 10 的 Spec 绑定规则，工具未完成与精度未通过使用不同语义表达。
- 有自动化测试覆盖正常样本、结构不一致、dtype 不一致、非有限值和超出容差。

## Comments

用户已在 P800 实机确认 `torch.testing` 可用；本票据不再设计 NumPy 等备用比较后端。

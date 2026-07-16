# 确认跨 CUDA/P800 的样本格式与精度比较

Type: task
Status: ready-for-human
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

已完成 CUDA preflight 的本地实现，尚未把本票据标为 resolved：

- `capture_golden.py --mode preflight` 绑定 Contract revision 2、`scan-002`、固定 CUDA SGLang revision 和特殊 SwiGLU，preflight 明确不消耗正式 CUDA Capture Session。
- 采集插件使用 SGLang 现有 `sglang.srt.plugins` 与 `HookRegistry`，只挂 target 的 `Step3p5ForCausalLM.forward` 和 `step_swiglu_with_limit`；draft 的 `Step3p5MTP.forward` 不在采集上下文内。
- 样本最多保存三种去重调用签名，只包含 BF16 `gate_up`、标量 `limit`、CUDA 期望输出和最小结构元数据；拒绝 `nn.Parameter` 与非有限值，第四种调用不落 tensor。
- preflight 会启动第二个进程，用 `torch.load(..., weights_only=True)` 读回，再按 `torch.testing.assert_close(atol=0.01, rtol=0.02)` 重放。
- Claude Code 项目入口位于 `.claude/skills/model-adaptation/SKILL.md`，N 卡执行步骤位于 `model-adaptation/references/cuda-capture-validation.md`。
- 本地 27 个测试全部通过；其中新增测试直接加载固定 SGLang revision 的真实 `HookRegistry` 与 `step3p5_ops.py`，确认两个 Hook、imported binding 传播、target `decode` 上下文和 `3 保存 + 1 重复 + 1 跳过签名`。无 CUDA 的本机只验证了 preflight 失败路径会生成可信证据且 `consumes_capture_session=false`。

下一人工动作是在真实 N 卡 SGLang 环境执行 `runs/cuda-preflight-001`，把 runbook 列出的六个文件带回审查。只有该结果通过后才继续正式模型采集。候选 `torch.save` 格式仍未在 P800 修改版 Torch 完成读回，因此本票据仍不满足跨端 acceptance criteria，`replay_compare.py` 的正式 Golden compare 路径也不在本轮提前宣称完成。

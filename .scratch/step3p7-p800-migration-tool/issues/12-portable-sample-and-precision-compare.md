# 确认跨 CUDA/P800 的样本格式与精度比较

Type: task
Status: resolved
Blocked by: none

## What to build

使用一个无权重的合成算子样本，在实际 CUDA Torch 和 P800 修改版 Torch 环境中验证可移植的数据格式，再把验证通过的格式用于 Golden Sample。

Golden Sample 保存 kernel 边界输入 Tensor、CUDA 期望输出 Tensor、容器结构和重放必需的非 Tensor 参数。revision 5 允许保存当前 kernel 调用直接使用的参数 Tensor；不保存完整 checkpoint、module `state_dict`、无关参数、完整 batch、KV cache 或运行时对象。

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
- 本地 28 个测试全部通过；其中新增测试直接加载固定 SGLang revision 的真实 `HookRegistry` 与 `step3p5_ops.py`，确认两个 Hook、imported binding 传播、target `decode` 上下文和 `3 保存 + 1 重复 + 1 跳过签名`。无 CUDA 的本机只验证了 preflight 失败路径会生成可信证据且 `consumes_capture_session=false`。

下一人工动作是在真实 N 卡 SGLang 环境执行 `runs/cuda-preflight-001`，把 runbook 列出的六个文件带回审查。只有该结果通过后才继续正式模型采集。候选 `torch.save` 格式仍未在 P800 修改版 Torch 完成读回，因此本票据仍不满足跨端 acceptance criteria，`replay_compare.py` 的正式 Golden compare 路径也不在本轮提前宣称完成。

2026-07-17 纠正：此前把 eager 误写为 EAGLE。旧 revision 2 / `scan-002` preflight 已取消且从未执行；本票据先由 Ticket 18 阻塞，待 revision 3 的 target-only eager `scan-003` 封存后，再按更新后的 runbook 执行 N 卡 preflight。

Ticket 18 已完成，`runs/scan-003` 与 revision 3 绑定通过，本票据现已解除扫描阻塞。下一人工动作仍是在真实 N 卡 SGLang 环境按更新后的 runbook 执行 `runs/cuda-preflight-001`；本地准备通过不等于 N 卡 preflight 通过。

2026-07-17 范围更新：用户决定保留模型原始 `Step3p5MLP.forward` 边界。revision 3 的旧 preflight 因此取消且从未执行；本票据改由 Ticket 20 阻塞，待 revision 4 的模型内重放实现完成后再进入实机验证。

Ticket 20 已完成并解除阻塞。revision 4 在 TP8 模型中只采集和重放 rank 0，最多三种 shape；样本保存 `x/output/limit` 与必要元数据，不保存权重或内部 Tensor。下一人工动作是按当前 runbook 在真实 N 卡运行 `scan-004` preflight；本地测试不等于 CUDA/P800 round-trip 已通过。

本地可执行性复核继续补强，但不替代上述 N 卡动作：

- 在两个固定 revision 的真实工作树上检查了 `scan-002` 的 13 个算子和 50 个 CUDA/Kunlun 源码锚点，文件与行号全部有效。
- 用当前 Contract revision 2、`scan-002` 和固定 CUDA SGLang 源码执行 `capture_golden.py --mode prepare`，成功得到 `PREPARED`，并固定特殊 SwiGLU、最多三个 BF16 样本和 `atol=0.01 / rtol=0.02`；该动作不使用 CUDA，也不消耗 Capture Session。
- 隔离 wheel 安装和 runbook 使用的 editable 安装都能发现唯一的 `model_adaptation_capture -> model_adaptation_capture.plugin:register` entry point。完整 SGLang loader 仍必须在依赖齐全的 N 卡 SGLang Python 中验证；本机环境缺少 `orjson`，没有把该失败写成插件或算子结论。
- Claude Code 2.1.191 的只读 headless smoke 确认项目 `.claude/skills` 被加载且发现唯一项目 Skill；该 smoke 禁用了 Bash，并在形成最终回复前达到预算上限，因此只作为 Skill 发现证据，不作为 CUDA preflight 通过证据。
- runbook 现在要求调用者显式设置非空 `SGLANG_WORKTREE`；空值会硬失败，不再可能退回默认目录后以 `skipped` 退出。

2026-07-17 revision 5 更新：Ticket 21 已用 `scan-005` 选择
`sgl_kernel.gemma_rmsnorm`，本票据现在由 Ticket 22 的 kernel adapter 阻塞。旧
`scan-004` MLP preflight 不再是当前人工动作。跨端格式验证要覆盖直接参数
`weight`，同时证明完整 checkpoint 和 module state 会被拒绝。

2026-07-17 复核更新：`scan-005` 保留为历史，当前 `scan-006` 改选
`_swiglu_silu_clamp_mul`，本票据改由 Ticket 23 阻塞。首选样本不保存权重；格式
验证仍要拒绝任何超出当前调用边界的参数、完整 checkpoint 和 module state。

2026-07-17 Ticket 23 完成：revision 5 adapter 已实现并通过本地测试，当前不再受
代码实现阻塞。下一人工动作是在 CUDA 机器执行 revision 5 preflight 并通过 GitHub
回传 Run；只有该证据通过后，才继续正式 CUDA Capture 和后续 P800 round-trip。

2026-07-17 CUDA evidence 更新：`runs/cuda-preflight-r5-001` 已通过 GitHub 回传。
简单校验确认结果绑定 Contract revision 5，rank 0 保存三个 shape，rank 1 未写样本，
CUDA 新进程 self-replay 三个样本全部通过，且
`consumes_capture_session=false`。本机没有把 Tensor 复算当作验收证据。本票据仍为
`ready-for-human`：下一步只在 P800 修改版 Torch 上读回这三份样本，记录 P800
Torch 版本，并用 Contract 固定的 `torch.testing.assert_close` 形成一次 PASS 和
一次有意 FAIL 的 `runs/p800-portability-r5-001`。该动作只确认格式与比较器，不执行
`kunlun_ops.swiglu` baseline，也不开始正式 Capture。

2026-07-17 P800 evidence 完成：`runs/p800-portability-r5-001` 直接基于
`runs/cuda-preflight-r5-001`，在 P800 修改版 Torch `2.5.1+cu118` 上读回三个 BF16
样本。三个输入/期望输出的结构、shape、dtype、有限值和固定
`torch.testing.assert_close(atol=0.01, rtol=0.02)` 全部通过；同 shape、同 dtype 的
有意数值偏差被正确拒绝并保留最小异常信息。Run 记录 CUDA Torch
`2.11.0+cu129`、P800 Torch 版本和源样本指针，没有保存 P800 actual Tensor。
跨 CUDA/P800 格式与比较器的最后一项实机证据已补齐，本票据关闭。

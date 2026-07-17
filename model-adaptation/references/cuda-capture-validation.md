# CUDA capture validation

## 当前状态

当前 Contract 是 revision 5，`scan-006` 在扫描完成后选择
`sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul`。
对应的 capture/replay adapter 尚未实现，所以现在不能执行 CUDA preflight。

仓库中现有 `Step3p5MLP.forward` adapter 和旧命令属于 revision 4 历史方案。不要
把 `scan-006` 的选择替换回 MLP，也不要用旧 adapter 生成 revision 5 证据。

当前唯一下一步以 `migration-spec.md` 为准：

```text
实现 Ticket 23 的 _swiglu_silu_clamp_mul capture/replay adapter；
本地测试通过前不得运行 CUDA preflight
```

这个等待不会消耗唯一 CUDA Capture Session。

## adapter 完成后必须验证的边界

必须 Hook 现有 SGLang call-site：

```text
sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul
```

不能新增 helper、自定义算子函数或模型 wrapper。

每个 rank 0 样本只保存：

- 输入 `x`；
- 标量 `gemm1_limit`；
- CUDA `output`；
- operator、shape、dtype、stride 和来源元数据。

禁止保存：

- 完整 checkpoint；
- module `state_dict`；
- 其他层或其他 kernel 的参数；
- 完整 batch、prompt、token、KV cache 或运行时 handle。

同一输入 shape 只保存第一次，最多三个 shape。其他 TP rank 正常执行模型但不
落盘。

## preflight 验收

Ticket 23 完成后，runbook 才能补入真实命令。preflight 至少要证明：

1. Contract revision 5、`spec-binding-004` 和 `scan-006` 绑定一致；
2. 请求的 operator 等于 `scan-006.selection.active_operator`；
3. HookRegistry 实际包裹的是上面的现有 call-site；
4. rank 0 保存三种 shape，重复 shape 和第四种 shape 不新增样本；
5. payload 不包含参数 Tensor，并拒绝 checkpoint、module state 和边界外参数；
6. 新进程能加载样本并调用同一 `_swiglu_silu_clamp_mul` 接口 self-replay；
7. 比较使用 Contract 固定的
   `torch.testing.assert_close(atol=0.01, rtol=0.02)`；
8. preflight 结果明确写 `consumes_capture_session=false`。

preflight 不启动正式 checkpoint 模型，不证明 P800 已有真实缺口，也不证明跨机器
序列化已经通过。

## 正式 CUDA Session

只有 preflight 通过并由人确认后才能开始：

1. 使用固定 CUDA revision、checkpoint、TP8、BF16 和 eager 启动实际模型；
2. 不传量化、投机解码、MTP、attention backend 或 MoE backend 参数；
3. 采集插件只增加环境变量，不改变两端模型启动参数；
4. 用真实文本请求触发 MoE 第 43、44 层的 clamp SwiGLU；
5. 收集最多三个 rank 0 shape；
6. 停止采集后，在同一次 CUDA Session 内完成全部样本 self-replay；
7. 全部通过后封存 Golden Run，再构建和校验 Handoff Bundle。

任一步失败都要保存 Run。正式 Session 一旦消耗，不得重新启动第二次采集。

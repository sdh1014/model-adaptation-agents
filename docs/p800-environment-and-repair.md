# P800 环境问题与 SwiGLU-clamp 修复审查

本文把两类事实分开记录：

- P800 环境中实际遇到的问题；
- revision 5 的数值结果与正式代码审查结论。

旧 Run 保持不可变。`runs/repair-attempt-1-r5-001` 的三个 shape 数值结果仍可说明
`kunlun_ops.swiglu(limit=...)` 能消除当前差异，但其中新增生产 helper 的实现形态
未通过正式审查，不能作为后续方案的代码基线。

## 1. 已遇到的环境问题

这些问题发生在算子执行之前，属于环境或导入链问题，不能记作 Operator Gap。

| 报错 | 原因 | 当时的处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'sglang_kunlun'` | 当前 Python 环境没有安装或暴露 Kunlun plugin | 对固定 worktree 做 editable install |
| `SGLANG_PLATFORM='kunlun' not found` | Kunlun 平台入口点没有在当前环境注册 | editable install 后设置 `SGLANG_PLATFORM=kunlun`、`SGLANG_USE_XPU=1` |
| `No module named 'sglang.srt.plugins'` | site-packages 中的 SGLang 抢先于固定 worktree 被导入 | replay worker 把 `<worktree>/python` 放到 `PYTHONPATH` 前部 |
| `No module named 'triton.tools.tensor_descriptor'` | 环境中的 Triton 与 flashinfer 导入要求不匹配 | 设置 `SGLANG_IS_FLASHINFER_AVAILABLE=false`，避开本轮不需要的 flashinfer |
| `Torch not compiled with XPU enabled` | Kunlun 平台没有正确激活时走到了错误设备路径 | 先解决 plugin 注册与环境变量，再执行 replay |

当时使用的环境准备方式是：

```bash
pip install -e /workspace/baidu/aicapx/sglang/sglang-kunlun \
  --no-deps --no-build-isolation

export SGLANG_PLATFORM=kunlun
export SGLANG_USE_XPU=1
export SGLANG_IS_FLASHINFER_AVAILABLE=false
```

这些命令只用于恢复正确运行环境，不是算子修复。

## 2. 已确认的数值差异

CUDA 的现有调用 `_swiglu_silu_clamp_mul(x, gemm1_limit)` 会：

1. 对 gate 半段执行 SiLU；
2. 把 gate 上界限制为 `gemm1_limit`；
3. 把 linear 半段限制在 `[-gemm1_limit, gemm1_limit]`；
4. 相乘得到输出。

固定 Kunlun revision 中的生产路径原先调用：

```python
kunlun_ops.swiglu(x=y, y=out1)
```

它没有把 `layer.moe_runner_config.gemm1_clamp_limit` 传给已有
`kunlun_ops.swiglu`。P800 baseline 的三个 Golden shape 中，一个通过、两个在固定
`torch.testing.assert_close(atol=0.01, rtol=0.02)` 下失败。

设备探针与旧 attempt 的数值 replay 表明，`kunlun_ops.swiglu` 已支持可选
`limit=<float>`；传入 limit 后三个 shape 均通过。这里保留的是数值结论，不接受旧
attempt 为回放而新增 helper 的做法。

## 3. 正式审查结论

`35aa72e` 相对 `03cb3a1` 的正式 Standards 与 Spec 审查保存在
`runs/code-review-35aa72e-001`。审查拒绝以下设计：

- 在 `unquant.py` 新增 `apply_gemm1_swiglu_clamp`；
- 用通用 `<module>:<callable>` 协议导入任意“修复函数”；
- 让 replay 验证新增 wrapper，而不是原有 `kunlun_ops.swiglu` Kernel Call。

原因很直接：这会改变原始算子边界，也为一个简单参数传递增加不必要的生产函数和
通用协议。

## 4. 修正后的内联补丁

新的建议补丁保存在
`runs/inline-repair-correction-r5-001/candidate.patch`，只修改原调用位置：

```diff
     d = y.shape[-1] // 2
     out1 = torch.empty(y.shape[:-1] + (d,), dtype=y.dtype, device=y.device)
-    kunlun_ops.swiglu(x=y, y=out1)
+    gemm1_limit = layer.moe_runner_config.gemm1_clamp_limit
+    if gemm1_limit is None:
+        kunlun_ops.swiglu(x=y, y=out1)
+    else:
+        kunlun_ops.swiglu(x=y, y=out1, limit=float(gemm1_limit))
```

这份补丁有三个边界：

- 不新增生产函数；
- 保持 `unquantized_fused_moe_apply_kunlun` 原有结构；
- 修复前后都调用已有 `kunlun_ops.swiglu`。

SOURCE 机器没有 P800 环境，因此该 Run 的状态是
`READY_FOR_P800_VALIDATION`，不是硬件 PASS。后续正式候选 replay 必须绑定
`candidate.patch`，但 `invocation_target` 仍然是 `kunlun_ops.swiglu`。

## 5. 对后续队列的影响

revision 6 不会把这一轮旧数据直接当作全队列 Golden。它会：

1. 重新绑定完整 gap queue；
2. 在一个新的 CUDA Session 中一次收齐计划修复项；
3. P800 按顺序逐项 baseline 和修复；
4. 后一项从前一项的累计通过 patch 开始；
5. 简单修复继续内联原调用位置；
6. 只有全部队列项通过才进入 `PASS / DONE`。

旧 revision 5 的环境、baseline 和数值证据继续保留为历史依据，不被改写。

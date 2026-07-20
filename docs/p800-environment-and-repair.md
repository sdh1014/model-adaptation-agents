# P800 环境与运行修复

本文只描述 P800 进程启动前的环境规则和已知现场问题。环境失败统一记为
`ENVIRONMENT`，不能记为 `OPERATOR_MISSING`。

## 1. 固定源码和导入顺序

所有局部算子测试和真实模型进程都先设置：

```bash
export SGLANG_PLATFORM=kunlun
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export PYTHONPATH="$SGLANG_KUNLUN_WORKTREE/python:$SGLANG_KUNLUN_WORKTREE/sglang-kunlun${PYTHONPATH:+:$PYTHONPATH}"
```

`SGLANG_KUNLUN_WORKTREE` 必须是 Contract 固定 revision 的 Git 根目录。
`PYTHONPATH` 必须让该 worktree 中的 SGLang 和 Kunlun plugin 排在 site-packages
之前。Agent 还要记录：

- `git rev-parse HEAD`；
- 实际导入的 `sglang.__file__` 和 `sglang_kunlun.__file__`；
- Torch 版本、设备可见性和设备数量；
- checkpoint 路径与配置摘要；
- 实际设置的每个可选环境变量及选择原因。

## 2. Preflight

Agent 在判断任何算子前依次验证：

1. 固定 SGLang 和 SGLang-Kunlun worktree 存在且 revision 正确；
2. Python 从固定 worktree 导入 `sglang` 和 `sglang_kunlun`；
3. Kunlun 平台已激活，Torch 能看到 Contract 要求的设备数量；
4. BF16 基础 Tensor 创建、设备搬运和一个简单 PyTorch 运算正常；
5. checkpoint 可读，配置摘要与 Contract 一致；
6. eager 启动参数没有启用 decode 或 prefill graph；
7. 不需要的 CUDA-only backend 没有被错误导入。

任一检查失败，先保存命令、traceback 和现场版本，再修复环境。环境通过前不创建
Operator Verification 的 PASS/FAIL 结论。

## 3. 可选运行变量

以下是候选项，不是统一启动模板。Agent 只能根据节点类型、通信拓扑、实际 backend
和当前错误选择，并在 Run Evidence 中记录原因。

| 环境变量 | 常见值 | 使用条件 |
|---|---|---|
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | D 节点 `32`；P 节点 `256` | 实际使用 DeepEP |
| `XSHMEM_SYMMETRIC_SIZE` | D 节点 2GB；P 节点 8GB | 实际使用 BKCL/RDMA |
| `XSHMEM_QP_NUM_PER_RANK` | `32` | 需要多流 RDMA |
| `BKCL_RDMA_VERBS` | `1` | 明确使用 RDMA Verbs |
| `SGLANG_HEALTH_CHECK_TIMEOUT` | `120` | 模型启动较慢 |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` | `0` | 为正确性关闭计算通信重叠 |
| `USE_FAST_ALLOC_EXTEND_KUNLUN` | `False` | 排查 fast allocator |
| `SGLANG_OPT_USE_JIT_NORM` | `True` | 实际选择 Kunlun JIT norm |
| `SGLANG_OPT_BF16_FP32_GEMM_ALGO` | `torch` | BF16/FP32 GEMM 使用 Torch 路径 |
| `SGLANG_OPT_SWIGLU_CLAMP_FUSION` | `0` | 排查 SwiGLU clamp 路由 |
| `SGLANG_OPT_USE_FUSED_HASH_TOPK` | `0` | 排查 fused hash TopK |
| `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK` | `0` | 排查 JIT fused TopK |
| `SGLANG_OPT_CP_REARRANGE_TRITON` | `False` | 排查 CP rearrange |
| `SGLANG_PREP_IN_CUDA_GRAPH` | `False` | 保持 eager 调试 |

如果某个值取决于 D/P 节点而现场无法确认节点类型，进入 `NEEDS_HUMAN`，不能猜值。

## 4. 已知环境问题

| 报错 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'sglang_kunlun'` | Kunlun plugin 未安装或未暴露 | 对固定 worktree 做 editable install，并核对导入路径 |
| `SGLANG_PLATFORM='kunlun' not found` | 平台入口点未注册 | 安装固定 plugin 后设置 `SGLANG_PLATFORM=kunlun` |
| `No module named 'sglang.srt.plugins'` | site-packages 抢先于固定源码 | 修正 `PYTHONPATH` 顺序 |
| `No module named 'triton.tools.tensor_descriptor'` | Triton 与 CUDA-only 依赖不匹配 | 确认本轮不需要该 backend 后关闭其导入 |
| `Torch not compiled with XPU enabled` | Kunlun 平台未正确激活或 Python 错误 | 先修 plugin、Torch 和设备环境 |

这些问题发生在生产算子执行之前，不能记为 `OPERATOR_MISSING`。

## 5. 代码修复规则

- 环境修复不能顺带修改算子语义。
- 简单算子修复保留原生产调用位置，不为测试新增生产 helper。
- 修改后先运行聚焦 CPU reference 与 P800 生产算子测试，再启动真实模型。
- 需要新增 C++、自定义 kernel 或底层注册时进入 `BLOCKED`。

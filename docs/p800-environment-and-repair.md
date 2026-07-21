# P800 环境与运行修复

本文只描述 P800 进程启动前的环境规则和已知现场问题。环境失败统一记为
`ENVIRONMENT`，不能记为 `OPERATOR_MISSING`。

## 1. 固定源码和导入顺序

所有局部算子测试和真实模型进程都先设置：

```bash
export SGLANG_PLATFORM=kunlun
export SGLANG_USE_XPU=1
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export PYTHONPATH="$SGLANG_KUNLUN_WORKTREE/python:$SGLANG_KUNLUN_WORKTREE/sglang-kunlun${PYTHONPATH:+:$PYTHONPATH}"
```

`SGLANG_KUNLUN_WORKTREE` 必须是 Contract 固定 revision 的 Git 根目录。
`PYTHONPATH` 必须让该 worktree 中的 SGLang 和 Kunlun plugin 排在 site-packages
之前。Kunlun platform 还依赖已注册的 Python entry point；只把目录放进
`PYTHONPATH` 不能证明 entry point 已注册，Preflight 必须实际调用 platform
discovery。

这些要求来自 Contract 固定的 Kunlun commit
`546ad8c682392922792bbbfe53a8bf575545f118`：

- `sglang-kunlun/pyproject.toml:18-22` 注册 `kunlun` platform 和 plugin；
- `python/sglang/srt/platforms/__init__.py:71-84` 按 `SGLANG_PLATFORM` 查找 entry
  point，找不到时直接报错；
- `sglang-kunlun/sitecustomize.py:49-57` 由 `SGLANG_PLATFORM=kunlun` 或
  `SGLANG_USE_XPU=1` 激活 Kunlun pre-shim；
- `sglang-kunlun/README.md:5-18` 给出固定仓库的导入顺序及三项环境设置；
- `python/sglang/srt/environ.py:518` 定义 FlashInfer 可用性开关。

Agent 还要记录：

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

以下是固定源码中可以核对的候选项，不是统一启动模板。Agent 只有在当前调用路径或
错误证据涉及该分支时才设置；每次都记录原值、新值和验证结果。

| 环境变量 | 固定源码语义 | 何时允许改 | 固定 commit 源码锚点 |
|---|---|---|---|
| `SGLANG_HEALTH_CHECK_TIMEOUT` | 未设置时为 20 秒，按整数读取 | 只有当前启动日志证明 health 检查超时才提高 | `python/sglang/srt/entrypoints/http_server.py:181-183` |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` | 布尔开关，默认开启 | 为正确性复现关闭重叠，并用同一请求确认影响 | `python/sglang/srt/environ.py:843-846`；`python/sglang/srt/models/deepseek_v4.py:358` |
| `USE_FAST_ALLOC_EXTEND_KUNLUN` | 除字符串 `0` 外都选择 fast allocator | 只有 allocator 路径相关失败时设为 `0` 做对照 | `sglang-kunlun/sglang_kunlun/hooks/mem_cache/kunlun_allocator.py:80-83` |
| `SGLANG_OPT_BF16_FP32_GEMM_ALGO` | 默认 `cublas`；值为 `deep_gemm` 时走 DeepGEMM，其余值走 `torch.mm` | 只有实际命中该 BF16/FP32 GEMM 时选择并记录 | `python/sglang/srt/environ.py:836-841`；`python/sglang/jit_kernel/dsv4/gemm.py:13-24` |
| `SGLANG_OPT_SWIGLU_CLAMP_FUSION` | 布尔开关，默认开启，并决定 clamp fusion 分支 | 验证 SwiGLU clamp 生产路由时可设 `0/1` 对照 | `python/sglang/srt/environ.py:836-841`；`python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:568-584` |
| `SGLANG_OPT_USE_FUSED_HASH_TOPK` | 布尔开关，默认开启 | 当前模型命中 hash TopK 时可设 `0` 验证 fallback | `python/sglang/srt/environ.py:824-826`；`python/sglang/srt/layers/moe/hash_topk.py:178-185` |
| `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK` | 布尔开关，默认开启，并选择 biased TopK 实现 | 验证 TopK 生产路由时可设 `0/1` 对照 | `python/sglang/srt/environ.py:824-826`；`python/sglang/srt/layers/moe/topk.py:1733-1744` |
| `SGLANG_PREP_IN_CUDA_GRAPH` | 布尔开关，默认开启 | 仅当 eager 参数仍触发 graph preparation 时设为 `0` | `python/sglang/srt/environ.py:843-849`；`python/sglang/srt/layers/attention/deepseek_v4_backend.py:571` |

没有固定源码锚点或当前现场证据的 D/P 节点数值、XSHMEM、BKCL、RDMA 等配置不在
本文预设。确实需要而 Agent 无法从节点和启动脚本确认时，把它记录为 `ENVIRONMENT`
BUG，按 Repair Loop 尝试可验证的范围内方案，不能猜值。

## 4. 环境报错的判定方式

除 platform discovery 的报错外，下列字符串本身不能证明根因。Agent 先保存当前
traceback 和导入路径，再按证据分类，不能把历史经验直接写成当前结论。

| 报错或症状 | 当前现场必须补的证据 | 可以得出的结论 |
|---|---|---|
| `ModuleNotFoundError: No module named 'sglang_kunlun'` | `importlib.util.find_spec("sglang_kunlun")`、解释器路径、editable install 元数据 | 只能先记 `ENVIRONMENT`；确认包不可见后再修安装或 `PYTHONPATH` |
| `SGLANG_PLATFORM='kunlun' not found` | platform entry point 列表和 `sglang.__file__` | 固定源码在 `python/sglang/srt/platforms/__init__.py:79-84` 明确要求注册 entry point |
| `No module named 'sglang.srt.plugins'` | `sglang.__file__`、`sys.path`、完整 import traceback | 只有导入位置不在固定 worktree 时才能归因于路径优先级 |
| `No module named 'triton.tools.tensor_descriptor'` | 首个导入该模块的固定源码位置、实际 backend 和依赖版本 | 只能说明依赖导入失败；不能在未确认调用路径时声称算子缺失 |
| `Torch not compiled with XPU enabled` | Kunlun platform 实例、Torch build/version、设备探测命令 | 只能先记设备环境失败；不能仅凭字符串断言是 plugin 或 Torch 的单一问题 |

这些报错都发生在确认 P800 Production Operator 成功执行之前，因此不能记为
`OPERATOR_MISSING`。

## 5. 代码修复规则

- 环境修复不能顺带修改算子语义。
- 简单算子修复保留原生产调用位置，不为测试新增生产 helper。
- 修改后先运行聚焦 CPU reference 与 P800 生产算子测试，再启动真实模型。
- 某个假设需要新增 C++、自定义 kernel 或底层注册时，将该 attempt 记为失败并继续
  寻找范围内方案；同一 BUG 达到 Contract 的 3 次上限后才进入 `BLOCKED`。

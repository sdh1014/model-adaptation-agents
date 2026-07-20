# 修正 revision 6 preflight 的固定源解析：按 seam 模块而非 kernel 定义模块

Type: bugfix
Status: fixed
Blocked by: 28

## 背景

在固定 CUDA revision（SGLang `49e384ce…`）的 TP8 环境执行

```
capture_golden.py --mode preflight-session --scan-result runs/scan-007/result.json
```

时，多 collector preflight worker 在第二个算子 `sgl_kernel.gemma_rmsnorm`
处以 `returncode=2` 失败：

```
loaded source /usr/local/lib/python3.12/dist-packages/sgl_kernel/elementwise.py
for sgl_kernel.gemma_rmsnorm
does not match fixed source /ssd1/sdh/sglang/python/sglang/srt/layers/layernorm.py
```

失败 Run 保留在 `runs/cuda-preflight-r6-001`（`PREFLIGHT_FAILED`，
`consumes_capture_session=false`），已作为历史证据提交到
`evidence/cuda-preflight-r6-001`。

## 根因

`scan-007` 与 `capture_golden.py` 对每个算子固定的 `capture_module_path` 是
**SGLang 侧调用它的源文件**（即 HookRegistry 打补丁的 seam 所在模块）：

| operator_id | hook_target（seam） | capture_module_path |
|---|---|---|
| `sglang…fused_moe._swiglu_silu_clamp_mul` | 同名 | fused_moe.py |
| `sgl_kernel.gemma_rmsnorm` | `sglang.srt.layers.layernorm.gemma_rmsnorm` | layernorm.py |
| `sgl_kernel.gemma_fused_add_rmsnorm` | `sglang.srt.layers.layernorm.gemma_fused_add_rmsnorm` | layernorm.py |
| `sgl_kernel.topk_sigmoid` | `sglang.srt.layers.moe.topk.topk_sigmoid` | topk.py |
| `sglang…prefill_attention._fwd_kernel` | `sglang.srt.layers.attention.triton_ops.prefill_attention.context_attention_fwd` | prefill_attention.py |

但 preflight worker 的固定源校验用的是**被调用对象的定义模块**：

```python
call = calls[item["operator_id"]]
loaded_source = Path(sys.modules[call.__module__].__file__).resolve()
```

`sglang.srt.layers.layernorm` 在 CUDA 分支里是
`from sgl_kernel import (… gemma_rmsnorm, gemma_fused_add_rmsnorm, …)`
的 **re-export**（见 `python/sglang/srt/layers/layernorm.py:89`）。因此
`gemma_rmsnorm.__module__` 指向已编译的 `sgl_kernel` 包（`elementwise.py`），
与固定的 seam 源文件 layernorm.py 不一致，直接中止。

`_swiglu_silu_clamp_mul` 与 `context_attention_fwd` 在各自 SGLang 模块内定义，
`__module__` 恰好等于 seam 模块，所以只有 re-export 的三个 `sgl_kernel.*`
（gemma_rmsnorm、gemma_fused_add_rmsnorm、topk_sigmoid）会命中此问题。

## 修复

按用户决定“使用 SGLang 侧调用它的 layernorm.py 做 preflight”，把固定源解析
改为从 **hook_target 的 seam 模块**取源文件，而不是被调用对象的定义模块：

```python
seam_module = item["hook_target"].rsplit(".", 1)[0]
loaded_source = Path(sys.modules[seam_module].__file__).resolve()
```

改动仅限 `model-adaptation/scripts/model_adaptation_capture/preflight.py` 的
多 collector session 校验分支（原 `calls` 映射及其未用的 operator-id 导入一并
删除）。语义保持不变：仍校验将被 Hook / self-replay 的调用来自固定 SGLang
worktree 的 seam 源文件，并配合 `capture_module_sha256` 校验 seam 文件字节。
单算子 SwiGLU（`preflight.py` 另一分支）与历史 MLP 分支使用本地定义的 callable，
`__module__` 已等于 seam 模块，不受影响，未改动。

## 影响与边界

- 不改 Contract Data；`contract_revision` 仍为 6。仅修工具的固定源解析 bug。
- 不新增 helper / 自定义算子 / wrapper；seam 仍是源码里已存在的 call-site。
- `capture_module_path` / `capture_module_sha256` 的固定值不变，仍指向 layernorm.py
  等 SGLang 侧文件。
- 失败 Run `runs/cuda-preflight-r6-001` 保留为历史；修复后的重跑使用新 Run
  `runs/cuda-preflight-r6-002`，不删除、不复用旧名。
- SOURCE 单测全部通过（ticket12/25/26/28 相关用例）。

## 验证

修复后重跑 `capture_golden.py --mode preflight-session`，五个算子每个三种
shape、rank 过滤与逐算子 CUDA self-replay 全部通过，`PREFLIGHT_PASSED`，
`consumes_capture_session=false`，证据保存在 `runs/cuda-preflight-r6-002`。

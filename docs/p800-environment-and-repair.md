# P800 环境校正与 SwiGLU-clamp 修复记录

本文件记录 Step-3.7-Flash → KLX P800 迁移中，为让「修复回放适配器」能导入被修复的
Kernel Call 边界所做的**环境校正过程**，以及对 `sglang-kunlun` 源码的**具体修改**。

之所以用文档而非直接推送整个 sglang worktree：worktree（`/workspace/baidu/aicapx/sglang`，
固定在 revision `546ad8c682392922792bbbfe53a8bf575545f118`）体积大且不属于本仓；按 spec，
通过的补丁应保持**未提交**留在 worktree 里。本文件 + `runs/repair-attempt-1-r5-001/candidate.patch`
足以在一台干净的机器上复现该修复。

---

## 1. 环境校正过程（import-path blocker）

目标：让回放 worker 能执行
`importlib.import_module("sglang_kunlun.hooks.layers.quantization.unquant")`
并取到被修复的可调用对象 `apply_gemm1_swiglu_clamp`。

导入该模块会触发 `hooks/layers/__init__` 的 eager MoE 链
（sglang token_dispatcher → flashinfer → triton），暴露出一连串**与算子无关的、
预先存在的环境不兼容**。未修改的 `unquant.py` 单独导入同样失败，可证明这些问题不是本次修改引入的。

按发现顺序，逐层的报错与修复：

| # | 报错 | 根因 | 修复（仅改启动环境，不动 flow 代码 / worktree 源码） |
|---|------|------|------|
| 1 | `ModuleNotFoundError: No module named 'sglang_kunlun'` | 包未安装、也不在 PYTHONPATH | `pip install -e /workspace/baidu/aicapx/sglang/sglang-kunlun --no-deps --no-build-isolation`（可逆：`pip uninstall sglang-kunlun`） |
| 2 | `RuntimeError: SGLANG_PLATFORM='kunlun' not found in discovered platform plugins` | 平台入口点未注册 | 同上的 editable install 会注册入口点 `sglang.srt.platforms: kunlun=sglang_kunlun.platform:activate`；并 `export SGLANG_PLATFORM=kunlun SGLANG_USE_XPU=1` |
| 3 | `No module named 'sglang.srt.plugins'` | site-packages 里的 sglang 缺 `srt.plugins`，被优先导入 | 把 `<worktree>/python` 放到 PYTHONPATH 最前，让 worktree 版 sglang 覆盖 site-packages 版。**回放工具已自动做这件事**（`replay_compare.py` 的 `_kernel_worker_environment` 把 `<worktree>/python` 前置进 PYTHONPATH） |
| 4 | `ModuleNotFoundError: No module named 'triton.tools.tensor_descriptor'` | flashinfer 需要 triton ≥ 3.1，环境是 triton 3.0.0 | `export SGLANG_IS_FLASHINFER_AVAILABLE=false` → `is_flashinfer_available()`（`python/sglang/srt/utils/common.py:337`）返回 False，跳过损坏的 flashinfer 导入 |
| 5 | `AssertionError: Torch not compiled with XPU enabled` | 只在 kunlun 平台未激活时出现 | 一旦 #1/#2 正确激活 kunlun 平台 + #4 关掉 flashinfer 后即消失 |

### 复现步骤（干净机器上，激活 P800 环境后）

conda 环境：`python310_torch29_cuda`（torch `2.9.0+cu129`，8×CUDA，triton 3.0.0，`kunlun_ops.swiglu` 可用）。

```bash
# 1. editable 安装 sglang-kunlun（注册 kunlun 平台入口点 + 使包可导入）
pip install -e /workspace/baidu/aicapx/sglang/sglang-kunlun --no-deps --no-build-isolation

# 2. 调用回放前，在 shell 里导出这些环境变量
export SGLANG_PLATFORM=kunlun
export SGLANG_USE_XPU=1
export SGLANG_IS_FLASHINFER_AVAILABLE=false
# （PYTHONPATH 里 <worktree>/python 前置由 replay_compare.py 自动完成，无需手动设）
```

验证：

```bash
python -c "from sglang_kunlun.hooks.layers.quantization import unquant; \
print(unquant.apply_gemm1_swiglu_clamp)"
```

worker 通过 `os.environ.copy()` 继承这些变量（见 `replay_compare.py:_kernel_worker_environment`），
因此在 shell 里 export 即可传播到子进程，无需改动任何工具代码。

---

## 2. sglang-kunlun 源码修改（SwiGLU-clamp 修复）

### 差距（gap）

- CUDA 参考实现 `_swiglu_silu_clamp_mul(x, gemm1_limit)`
  （`python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:327`）：
  SiLU 门控 SwiGLU，**门控半段 clamp 到 `max=gemm1_limit`**，
  **线性半段 clamp 到 `[-gemm1_limit, gemm1_limit]`**。
- Kunlun 基线调用 `kunlun_ops.swiglu(x=y, y=out1)`，**忽略了 gemm1_limit → 缺少 clamp**
  → 3 个 Golden shape 里有 2 个精度不过（绝对误差 95.5 与 8.75）。

### 最小修复

已确认 `kunlun_ops.swiglu` 本身接受可选的 `limit=<float>` kwarg：传入后门控半段
clamp 到 `max=limit`、线性半段 clamp 到 `[-limit, limit]` —— 经**设备上验证**与 CUDA
`_swiglu_silu_clamp_mul` 完全一致（atol 0.01 / rtol 0.02，2 行 bf16 探针）。不传 `limit`
则退化为无 clamp 的 SwiGLU。故最小忠实修复就是把 `limit=` 传给已有的 swiglu 调用 —— 不新增算子。

`gemm1_limit` 来源：`layer.moe_runner_config.gemm1_clamp_limit`
（`MoeRunnerConfig.gemm1_clamp_limit: Optional[float] = None`；CUDA 的 `fused_moe.py`
把同一字段传进 `_swiglu_silu_clamp_mul`）。

### 修改位置（唯一允许改动的路径 / ALLOWED_PATH）

`sglang-kunlun/sglang_kunlun/hooks/layers/quantization/unquant.py`
（在 worktree `/workspace/baidu/aicapx/sglang` 内，固定 revision `546ad8c6...`）。

将 SwiGLU-clamp 段抽成一个**生产路径使用的模块级可调用对象**（签名 `(x, gemm1_limit) -> out`），
以便回放适配器能 import 并调用；forward 改为传入
`layer.moe_runner_config.gemm1_clamp_limit`。

完整 diff（与 `runs/repair-attempt-1-r5-001/candidate.patch` 一致）：

```diff
diff --git a/sglang-kunlun/sglang_kunlun/hooks/layers/quantization/unquant.py b/sglang-kunlun/sglang_kunlun/hooks/layers/quantization/unquant.py
index a2054be4c..9ffbd993b 100644
--- a/sglang-kunlun/sglang_kunlun/hooks/layers/quantization/unquant.py
+++ b/sglang-kunlun/sglang_kunlun/hooks/layers/quantization/unquant.py
@@ -29,6 +29,28 @@ if TYPE_CHECKING:  # pragma: no cover
     )


+def apply_gemm1_swiglu_clamp(
+    x: torch.Tensor,
+    gemm1_limit: "float | None",
+) -> torch.Tensor:
+    """SwiGLU activation with the gemm1 clamp used by the fused-MoE path.
+
+    Mirrors the CUDA ``_swiglu_silu_clamp_mul`` reference: SiLU-gated SwiGLU
+    with the gate clamped to ``max=gemm1_limit`` and the linear half clamped to
+    ``[-gemm1_limit, gemm1_limit]``. ``kunlun_ops.swiglu`` applies the same
+    clamp when given ``limit=``; without a limit it is the un-clamped SwiGLU.
+    """
+    import kunlun_ops  # local-only dep
+
+    d = x.shape[-1] // 2
+    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
+    if gemm1_limit is None:
+        kunlun_ops.swiglu(x=x, y=out)
+    else:
+        kunlun_ops.swiglu(x=x, y=out, limit=float(gemm1_limit))
+    return out
+
+
 @plugin_hook(
     "sglang.srt.layers.quantization.unquant.UnquantizedFusedMoEMethod.apply",
     type=HookType.REPLACE,
@@ -119,9 +141,7 @@ def unquantized_fused_moe_apply_kunlun(
         act=None,
     )

-    d = y.shape[-1] // 2
-    out1 = torch.empty(y.shape[:-1] + (d,), dtype=y.dtype, device=y.device)
-    kunlun_ops.swiglu(x=y, y=out1)
+    out1 = apply_gemm1_swiglu_clamp(y, layer.moe_runner_config.gemm1_clamp_limit)
     out1 = out1.reshape(-1, out1.shape[-1])

     out = cache[: M * top_k * layer.w2_weight.shape[1]].view(
```

### 在干净机器上重新应用

```bash
cd /workspace/baidu/aicapx/sglang
git checkout 546ad8c682392922792bbbfe53a8bf575545f118
git apply /workspace/model-adaptation-agents/runs/repair-attempt-1-r5-001/candidate.patch
# 该补丁按 spec 保持未提交，作为 worktree 里唯一的工作区改动
```

---

## 3. 结果

- attempt 1 一次通过：3/3 Golden shape 全过固定精度门（atol 0.01 / rtol 0.02），0 失败。
- Working State 推进到 rev 34，`status: PASS`，`next_action: none`。
- 通过的回放 Run：`runs/repair-attempt-1-r5-001/`（`workspace_state: PATCH_RETAINED`）；
  经适配器的回放：`runs/repair-attempt-1-r5-001/replay/`
  （`invocation_target: repair-kernel-call/v1:sglang_kunlun.hooks.layers.quantization.unquant:apply_gemm1_swiglu_clamp`）。
- 全量测试：136 passed / 1 skipped。
- Contract Data 哈希未变：`f683ea62427279da3c899a3ef0afcfa1d16298b7424154664c7e3fac9b13d938`。

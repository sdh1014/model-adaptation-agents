# CUDA preflight 与证据回传

## 当前边界

Contract revision 5 和 `scan-006` 选择的活动调用是：

```text
sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul
```

Ticket 23 已实现对应 adapter。preflight 只验证现有 SGLang Hook、rank 0 三种 shape
格式和新进程 CUDA self-replay；不加载 checkpoint，不启动正式模型，也不消耗唯一
CUDA Capture Session。

`runs/cuda-preflight-r5-001` 已验证 adapter-001。随后 `adapter-002` 增加样本摘要
与 replay 证据绑定，新的 `runs/cuda-preflight-r5-002` 也已回传并通过。旧 Run
只作历史；当前源码由 r5-002 解锁 Ticket 15。以下步骤保留为这份证据的复现记录，
不得覆盖已经封存的 Run。

不要使用 revision 4 的 `Step3p5MLP.forward` 命令，不要在本地 Mac 运行本步骤。

## 1. 在 CUDA 机器拉取实现

以下命令都在 `model-adaptation-agents` 仓库根目录执行：

```bash
git fetch origin
git switch chore/step3p7-p800-baseline
git pull --ff-only origin chore/step3p7-p800-baseline
git status --short
```

`git status --short` 必须没有输出。

设置已有 SGLang 0.5.14 源码路径，不要重新下载：

```bash
export SGLANG_WORKTREE=/替换为已有的/sglang-0.5.14路径
export PREFLIGHT_RUN=runs/cuda-preflight-r5-002
export OPERATOR_ID=sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul
```

检查固定源码：

```bash
test -n "${SGLANG_WORKTREE:-}"
test -f "$SGLANG_WORKTREE/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py"
test "$(git -C "$SGLANG_WORKTREE" rev-parse HEAD)" = "49e384ce9d304648e9959666ecb8ce8cd98d0deb"
test ! -e "$PREFLIGHT_RUN"
```

任何一条失败都停止，不要换成别的 revision，也不要覆盖已有 Run。

## 2. 确认 CUDA Python 并安装采集插件

使用实际运行 SGLang 的同一个 Python：

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available(), "当前 Python 看不到 CUDA"
print("torch_version=", torch.__version__)
print("cuda_device=", torch.cuda.get_device_name(torch.cuda.current_device()))
PY
```

安装仓库中的轻量插件：

```bash
python -m pip install -e ./model-adaptation --no-deps --no-build-isolation
```

先运行不会启动 CUDA preflight 的 adapter 测试：

```bash
PYTHONPYCACHEPREFIX=/tmp/model-adaptation-agents-pyc \
python -m unittest \
  tests.test_ticket23_swiglu_clamp_adapter \
  tests.test_ticket23_kernel_call_runtime \
  tests.test_ticket23_kernel_replay \
  tests.test_ticket12_sglang_capture_plugin
```

测试必须全部通过。其中
`test_latest_adapter_run_records_source_only_hardening` 会检查
`runs/adapter-002/result.json` 中的源码 SHA 与当前 checkout 完全一致。

## 3. 执行当前源码的 revision 5 preflight

```bash
python model-adaptation/scripts/capture_golden.py \
  --mode preflight \
  --spec migration-spec.md \
  --run-dir "$PREFLIGHT_RUN" \
  --scan-result runs/scan-006/result.json \
  --operator-id "$OPERATOR_ID" \
  --sglang-worktree "$SGLANG_WORKTREE"
```

该命令会启动三个新进程：

1. rank 0 通过真实 `HookRegistry` 包裹已有 `_swiglu_silu_clamp_mul`，合成三种
   BF16 shape，再验证重复 shape 和第四种 shape 不落 Tensor；
2. rank 1 通过同一 Hook 调用，并验证 rank 0 的 state 和三份样本完全未改变；
3. 重新加载三份样本，直接调用同一个 CUDA 函数，并用固定
   `torch.testing.assert_close(atol=0.01, rtol=0.02)` 比较。

## 4. 本机验收结果

```bash
PREFLIGHT_RUN="$PREFLIGHT_RUN" python - <<'PY'
import json
import hashlib
import os
from pathlib import Path

import torch

run = Path(os.environ["PREFLIGHT_RUN"])
result = json.loads((run / "result.json").read_text())
state = json.loads((run / "capture-state.json").read_text())
summary = json.loads((run / "preflight-capture-summary.json").read_text())
rank_filter = json.loads(
    (run / "preflight-rank-filter-summary.json").read_text()
)
replay = json.loads((run / "worker-result.json").read_text())
replay_config = json.loads(
    (run / "preflight-replay-config.json").read_text()
)
sample_files_path = run / "sample-files.json"
sample_files_sha256 = hashlib.sha256(
    sample_files_path.read_bytes()
).hexdigest()
operator = (
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
    "_swiglu_silu_clamp_mul"
)

assert result["passed"] is True
assert result["capture_status"] == "PREFLIGHT_PASSED"
assert result["consumes_capture_session"] is False
assert result["operator_id"] == operator
assert state["operator_id"] == operator
assert state["tp_rank"] == 0
assert state["tensor_parallel_size"] == 8
assert state["saved_shape_count"] == 3
assert state["repeated_call_count"] == 1
assert state["skipped_call_count"] == 1
assert summary["applied_hooks"] == [operator]
assert summary["checkpoint_loaded"] is False
assert rank_filter["applied_hooks"] == [operator]
assert rank_filter["runtime_tp_rank"] == 1
assert rank_filter["capture_tp_rank"] == 0
assert rank_filter["hook_call_count"] == 5
assert replay["passed"] is True
assert replay["invocation_target"] == operator
assert replay["checked_shape_count"] == 3
assert replay["actual_tensors_saved"] is False
assert replay["sample_files_sha256"] == sample_files_sha256
assert replay_config["sample_files_sha256"] == sample_files_sha256

samples = sorted((run / "samples").glob("*.pt"))
assert len(samples) == 3
for path in samples:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload["inputs"]) == {"x"}
    assert payload["parameters"] == {}
    assert set(payload["non_tensor_args"]) == {"gemm1_limit"}
    assert set(payload["outputs"]) == {"output"}
    assert payload["inputs"]["x"].dtype == torch.bfloat16
    assert payload["outputs"]["output"].dtype == torch.bfloat16

print("CUDA revision 5 adapter-002 preflight: PASS")
PY
```

如果这里失败，保留整个 Run，不要删除、覆盖或开始正式 Capture。

## 5. 通过 GitHub 回传

preflight 通过或失败都创建独立 evidence 分支。若分支名已经存在，把末尾编号加一：

```bash
git switch -c evidence/cuda-preflight-r5-002
git add "$PREFLIGHT_RUN"
git commit -m "evidence: add adapter-002 CUDA preflight"
git push -u origin evidence/cuda-preflight-r5-002
```

回传以下信息：

```text
evidence branch: evidence/cuda-preflight-r5-002
preflight run: runs/cuda-preflight-r5-002
result: PASS 或 FAIL
```

整个 Run 都要提交，其中包括三份 `.pt` 样本、`result.json`、`capture-state.json`、
三份 config、三份 log、两个 capture summary、`sample-files.json` 和 worker
result。preflight 样本很小，不需要把 checkpoint 提交到 GitHub。

推送完成后停止。不要继续正式模型 Capture；先让当前实现会话检查 evidence。

## preflight 通过后的正式 Session 边界

`cuda-preflight-r5-002` 已回传并通过当前源码校验，因此这个前置条件已经满足。
正式 Session 仍须使用固定 checkpoint、TP8、BF16 和 eager，
不传量化、投机解码、MTP、attention backend 或 MoE backend 参数。采集插件只增加
环境变量，不改变两端模型启动参数。真实文本请求触发第 43、44 层后，保存最多三个
rank 0 shape，并通过 `replay_compare.py --mode kernel-replay
--execution-site cuda` 完成 CUDA self-replay。

正式 Session 一旦消耗，不得重新启动第二次采集。

# Ticket 15：唯一正式 CUDA Capture Session

> **采集已经完成，不要再次执行。** 人已明确模型加载前、未进入 Hook 且没有产生
> 样本的失败不计入 Capture Session。PID `114828` 是唯一实际采集 Session，
> `runs/cuda-golden-r5-001` 已接受并封存。下一步只执行
> `model-adaptation/references/formal-cuda-handoff.md`，不得重启模型。

## 这一步会做什么

本页只在 CUDA 机器执行。它会启动一次真实的 Step-3.7-Flash TP8 服务，采集当前
活动 Kernel Call 的 rank 0 输入与 CUDA 输出，然后记录样本摘要并完成 CUDA
self-replay。

这次启动就是唯一正式 Session。`session-start.json` 成功推到固定远端 evidence
分支后，无论后续成功还是失败，都不得创建第二个正式 Session。失败时也向同一分支
追加已有证据；不要删除后换一个 Run 名或分支名重试。

这次回传后先停止。Agent 要核对真实 Golden，再生成包含
`WAITING / HANDOFF` 状态的临时 Spec。当前阶段不要构建 bundle，也不要自行编写或
修改下一状态 Spec。

## 1. 拉取唯一执行分支

以下命令都在 `model-adaptation-agents` 仓库根目录执行：

```bash
set -euo pipefail
git fetch origin
git switch chore/step3p7-p800-baseline
git pull --ff-only origin chore/step3p7-p800-baseline
WORKTREE_STATUS="$(git status --short)"
test -z "$WORKTREE_STATUS"
unset WORKTREE_STATUS
```

上述 clean check 必须通过。然后设置固定路径和 Run 名：

```bash
export SGLANG_WORKTREE=/ssd1/sdh/sglang
export SESSION_RUN=runs/cuda-formal-session-r5-001
export GOLDEN_RUN=runs/cuda-golden-r5-001
export RECORD_RUN=runs/cuda-golden-r5-001-sample-record
export MODEL_ID=stepfun-ai/Step-3.7-Flash
export MODEL_REVISION=5f6244077ac62e04eec3f320501ff8c2b293373a
export OPERATOR_ID=sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul
export BASE_URL=http://127.0.0.1:30000
export EVIDENCE_BRANCH=evidence/cuda-formal-capture-r5-001
```

如果实际 SGLang 路径不同，只修改 `SGLANG_WORKTREE`。其他值不要改。

确认固定源码和三个全新目录：

```bash
test -n "${SGLANG_WORKTREE:-}"
test "$(git -C "$SGLANG_WORKTREE" rev-parse HEAD)" = \
  "49e384ce9d304648e9959666ecb8ce8cd98d0deb"
test -f "$SGLANG_WORKTREE/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py"
test ! -e "$SESSION_RUN"
test ! -e "$GOLDEN_RUN"
test ! -e "$RECORD_RUN"
REMOTE_EVIDENCE="$(git ls-remote --heads origin "refs/heads/$EVIDENCE_BRANCH")"
test -z "$REMOTE_EVIDENCE"
unset REMOTE_EVIDENCE
```

任一检查失败都停止；此时还没有消耗 Session。

## 2. 核对四项前置证据

先安装当前仓库的采集插件。必须使用实际启动 SGLang 的同一个 Python：

```bash
python -m pip install -e ./model-adaptation --no-deps --no-build-isolation
```

用下面的只读检查确认当前 Contract、扫描、adapter 和最新 CUDA preflight：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

operator = (
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
    "_swiglu_silu_clamp_mul"
)
paths = {
    "binding": Path("runs/spec-binding-004/result.json"),
    "scan": Path("runs/scan-006/result.json"),
    "adapter": Path("runs/adapter-002/result.json"),
    "preflight": Path("runs/cuda-preflight-r5-002/result.json"),
}
data = {name: json.loads(path.read_text()) for name, path in paths.items()}
binding = data["binding"]["spec_binding"]

assert data["binding"]["passed"] is True
assert data["scan"]["scan_complete"] is True
assert data["scan"]["spec_binding"] == binding
assert data["scan"]["selection"]["active_operator"] == operator
assert data["adapter"]["passed"] is True
assert data["adapter"]["spec_binding"] == binding
assert data["adapter"]["active_operator"] == operator
assert data["preflight"]["passed"] is True
assert data["preflight"]["spec_binding"] == binding
assert data["preflight"]["operator_id"] == operator
assert data["preflight"]["capture_status"] == "PREFLIGHT_PASSED"
assert data["preflight"]["consumes_capture_session"] is False

for source in data["adapter"]["source_files"]:
    actual = hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
    assert actual == source["sha256"], source["path"]

print("formal CUDA prerequisites: PASS")
PY
```

再确认当前 Python 至少能看到八张 CUDA 卡，且 30000 端口尚未被占用：

```bash
python - <<'PY'
import socket

import torch

assert torch.cuda.is_available(), "当前 Python 看不到 CUDA"
assert torch.cuda.device_count() >= 8, torch.cuda.device_count()
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 30000))
print("cuda_devices=", torch.cuda.device_count())
print("torch_version=", torch.__version__)
print("port_30000=available")
PY
```

这里明确绑定：

- `runs/spec-binding-004/result.json`
- `runs/scan-006/result.json`
- `runs/adapter-002/result.json`
- `runs/cuda-preflight-r5-002/result.json`

检查失败时停止，不要启动模型。

## 3. 生成 Golden 配置

`prepare` 只生成配置，本身不消耗正式 Session：

```bash
python model-adaptation/scripts/capture_golden.py \
  --mode prepare \
  --spec migration-spec.md \
  --run-dir "$GOLDEN_RUN" \
  --scan-result runs/scan-006/result.json \
  --operator-id "$OPERATOR_ID" \
  --sglang-worktree "$SGLANG_WORKTREE"
```

命令只有在 Contract、Scan、活动调用、固定源码 revision、TP8/rank 0、BF16 和
最多三个 shape 全部匹配时才会成功；它还固定样本只包含
`x + gemm1_limit -> output`，参数 Tensor 列表为空。这些检查由
`capture_golden.py` 和采集插件完成，不在 runbook 重写。

## 4. 在 GitHub 原子占位唯一 Session

先创建 evidence 分支，再写入 Session 标记。标记会直接复用 Golden 配置中的
Spec 绑定，不另造一份 Contract 值：

```bash
git switch -c "$EVIDENCE_BRANCH"

SESSION_RUN="$SESSION_RUN" GOLDEN_RUN="$GOLDEN_RUN" \
OPERATOR_ID="$OPERATOR_ID" SGLANG_WORKTREE="$SGLANG_WORKTREE" \
python - <<'PY'
import json
import os
import subprocess
import uuid
from pathlib import Path

run = Path(os.environ["SESSION_RUN"])
run.mkdir(parents=True, exist_ok=False)
golden = Path(os.environ["GOLDEN_RUN"])
config = json.loads((golden / "capture-config.json").read_text())
value = {
    "tool": "migration-agent",
    "action": "reserve_formal_cuda_capture",
    "spec_binding": config["spec_binding"],
    "session_id": run.name,
    "reservation_id": uuid.uuid4().hex,
    "status": "ACTIVE",
    "consumes_capture_session": True,
    "golden_run": os.environ["GOLDEN_RUN"],
    "operator_id": os.environ["OPERATOR_ID"],
    "tool_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "sglang_revision": subprocess.check_output(
        ["git", "-C", os.environ["SGLANG_WORKTREE"], "rev-parse", "HEAD"],
        text=True,
    ).strip(),
}
(run / "session-start.json").write_text(
    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
print("唯一正式 Session 本地标记已生成，尚未授权启动模型。")
PY
```

把标记和 prepare 证据先推到固定远端分支：

```bash
git add "$SESSION_RUN/session-start.json" \
  "$GOLDEN_RUN/capture-config.json" \
  "$GOLDEN_RUN/capture.log" \
  "$GOLDEN_RUN/result.json"
git commit -m "evidence: reserve formal CUDA capture session"
RESERVATION_PUSH_OUTPUT="$(
  LC_ALL=C git push --porcelain -u origin \
    "${EVIDENCE_BRANCH}:${EVIDENCE_BRANCH}"
)"
printf '%s\n' "$RESERVATION_PUSH_OUTPUT"
EXPECTED_RESERVATION_LINE=$'*\t'"refs/heads/${EVIDENCE_BRANCH}:"\
"refs/heads/${EVIDENCE_BRANCH}"$'\t[new branch]'
printf '%s\n' "$RESERVATION_PUSH_OUTPUT" |
  grep -Fx "$EXPECTED_RESERVATION_LINE"
unset EXPECTED_RESERVATION_LINE
unset RESERVATION_PUSH_OUTPUT
```

只有输出包含精确的 `[new branch]` 行，才允许继续启动模型。普通的
`[up to date]` 即使退出码为 0 也会被拒绝，避免同一 checkout 重复执行。并发
checkout 的随机 `reservation_id` 不同，后到的 push 会因远端分支已存在而失败。
从此无论成功还是失败，都只向这个分支追加证据；它就是跨 checkout 可见的唯一
占位。检查失败时不要启动模型，核对远端同名分支并停止。

## 5. 启动真实 TP8/BF16/eager 服务

采集只增加两个环境变量，不修改模型调用。命令固定为 target-only、TP8、BF16，
并分别关闭 decode 和 prefill CUDA Graph。不要追加量化、投机解码、MTP、
attention backend 或 MoE backend 选项。

```bash
export MODEL_ADAPTATION_CAPTURE_CONFIG="$PWD/$GOLDEN_RUN/capture-config.json"
export SGLANG_PLUGINS=model_adaptation_capture
export PYTHONPATH="$PWD/model-adaptation/scripts:$SGLANG_WORKTREE/python${PYTHONPATH:+:$PYTHONPATH}"

python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --tp-size 8 \
  --dtype bfloat16 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --host 127.0.0.1 \
  --port 30000 \
  >"$SESSION_RUN/server.log" 2>&1 &
export SERVER_PID=$!
```

查看最近的启动日志，并探测服务：

```bash
tail -n 100 "$SESSION_RUN/server.log"
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "服务进程已经退出；进入失败处理。" >&2
  false
fi
if curl -fsS "$BASE_URL/health"; then
  echo "server_ready=true"
else
  echo "server_ready=false；等待后重复本节两条检查命令，不要重启服务。"
fi
```

`server_ready=true` 后才继续。若仍未就绪，等待后重复本节命令；不要因为启动慢而
另起第二个服务进程。若当前进程退出，直接进入“失败时怎么做”。

## 6. 保存经过筛选的实际运行参数

`/server_info` 会返回大量服务信息。先写到临时文件，只把本 Demo 允许公开且需要
核验的字段写入 `runtime-profile.json`；原始 `server_info` 可能包含无关配置，
不得提交。

```bash
export RAW_SERVER_INFO="$(mktemp)"
curl -fsS "$BASE_URL/server_info" >"$RAW_SERVER_INFO"

RAW_SERVER_INFO="$RAW_SERVER_INFO" SESSION_RUN="$SESSION_RUN" \
MODEL_ID="$MODEL_ID" MODEL_REVISION="$MODEL_REVISION" python - <<'PY'
import json
import os
from pathlib import Path

raw = json.loads(Path(os.environ["RAW_SERVER_INFO"]).read_text())
checks = {
    "model_path": os.environ["MODEL_ID"],
    "revision": os.environ["MODEL_REVISION"],
    "tp_size": 8,
    "dtype": "bfloat16",
    "quantization": None,
    "speculative_algorithm": None,
    "speculative_draft_model_path": None,
    "cuda_graph_backend_decode": "disabled",
    "cuda_graph_backend_prefill": "disabled",
}
for field, expected in checks.items():
    assert raw.get(field) == expected, (field, raw.get(field), expected)
assert isinstance(raw.get("attention_backend"), str) and raw["attention_backend"]
assert isinstance(raw.get("moe_runner_backend"), str) and raw["moe_runner_backend"]

profile = {
    **checks,
    "attention_backend": raw["attention_backend"],
    "moe_runner_backend_argument": raw["moe_runner_backend"],
    "sglang_version": raw.get("version"),
}
path = Path(os.environ["SESSION_RUN"]) / "runtime-profile.json"
path.write_text(
    json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(profile, ensure_ascii=False, indent=2))
PY

rm -f "$RAW_SERVER_INFO"
unset RAW_SERVER_INFO
```

这里的 `moe_runner_backend_argument` 只是 SGLang 参数值；默认可能仍显示 `auto`，
不能把它冒充成实际 runner。实际 runner 要等目标 Hook 真正产生样本后，再用固定
源码调用链确认。

## 7. 发出一条固定文本请求

当前活动调用在固定文本路径的 MoE 第 43、44 层可达。一条请求可以产生一个或多个
真实 shape；本 Demo 接受一到三种，不需要为了凑满三种而重启服务。

```bash
SESSION_RUN="$SESSION_RUN" python - <<'PY'
import json
import os
from pathlib import Path

request = {
    "text": "请用一句话说明模型迁移。",
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 8,
    },
}
endpoint = "/generate"
(Path(os.environ["SESSION_RUN"]) / "request.json").write_text(
    json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
print("request endpoint:", endpoint)
PY

curl -fsS \
  -H "Content-Type: application/json" \
  --data-binary "@$SESSION_RUN/request.json" \
  "$BASE_URL/generate" \
  >"$SESSION_RUN/response.json"
```

请求返回后立即停止服务，不再发送其他输入：

```bash
kill -INT "$SERVER_PID"
wait "$SERVER_PID" || true
unset SERVER_PID
unset MODEL_ADAPTATION_CAPTURE_CONFIG
unset SGLANG_PLUGINS
```

## 8. 先记录样本字节，再做 CUDA self-replay

顺序不能颠倒。先让现有工具核对 Golden state 仍为 `ACTIVE`、checkpoint、
TP8/rank 0、样本数一到三且没有 self-replay，再生成不可变的样本文件摘要：

```bash
python model-adaptation/scripts/handoff_bundle.py \
  --mode record-samples \
  --spec migration-spec.md \
  --run-dir "$RECORD_RUN" \
  --golden-run "$GOLDEN_RUN"
```

只有上一步成功，才能在新进程直接调用现有 CUDA Kernel Call：

```bash
python model-adaptation/scripts/replay_compare.py \
  --mode kernel-replay \
  --spec migration-spec.md \
  --run-dir "$GOLDEN_RUN/self-replay" \
  --scan-result runs/scan-006/result.json \
  --operator-id "$OPERATOR_ID" \
  --golden-run "$GOLDEN_RUN" \
  --execution-site cuda \
  --sglang-worktree "$SGLANG_WORKTREE"
```

核对全部样本通过，并封存 Session 结果：

```bash
SESSION_RUN="$SESSION_RUN" GOLDEN_RUN="$GOLDEN_RUN" \
RECORD_RUN="$RECORD_RUN" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

session = Path(os.environ["SESSION_RUN"])
golden = Path(os.environ["GOLDEN_RUN"])
record = Path(os.environ["RECORD_RUN"])
state = json.loads((golden / "capture-state.json").read_text())
replay = json.loads((golden / "self-replay" / "result.json").read_text())
record_result = json.loads((record / "result.json").read_text())
runtime = json.loads((session / "runtime-profile.json").read_text())
start = json.loads((session / "session-start.json").read_text())

assert state["status"] == "SEALED"
assert state["capture_closed"] is True
assert state["self_replay"]["passed"] is True
assert start["status"] == "ACTIVE"
assert state["operator_id"] == start["operator_id"]
assert state["tp_rank"] == 0
assert state["tensor_parallel_size"] == 8
assert 1 <= state["saved_shape_count"] <= 3
assert replay["passed"] is True
assert replay["checked_shape_count"] == state["saved_shape_count"]
assert replay["failed_shape_count"] == 0
assert replay["actual_tensors_saved"] is False
assert record_result["passed"] is True
assert record_result["recorded_before_self_replay"] is True
assert (
    state["self_replay"]["sample_files_sha256"]
    == replay["sample_files_sha256"]
    == record_result["sample_files_sha256"]
)

result = {
    "tool": "migration-agent",
    "action": "complete_formal_cuda_capture",
    "spec_binding": state["spec_binding"],
    "session_id": start["session_id"],
    "reservation_id": start["reservation_id"],
    "passed": True,
    "capture_status": "SEALED",
    "consumes_capture_session": True,
    "formal_session_consumed": True,
    "golden_run": os.environ["GOLDEN_RUN"],
    "sample_record_run": os.environ["RECORD_RUN"],
    "operator_id": state["operator_id"],
    "captured_shape_count": state["saved_shape_count"],
    "sample_files_sha256": replay["sample_files_sha256"],
    "cuda_self_replay_result_sha256": hashlib.sha256(
        (golden / "self-replay" / "result.json").read_bytes()
    ).hexdigest(),
    "runtime_profile": runtime,
    "actual_backends": {
        "attention": runtime["attention_backend"],
        "moe_runner": "triton",
        "moe_runner_evidence": {
            "kind": "captured-hook-plus-fixed-source",
            "captured_operator": state["operator_id"],
            "runner_source": (
                "python/sglang/srt/layers/moe/moe_runner/triton.py:"
                "TritonRunnerCore.run"
            ),
            "leaf_source": (
                "python/sglang/srt/layers/moe/moe_runner/triton_utils/"
                "fused_moe.py:_swiglu_silu_clamp_mul"
            ),
        },
    },
    "bundle_status": "NOT_BUILT",
    "next_action": "agent_review_before_bundle_spec",
}
(session / "result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
```

## 9. 通过 GitHub 回传并停止

先确认没有单个样本超过 GitHub 普通文件限制：

```bash
find "$GOLDEN_RUN/samples" -type f -size +95M -print
```

该命令必须没有输出。有输出时不要修改、压缩或重采样，直接报告文件路径和大小。

成功时向第 4 步已经占位的 evidence 分支追加三份完整 Run：

```bash
git add "$SESSION_RUN" "$GOLDEN_RUN" "$RECORD_RUN"
git commit -m "evidence: add formal CUDA golden capture"
git push origin "$EVIDENCE_BRANCH"
```

回传：

```text
evidence branch: evidence/cuda-formal-capture-r5-001
session run: runs/cuda-formal-session-r5-001
golden run: runs/cuda-golden-r5-001
record run: runs/cuda-golden-r5-001-sample-record
result: PASS
```

推送后停止。不要自行编写或修改下一状态 Spec，不要运行 bundle build，也不要开始
P800 baseline。Agent 核验这三份 Run 后，会生成 bundle 所需的临时 Spec 和第二段
CUDA 命令。

## 失败时怎么做

只要第 4 步的远端占位已经成功，任何失败都按“唯一 Session 已消耗”处理：

1. 如果服务进程仍在，先停止它；
2. 不删除或覆盖 `SESSION_RUN`、`GOLDEN_RUN`、`RECORD_RUN` 中已经存在的内容；
3. 把失败命令、退出码和错误摘要写入
   `runs/cuda-formal-session-r5-001/failure.txt`；
4. 在当前 `evidence/cuda-formal-capture-r5-001` 分支执行：

   ```bash
   git add "$SESSION_RUN" "$GOLDEN_RUN"
   if test -e "$RECORD_RUN"; then
     git add "$RECORD_RUN"
   fi
   git commit -m "evidence: record failed formal CUDA capture"
   git push origin "$EVIDENCE_BRANCH"
   ```

5. 停止，等待 Agent 把 Spec 置为有证据的 `BLOCKED / CUDA_CAPTURE`。

不得创建第二个正式 Session，也不得用 preflight 样本替换真实 Golden。

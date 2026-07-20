# Revision 6：正式 CUDA Capture、审查与 Handoff Bundle

> **尚未执行。** 本页只绑定 Contract revision 6 与
> `runs/scan-007/result.json`。旧 revision 5 runbook 和历史采集适配器验证均不能
> 代替本页的环境检查或正式 CUDA Golden Capture。

## 1. 拉取固定执行分支

以下命令都在 CUDA 机器的 `model-adaptation-agents` 仓库根目录执行：

```bash
set -euo pipefail
git fetch origin
git switch chore/step3p7-p800-baseline
git pull --ff-only origin chore/step3p7-p800-baseline
WORKTREE_STATUS="$(git status --short)"
test -z "$WORKTREE_STATUS"
unset WORKTREE_STATUS

export SGLANG_CUDA_WORKTREE=/替换为已有的/sglang-0.5.14路径
export SCAN_RESULT=runs/scan-007/result.json
export ENV_RUN="${ENV_RUN:-runs/cuda-environment-preflight-r6-001}"
export SESSION_RUN=runs/cuda-formal-session-r6-001
export RECORD_ROOT=runs/cuda-golden-record-r6-001
export PLAN_RUN=runs/handoff-plan-r6-001
export BUNDLE_SPEC="$PLAN_RUN/migration-spec.md"
export BUILD_RUN=runs/handoff-build-r6-001
export VERIFY_RUN=runs/handoff-verify-cuda-r6-001
export EVIDENCE_BRANCH=evidence/cuda-formal-capture-r6-001
export MODEL_ID=stepfun-ai/Step-3.7-Flash
export MODEL_REVISION=5f6244077ac62e04eec3f320501ff8c2b293373a
export BASE_URL=http://127.0.0.1:30000
```

确认固定源码、输入和所有输出位置都尚未存在：

```bash
test "$(git -C "$SGLANG_CUDA_WORKTREE" rev-parse HEAD)" = \
  "49e384ce9d304648e9959666ecb8ce8cd98d0deb"
test -f "$SGLANG_CUDA_WORKTREE/examples/assets/example_image.png"
test ! -e "$ENV_RUN"
test ! -e "$SESSION_RUN"
test ! -e "$RECORD_ROOT"
test ! -e "$PLAN_RUN"
test ! -e "$BUILD_RUN"
test ! -e "$VERIFY_RUN"
REMOTE_EVIDENCE="$(
  git ls-remote --heads origin "refs/heads/$EVIDENCE_BRANCH"
)"
test -z "$REMOTE_EVIDENCE"
unset REMOTE_EVIDENCE
```

任何检查失败都停止。只有文末环境失败恢复流程可以从 Working State 指定一个新的
不可变 `$ENV_RUN`；正式 `SESSION_RUN` 和 evidence 分支不得换名绕过。

## 2. 只做环境 preflight

使用实际启动 SGLang 的同一个 Python 安装当前采集插件：

```bash
python -m pip install -e ./model-adaptation --no-deps --no-build-isolation
```

环境 preflight 不读取 Scan 或算子，不 Hook、不保存 Tensor，也不 self-replay：

```bash
python model-adaptation/scripts/capture_golden.py \
  --spec migration-spec.md \
  --run-dir "$ENV_RUN" \
  --mode preflight \
  --sglang-worktree "$SGLANG_CUDA_WORKTREE"
```

显式检查业务结果；命令创建了结果文件不等于环境通过：

```bash
ENV_RUN="$ENV_RUN" python - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(
    (Path(os.environ["ENV_RUN"]) / "result.json").read_text()
)
assert result["action"] == "environment-preflight"
assert result["passed"] is True
assert result["preflight_status"] == "PREFLIGHT_PASSED"
assert result["consumes_capture_session"] is False
assert "operator_id" not in result
print("revision 6 CUDA environment preflight: PASS")
PY
```

失败时不要准备或占位正式 Session，必须执行文末“环境失败”处理，不能让
`set -e` 退出后仍把 Spec 留在旧状态。

## 3. 从 Scan 准备唯一 Session

`prepare-session` 从不可变 Scan Run 读取完整 gap queue 和固定请求，本身不启动
模型、不采集 Tensor，也不消耗正式 Session：

```bash
python model-adaptation/scripts/capture_golden.py \
  --spec migration-spec.md \
  --run-dir "$SESSION_RUN" \
  --mode prepare-session \
  --scan-result "$SCAN_RESULT" \
  --sglang-worktree "$SGLANG_CUDA_WORKTREE"
```

立即校验 Session 配置与 Scan 的 gap queue、capture plan 和请求顺序完全一致，同时
生成后处理使用的动态映射。这里不手写任何 operator id：

```bash
SESSION_RUN="$SESSION_RUN" RECORD_ROOT="$RECORD_ROOT" \
SCAN_RESULT="$SCAN_RESULT" python - <<'PY'
import json
import os
from pathlib import Path

session = Path(os.environ["SESSION_RUN"])
scan = json.loads(Path(os.environ["SCAN_RESULT"]).read_text())
config = json.loads((session / "capture-config.json").read_text())
result = json.loads((session / "result.json").read_text())

expected_ids = [
    item["operator_id"]
    for item in scan["gap_queue"]
    if item["verdict"] == "CAPTURE_REQUIRED"
]
operator_ids = [item["operator_id"] for item in config["operators"]]
request_modes = [item["input_mode"] for item in config["requests"]]
assert result["passed"] is True
assert result["action"] == "prepare-session"
assert result["operator_ids"] == expected_ids == operator_ids
assert result["request_modes"] == request_modes
assert request_modes == ["text-only", "single-image"]
assert len(config["operators"]) == len(scan["capture_plan"])
assert "preflight_tp_context" not in config
assert all(
    item["max_shapes"] == 3
    and item["tp_rank"] == 0
    and "preflight_tp_context" not in item
    for item in config["operators"]
)

record_root = Path(os.environ["RECORD_ROOT"])
rows = []
for ordinal, item in enumerate(config["operators"], start=1):
    rows.append(
        "\t".join(
            [
                f"{ordinal:02d}",
                item["operator_id"],
                item["run_dir"],
                str((record_root / f"{ordinal:02d}").resolve()),
            ]
        )
    )
(session / "operator-map.tsv").write_text("\n".join(rows) + "\n")
print("session operators=", len(operator_ids))
print("request modes=", request_modes)
PY
```

## 4. 在 GitHub 占位，然后才能启动模型

本步骤将生成 `$SESSION_RUN/session-start.json` 并首次推送固定 evidence 分支。
只有远端原子创建成功，才授权后续模型启动。

```bash
git switch -c "$EVIDENCE_BRANCH"

SESSION_RUN="$SESSION_RUN" ENV_RUN="$ENV_RUN" \
SCAN_RESULT="$SCAN_RESULT" \
SGLANG_CUDA_WORKTREE="$SGLANG_CUDA_WORKTREE" python - <<'PY'
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

session = Path(os.environ["SESSION_RUN"])
environment = Path(os.environ["ENV_RUN"])
scan = Path(os.environ["SCAN_RESULT"])
config_path = session / "capture-config.json"
config = json.loads(config_path.read_text())
environment_result = json.loads(
    (environment / "result.json").read_text()
)
assert environment_result["passed"] is True
value = {
    "tool": "migration-agent",
    "action": "reserve_formal_cuda_capture_session",
    "spec_binding": config["spec_binding"],
    "session_id": session.name,
    "reservation_id": uuid.uuid4().hex,
    "status": "ACTIVE",
    "reservation_active": True,
    "consumes_capture_session": False,
    "capture_session_consumed": False,
    "scan_result": os.environ["SCAN_RESULT"],
    "scan_result_sha256": hashlib.sha256(scan.read_bytes()).hexdigest(),
    "capture_config_sha256": hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest(),
    "environment_preflight_run": os.environ["ENV_RUN"],
    "environment_preflight_result_sha256": hashlib.sha256(
        (environment / "result.json").read_bytes()
    ).hexdigest(),
    "operator_ids": [
        item["operator_id"] for item in config["operators"]
    ],
    "request_modes": [
        item["input_mode"] for item in config["requests"]
    ],
    "tool_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "sglang_revision": subprocess.check_output(
        [
            "git",
            "-C",
            os.environ["SGLANG_CUDA_WORKTREE"],
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip(),
}
(session / "session-start.json").write_text(
    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY
```

`session-start.json` 已形成后，先把 Working State 同步到这个 reservation。
`execution_site` 在这里记录为 CUDA；此前若已有环境失败证据，revision 会在其基础上
继续递增：

```bash
SESSION_RUN="$SESSION_RUN" python - <<'PY'
import os
import re
from pathlib import Path


def field_value(text, name):
    matches = re.findall(
        rf"^- `{re.escape(name)}`: `([^`]*)`$",
        text,
        flags=re.MULTILINE,
    )
    assert len(matches) == 1, name
    return matches[0]


def set_field(text, name, value):
    pattern = rf"^- `{re.escape(name)}`: `[^`]*`$"
    replacement = f"- `{name}`: `{value}`"
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert count == 1, name
    return updated


spec_path = Path("migration-spec.md")
source = spec_path.read_text()
session_run = os.environ["SESSION_RUN"]
assert (Path(session_run) / "session-start.json").is_file()
current_revision = int(field_value(source, "state_revision"))
assert current_revision >= 49
assert field_value(source, "status") == "ACTIVE"
assert field_value(source, "phase") == "CUDA_CAPTURE"
assert field_value(source, "execution_site") in {"SOURCE", "CUDA"}
assert field_value(source, "capture_session_id") == "null"
assert field_value(source, "session_status") == "NOT_STARTED"

target_revision = current_revision + 1
updates = {
    "state_revision": str(target_revision),
    "execution_site": "CUDA",
    "last_completed_action": "revision_6_cuda_capture_session_reserved",
    "last_run": session_run,
    "next_action": (
        "在同一个 reservation、SESSION_RUN 和 evidence 分支启动一次 "
        "TP8/BF16/eager 模型，完成全部 Golden、self-replay 和 Handoff "
        "Bundle build/verify"
    ),
    "capture_session_id": session_run,
    "session_status": "ACTIVE",
}
for name, value in updates.items():
    source = set_field(source, name, value)

decision = (
    f"| `{target_revision}` | revision 6 正式 CUDA Session 已在 GitHub "
    "evidence 分支占位；尚未产生 Golden Sample，只允许在同一个 "
    "reservation 内继续 | "
    f"`{session_run}/session-start.json` |"
)
marker = "\n<!-- AGENT-WRITABLE WORKING STATE: END -->"
assert marker in source
source = source.replace(marker, "\n" + decision + "\n" + marker, 1)
temporary = spec_path.with_suffix(".md.next")
temporary.write_text(source)
os.replace(temporary, spec_path)
print("reserve_revision_6_working_state=", target_revision)
PY

git add migration-spec.md "$ENV_RUN" "$SESSION_RUN"
git diff --cached --check
git commit -m "evidence: reserve revision 6 CUDA capture session"
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

只有精确的 `[new branch]` 行可以继续；`[up to date]` 代表这个 reservation 已存在，
必须停止。后续成功、失败或未进入 Hook 的模型加载修正都只使用同一个 reservation、
`SESSION_RUN` 和 evidence 分支，绝不创建第二个。

## 5. 启动真实 TP8/BF16/eager 服务

每次启动都用当前 Working State revision 创建新目录。零样本失败后的同
reservation 重试会先递增 revision，因此不会覆盖之前已经提交的日志或请求证据：

```bash
LAUNCH_STATE_REVISION="$(
python - <<'PY'
import re
from pathlib import Path

source = Path("migration-spec.md").read_text()
matches = re.findall(
    r"^- `state_revision`: `([0-9]+)`$",
    source,
    flags=re.MULTILINE,
)
assert len(matches) == 1
print(matches[0])
PY
)"
export LAUNCH_STATE_REVISION
export LAUNCH_EVIDENCE_DIR="$SESSION_RUN/launch-$LAUNCH_STATE_REVISION"
test ! -e "$LAUNCH_EVIDENCE_DIR"
mkdir "$LAUNCH_EVIDENCE_DIR"

export MODEL_ADAPTATION_CAPTURE_CONFIG="$PWD/$SESSION_RUN/capture-config.json"
export SGLANG_PLUGINS=model_adaptation_capture
export PYTHONPATH="$PWD/model-adaptation/scripts:$SGLANG_CUDA_WORKTREE/python${PYTHONPATH:+:$PYTHONPATH}"

python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --trust-remote-code \
  --tp-size 8 \
  --dtype bfloat16 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --host 127.0.0.1 \
  --port 30000 \
  >"$LAUNCH_EVIDENCE_DIR/server.log" 2>&1 &
export SERVER_PID=$!
```

后续步骤必须继续使用这个进程，不得为补 shape 或补算子启动第二个服务。

等待这个进程健康；启动慢时只继续等待，不另起进程：

```bash
SERVER_READY=false
for _ in $(seq 1 120); do
  if curl -fsS "$BASE_URL/health" >/dev/null; then
    SERVER_READY=true
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 10
done
test "$SERVER_READY" = true
unset SERVER_READY
```

服务健康后，只保存本流程需要的运行信息：

```bash
curl -fsS "$BASE_URL/server_info" \
  >"$LAUNCH_EVIDENCE_DIR/server-info.raw.json"

LAUNCH_EVIDENCE_DIR="$LAUNCH_EVIDENCE_DIR" MODEL_ID="$MODEL_ID" \
MODEL_REVISION="$MODEL_REVISION" python - <<'PY'
import json
import os
from pathlib import Path

launch = Path(os.environ["LAUNCH_EVIDENCE_DIR"])
raw = json.loads((launch / "server-info.raw.json").read_text())
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
assert isinstance(raw.get("attention_backend"), str)
assert raw["attention_backend"]
assert isinstance(raw.get("moe_runner_backend"), str)
assert raw["moe_runner_backend"]
profile = {
    **checks,
    "attention_backend": raw["attention_backend"],
    "moe_runner_backend_argument": raw["moe_runner_backend"],
    "sglang_version": raw.get("version"),
}
(launch / "runtime-profile.json").write_text(
    json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY

rm "$LAUNCH_EVIDENCE_DIR/server-info.raw.json"
```

## 6. 发送 Session 配置中的固定文本和单图请求

请求内容只从 `config["requests"]` 读取。生成的两个文件必须是
`request-01.json` 和 `request-02.json`：

```bash
SESSION_RUN="$SESSION_RUN" \
LAUNCH_EVIDENCE_DIR="$LAUNCH_EVIDENCE_DIR" python - <<'PY'
import json
import os
from pathlib import Path

session = Path(os.environ["SESSION_RUN"])
launch = Path(os.environ["LAUNCH_EVIDENCE_DIR"])
config = json.loads((session / "capture-config.json").read_text())
requests = config["requests"]
assert [item["input_mode"] for item in requests] == [
    "text-only",
    "single-image",
]
for index, item in enumerate(requests, start=1):
    (launch / f"request-{index:02d}.json").write_text(
        json.dumps(
            item["request"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
PY

for request in \
  "$LAUNCH_EVIDENCE_DIR/request-01.json" \
  "$LAUNCH_EVIDENCE_DIR/request-02.json"
do
  response="${request/request-/response-}"
  curl -fsS \
    -H "Content-Type: application/json" \
    --data-binary "@$request" \
    "$BASE_URL/generate" \
    >"$response"
done
unset request response
```

两条请求都成功返回后停止服务，不再发送输入：

```bash
kill -INT "$SERVER_PID"
wait "$SERVER_PID" || true
unset SERVER_PID
unset MODEL_ADAPTATION_CAPTURE_CONFIG
unset SGLANG_PLUGINS
```

## 7. 对每个 Golden 先记录字节，再做 CUDA self-replay

后处理只读取 `operator-map.tsv`；算子增减由 Scan 和 Session 配置决定，不修改本
runbook。每个 Golden 都必须先 record，再在新进程中调用原 CUDA 路径：

```bash
mkdir "$RECORD_ROOT"

while IFS=$'\t' read -r ordinal operator_id golden_run record_run
do
  python model-adaptation/scripts/handoff_bundle.py \
    --mode record-samples \
    --spec migration-spec.md \
    --run-dir "$record_run" \
    --golden-run "$golden_run" \
    --scan-result "$SCAN_RESULT"

  python model-adaptation/scripts/replay_compare.py \
    --mode kernel-replay \
    --spec migration-spec.md \
    --run-dir "$golden_run/self-replay" \
    --scan-result "$SCAN_RESULT" \
    --operator-id "$operator_id" \
    --golden-run "$golden_run" \
    --execution-site cuda \
    --sglang-worktree "$SGLANG_CUDA_WORKTREE"
done <"$SESSION_RUN/operator-map.tsv"
unset ordinal operator_id golden_run record_run
```

生成正式 Session 结果前，逐项检查状态、样本数、摘要和共同采集进程：

```bash
SESSION_RUN="$SESSION_RUN" RECORD_ROOT="$RECORD_ROOT" \
SCAN_RESULT="$SCAN_RESULT" \
LAUNCH_EVIDENCE_DIR="$LAUNCH_EVIDENCE_DIR" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


session = Path(os.environ["SESSION_RUN"]).resolve()
record_root = Path(os.environ["RECORD_ROOT"]).resolve()
scan_path = Path(os.environ["SCAN_RESULT"]).resolve()
config_path = session / "capture-config.json"
config = json.loads(config_path.read_text())
scan = json.loads(scan_path.read_text())
start = json.loads((session / "session-start.json").read_text())
launch = Path(os.environ["LAUNCH_EVIDENCE_DIR"]).resolve()
assert launch.parent == session
runtime = json.loads((launch / "runtime-profile.json").read_text())
for index in (1, 2):
    assert (launch / f"request-{index:02d}.json").is_file()
    assert (launch / f"response-{index:02d}.json").is_file()

expected_ids = [
    item["operator_id"]
    for item in scan["gap_queue"]
    if item["verdict"] == "CAPTURE_REQUIRED"
]
assert [item["operator_id"] for item in config["operators"]] == expected_ids
assert start["operator_ids"] == expected_ids

process_ids = set()
goldens = []
for ordinal, item in enumerate(config["operators"], start=1):
    operator_id = item["operator_id"]
    golden = Path(item["run_dir"]).resolve()
    state_path = golden / "capture-state.json"
    state = json.loads(state_path.read_text())
    replay = json.loads((golden / "self-replay" / "result.json").read_text())
    record = json.loads(
        (record_root / f"{ordinal:02d}" / "result.json").read_text()
    )
    assert state["operator_id"] == operator_id
    assert state["status"] == "SEALED"
    assert state["capture_closed"] is True
    assert state["self_replay"]["passed"] is True
    assert state["tp_rank"] == 0
    assert state["tensor_parallel_size"] == 8
    assert 1 <= state["saved_shape_count"] <= 3
    assert state["saved_shape_count"] == len(state["samples"])
    assert state["capture_session_config"] == str(config_path)
    assert replay["operator_id"] == operator_id
    assert replay["passed"] is True
    assert replay["checked_shape_count"] == state["saved_shape_count"]
    assert replay["failed_shape_count"] == 0
    assert replay["actual_tensors_saved"] is False
    assert record["operator_id"] == operator_id
    assert record["passed"] is True
    assert record["recorded_before_self_replay"] is True
    assert (
        state["self_replay"]["sample_files_sha256"]
        == replay["sample_files_sha256"]
        == record["sample_files_sha256"]
    )
    process_ids.add(state["capture_process_id"])
    goldens.append(
        {
            "operator_id": operator_id,
            "golden_run": item["run_dir"],
            "capture_state_sha256": sha256(state_path),
            "sample_files_sha256": sha256(golden / "sample-files.json"),
        }
    )

assert len(process_ids) == 1
capture_process_id = next(iter(process_ids))
assert isinstance(capture_process_id, int) and capture_process_id > 0
formal = {
    "tool": "migration-agent",
    "action": "complete_formal_cuda_capture_session",
    "spec_binding": config["spec_binding"],
    "passed": True,
    "capture_status": "SEALED",
    "consumes_capture_session": True,
    "formal_session_consumed": True,
    "session_id": start["session_id"],
    "reservation_id": start["reservation_id"],
    "scan_result_sha256": sha256(scan_path),
    "capture_config_sha256": sha256(config_path),
    "operator_ids": expected_ids,
    "request_modes": [
        item["input_mode"] for item in config["requests"]
    ],
    "capture_process_id": capture_process_id,
    "goldens": goldens,
    "launch_evidence_dir": os.environ["LAUNCH_EVIDENCE_DIR"],
    "runtime_profile": runtime,
    "bundle_status": "NOT_BUILT",
}
(session / "formal-result.json").write_text(
    json.dumps(formal, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
print("formal CUDA Golden Runs=", len(goldens))
print("capture_process_id=", capture_process_id)
PY
```

## 8. 生成只供 Bundle 使用的 WAITING Spec

当前 `migration-spec.md` 已记录 reservation，revision 至少为 50。下面基于当前
revision 动态生成下一版候选 Spec；候选只有在 Bundle build 和 verify 都通过后才会
替换当前文件：

```bash
SESSION_RUN="$SESSION_RUN" ENV_RUN="$ENV_RUN" PLAN_RUN="$PLAN_RUN" \
BUNDLE_SPEC="$BUNDLE_SPEC" BUILD_RUN="$BUILD_RUN" \
VERIFY_RUN="$VERIFY_RUN" python - <<'PY'
import hashlib
import json
import os
import re
from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def field_value(text, name):
    matches = re.findall(
        rf"^- `{re.escape(name)}`: `([^`]*)`$",
        text,
        flags=re.MULTILINE,
    )
    assert len(matches) == 1, name
    return matches[0]


def set_field(text, name, value):
    pattern = rf"^- `{re.escape(name)}`: `[^`]*`$"
    replacement = f"- `{name}`: `{value}`"
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert count == 1, name
    return updated


repo = Path.cwd().resolve()
session = Path(os.environ["SESSION_RUN"]).resolve()
formal_path = session / "formal-result.json"
formal = json.loads(formal_path.read_text())
source = Path("migration-spec.md").read_text()
assert formal["passed"] is True
assert formal["capture_status"] == "SEALED"
current_revision = int(field_value(source, "state_revision"))
assert current_revision >= 50
assert field_value(source, "status") == "ACTIVE"
assert field_value(source, "phase") == "CUDA_CAPTURE"
assert field_value(source, "execution_site") == "CUDA"
assert field_value(source, "capture_session_id") == os.environ["SESSION_RUN"]
assert field_value(source, "session_status") == "ACTIVE"
target_revision = current_revision + 1

relative_goldens = {
    item["operator_id"]: str(
        Path(item["golden_run"]).resolve().relative_to(repo)
    )
    for item in formal["goldens"]
}
captured_counts = {}
for item in formal["goldens"]:
    state = json.loads(
        (Path(item["golden_run"]) / "capture-state.json").read_text()
    )
    captured_counts[item["operator_id"]] = state["saved_shape_count"]

updates = {
    "state_revision": str(target_revision),
    "status": "WAITING",
    "phase": "HANDOFF",
    "execution_site": "CUDA",
    "last_completed_action": "revision_6_cuda_capture_and_bundle_verified",
    "last_run": os.environ["VERIFY_RUN"],
    "next_action": (
        "人工复制 runs/handoff-build-r6-001/bundle 到 P800，先执行 "
        "manifest verify；通过后调用模型适配 Skill 进入 gap queue baseline"
    ),
    "session_status": "SEALED",
    "golden_runs": json.dumps(
        relative_goldens,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ),
    "captured_sample_counts": json.dumps(
        captured_counts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ),
    "bundle_status": "VALID",
    "bundle_path": f"{os.environ['BUILD_RUN']}/bundle",
    "manifest_path": f"{os.environ['BUILD_RUN']}/bundle/manifest.json",
}
for name, value in updates.items():
    source = set_field(source, name, value)

for operator_id, golden_run in relative_goldens.items():
    source = replace_once(
        source,
        f"| `{operator_id}` | `CAPTURE_REQUIRED` | `null` |",
        f"| `{operator_id}` | `CAPTURE_REQUIRED` | `{golden_run}` |",
    )
source = replace_once(
    source,
    "| revision 6 CUDA environment preflight passed without reading Scan "
    "or operators | `PENDING` | `null` |",
    f"| revision 6 CUDA environment preflight passed without reading Scan "
    f"or operators | `PASS` | `{os.environ['ENV_RUN']}/result.json` |",
)
source = replace_once(
    source,
    "| one revision 6 CUDA Session and at most three samples per operator "
    "| `PENDING` | `null` |",
    f"| one revision 6 CUDA Session and at most three samples per operator "
    f"| `PASS` | `{os.environ['SESSION_RUN']}/formal-result.json` |",
)
source = replace_once(
    source,
    "| every planned operator has a SEALED Golden Run | `PENDING` | "
    "`null` |",
    f"| every planned operator has a SEALED Golden Run | `PASS` | "
    f"`{os.environ['SESSION_RUN']}/formal-result.json` |",
)
decision = (
    f"| `{target_revision}` | revision 6 唯一正式 CUDA Session 完成"
    "文本与单图请求，"
    "全部 Scan 驱动 Golden 在共同 rank-0 采集进程中封存并 self-replay "
    "通过；同一 CUDA Agent 随后构建并验证动态 Handoff Bundle，进入人工"
    "复制 | "
    f"`{os.environ['SESSION_RUN']}/formal-result.json`、"
    f"`{os.environ['VERIFY_RUN']}/result.json` |"
)
marker = "\n<!-- AGENT-WRITABLE WORKING STATE: END -->"
assert marker in source
source = source.replace(marker, "\n" + decision + "\n" + marker, 1)

plan = Path(os.environ["PLAN_RUN"])
plan.mkdir(parents=True, exist_ok=False)
Path(os.environ["BUNDLE_SPEC"]).write_text(source)
result = {
    "tool": "migration-agent",
    "action": "prepare_revision_6_handoff_spec",
    "spec_binding": formal["spec_binding"],
    "passed": True,
    "consumes_capture_session": False,
    "formal_result_sha256": hashlib.sha256(
        formal_path.read_bytes()
    ).hexdigest(),
    "target_working_state": {
        "target_revision": target_revision,
        "status": "WAITING",
        "phase": "HANDOFF",
        "execution_site": "CUDA",
        "bundle_status": "VALID",
    },
}
(plan / "result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY
```

## 9. 动态构建并验证 Handoff Bundle

从映射生成重复参数；runbook 不保存算子常量：

```bash
golden_args=()
record_args=()
while IFS=$'\t' read -r ordinal operator_id golden_run record_run
do
  golden_args+=(--golden-run "$golden_run")
  record_args+=(--sample-record-result "$record_run/result.json")
done <"$SESSION_RUN/operator-map.tsv"

python model-adaptation/scripts/handoff_bundle.py \
  --mode build \
  --spec migration-spec.md \
  --run-dir "$BUILD_RUN" \
  --scan-result "$SCAN_RESULT" \
  "${golden_args[@]}" \
  --bundle-spec "$BUNDLE_SPEC" \
  "${record_args[@]}"

python model-adaptation/scripts/handoff_bundle.py \
  --mode verify \
  --spec migration-spec.md \
  --run-dir "$VERIFY_RUN" \
  --bundle-dir "$BUILD_RUN/bundle"

unset golden_args record_args
unset ordinal operator_id golden_run record_run
```

显式检查 Manifest v2、算子顺序和候选 Spec：

```bash
SESSION_RUN="$SESSION_RUN" BUILD_RUN="$BUILD_RUN" \
VERIFY_RUN="$VERIFY_RUN" BUNDLE_SPEC="$BUNDLE_SPEC" python - <<'PY'
import json
import os
from pathlib import Path

session = Path(os.environ["SESSION_RUN"])
build = Path(os.environ["BUILD_RUN"])
verify = Path(os.environ["VERIFY_RUN"])
formal = json.loads((session / "formal-result.json").read_text())
build_result = json.loads((build / "result.json").read_text())
verify_result = json.loads((verify / "result.json").read_text())
manifest = json.loads((build / "bundle" / "manifest.json").read_text())

assert manifest["schema"] == "handoff-manifest/v2"
assert build_result["passed"] is True
assert build_result["operator_ids"] == formal["operator_ids"]
assert verify_result["passed"] is True
assert verify_result["manifest_verified"] is True
assert verify_result["operator_ids"] == formal["operator_ids"]
assert verify_result["manifest_sha256"] == build_result["manifest_sha256"]
assert (
    (build / "bundle" / "migration-spec.md").read_bytes()
    == Path(os.environ["BUNDLE_SPEC"]).read_bytes()
)
print("revision 6 CUDA Handoff Bundle: PASS")
PY
```

## 10. 校验通过后才更新 Spec，并完成最终回传

先确认 GitHub 可接收所有待回传文件；有输出就停止，不压缩、不删 Tensor，也不重采：

```bash
find \
  "$SESSION_RUN" \
  "$RECORD_ROOT" \
  "$PLAN_RUN" \
  "$BUILD_RUN" \
  "$VERIFY_RUN" \
  -type f -size +95M -print
```

该命令必须没有输出。然后原子替换当前 Spec：

```bash
cp "$BUNDLE_SPEC" migration-spec.md.next
cmp -s migration-spec.md.next "$BUILD_RUN/bundle/migration-spec.md"
mv migration-spec.md.next migration-spec.md
grep -F -- '- `status`: `WAITING`' migration-spec.md
grep -F -- '- `phase`: `HANDOFF`' migration-spec.md

PLAN_RUN="$PLAN_RUN" python - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(
    (Path(os.environ["PLAN_RUN"]) / "result.json").read_text()
)
target = result["target_working_state"]["target_revision"]
assert f"- `state_revision`: `{target}`" in Path(
    "migration-spec.md"
).read_text()
print("installed Working State revision=", target)
PY
```

将 reservation 之后的全部证据一次性追加到原分支：

```bash
git add migration-spec.md \
  "$SESSION_RUN" \
  "$RECORD_ROOT" \
  "$PLAN_RUN" \
  "$BUILD_RUN" \
  "$VERIFY_RUN"
git diff --cached --check
git commit -m "evidence: add revision 6 CUDA capture and bundle"
git push origin "$EVIDENCE_BRANCH"
```

回传下面五项后停止，不在 CUDA 机器开始 P800 replay：

```bash
printf 'evidence branch: %s\n' "$EVIDENCE_BRANCH"
printf 'environment run: %s\n' "$ENV_RUN"
printf 'session run: %s\n' "$SESSION_RUN"
printf 'bundle: %s/bundle\n' "$BUILD_RUN"
printf 'result: PASS\n'
```

## 失败边界

### 环境失败

环境 preflight 失败时不创建正式 Session、不生成 Session 配置，也不占位正式
Session evidence 分支。保留不可变 `$ENV_RUN`，把失败动作和下一个新 ENV Run
写入 Working State，再只为环境证据创建或续用
`evidence/cuda-environment-preflight-r6-001`。环境失败不能记作 Operator Gap。

```bash
ENV_RUN="$ENV_RUN" python - <<'PY'
import json
import os
import re
from pathlib import Path


def field_value(text, name):
    matches = re.findall(
        rf"^- `{re.escape(name)}`: `([^`]*)`$",
        text,
        flags=re.MULTILINE,
    )
    assert len(matches) == 1, name
    return matches[0]


def set_field(text, name, value):
    pattern = rf"^- `{re.escape(name)}`: `[^`]*`$"
    replacement = f"- `{name}`: `{value}`"
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert count == 1, name
    return updated


environment_run = Path(os.environ["ENV_RUN"])
result_path = environment_run / "result.json"
result = json.loads(result_path.read_text())
assert result["action"] == "environment-preflight"
assert result["passed"] is False
assert result["preflight_status"] == "PREFLIGHT_FAILED"
assert result["consumes_capture_session"] is False
assert "operator_id" not in result

pattern = re.compile(r"cuda-environment-preflight-r6-([0-9]{3})")
ordinals = []
for path in Path("runs").glob("cuda-environment-preflight-r6-[0-9][0-9][0-9]"):
    match = pattern.fullmatch(path.name)
    if match:
        ordinals.append(int(match.group(1)))
current_match = pattern.fullmatch(environment_run.name)
assert current_match is not None
assert int(current_match.group(1)) in ordinals
next_environment_run = (
    f"runs/cuda-environment-preflight-r6-{max(ordinals) + 1:03d}"
)
assert not Path(next_environment_run).exists()

spec_path = Path("migration-spec.md")
source = spec_path.read_text()
current_revision = int(field_value(source, "state_revision"))
assert current_revision >= 49
assert field_value(source, "status") == "ACTIVE"
assert field_value(source, "phase") == "CUDA_CAPTURE"
assert field_value(source, "execution_site") in {"SOURCE", "CUDA"}
assert field_value(source, "capture_session_id") == "null"
assert field_value(source, "session_status") == "NOT_STARTED"
target_revision = current_revision + 1
updates = {
    "state_revision": str(target_revision),
    "execution_site": "CUDA",
    "last_completed_action": (
        "revision_6_cuda_environment_preflight_failed"
    ),
    "last_run": os.environ["ENV_RUN"],
    "next_action": (
        "在当前 environment evidence 分支修正 CUDA 环境，设置 "
        f"ENV_RUN={next_environment_run}，从正式 runbook 第 2 节重跑；"
        "通过后继续第 3 节"
    ),
}
for name, value in updates.items():
    source = set_field(source, name, value)

decision = (
    f"| `{target_revision}` | revision 6 CUDA 环境 preflight 失败；"
    "未读取算子、未生成 Session、未保存 Tensor。保留本 Run，并只允许"
    f"使用新 ENV Run `{next_environment_run}` 重试 | "
    f"`{result_path}` |"
)
marker = "\n<!-- AGENT-WRITABLE WORKING STATE: END -->"
assert marker in source
source = source.replace(marker, "\n" + decision + "\n" + marker, 1)
temporary = spec_path.with_suffix(".md.next")
temporary.write_text(source)
os.replace(temporary, spec_path)
print("record_revision_6_environment_failure=", target_revision)
print("next_environment_run=", next_environment_run)
PY

export ENV_EVIDENCE_BRANCH=evidence/cuda-environment-preflight-r6-001
CURRENT_BRANCH="$(git branch --show-current)"
if [ "$CURRENT_BRANCH" != "$ENV_EVIDENCE_BRANCH" ]; then
  test -z "$(git branch --list "$ENV_EVIDENCE_BRANCH")"
  git switch -c "$ENV_EVIDENCE_BRANCH"
fi
unset CURRENT_BRANCH

git add migration-spec.md "$ENV_RUN"
git diff --cached --check
git commit -m "evidence: record revision 6 CUDA environment failure"
git push -u origin "$ENV_EVIDENCE_BRANCH"
```

提交后停止。修正环境时保持在这个分支，按 Working State 指定的新 `ENV_RUN` 只重复
第 2 节；通过后继续第 3 节。不要回到第 1 节把已存在的环境 Run 当作新 Run，也
不要创建正式 Session evidence 分支，直到第 4 节 reservation。

### reservation 成功后的失败

先停止仍存活的服务进程。记录实际失败命令、退出码和一行错误摘要，然后同时检查
`capture-state.json` 与已经落盘的 `.pt`；任何一处证明样本存在，都视为正式
Session 已消耗：

```bash
export FAILED_COMMAND='替换为实际失败命令'
export FAILED_EXIT_CODE='替换为实际退出码'
export FAILURE_SUMMARY='替换为单行错误摘要'

SESSION_RUN="$SESSION_RUN" \
LAUNCH_EVIDENCE_DIR="$LAUNCH_EVIDENCE_DIR" \
FAILED_COMMAND="$FAILED_COMMAND" \
FAILED_EXIT_CODE="$FAILED_EXIT_CODE" \
FAILURE_SUMMARY="$FAILURE_SUMMARY" python - <<'PY'
import json
import os
import re
from pathlib import Path


def field_value(text, name):
    matches = re.findall(
        rf"^- `{re.escape(name)}`: `([^`]*)`$",
        text,
        flags=re.MULTILINE,
    )
    assert len(matches) == 1, name
    return matches[0]


def set_field(text, name, value):
    pattern = rf"^- `{re.escape(name)}`: `[^`]*`$"
    replacement = f"- `{name}`: `{value}`"
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert count == 1, name
    return updated


session = Path(os.environ["SESSION_RUN"])
launch = Path(os.environ["LAUNCH_EVIDENCE_DIR"])
assert launch.parent == session
assert launch.is_dir()
states = sorted(session.glob("operators/*/capture-state.json"))
sample_files = sorted(session.glob("operators/*/samples/*.pt"))
sample_artifacts = sorted(
    path
    for samples_dir in session.glob("operators/*/samples")
    for path in samples_dir.rglob("*")
    if path.is_file()
)
saved_shape_count = 0
state_errors = []
state_records = []
state_fields = {
    "schema",
    "spec_binding",
    "operator_id",
    "tp_rank",
    "tensor_parallel_size",
    "checkpoint",
    "loaded_checkpoint",
    "capture_session_config",
    "capture_process_id",
    "status",
    "capture_closed",
    "saved_shape_count",
    "repeated_call_count",
    "skipped_call_count",
    "samples",
    "skipped_signatures",
}
for path in states:
    try:
        value = json.loads(path.read_text())
        config = json.loads(
            (path.parent / "capture-config.json").read_text()
        )
        assert isinstance(value, dict)
        assert isinstance(config, dict)
        assert set(value) == state_fields
        assert config.get("schema") == "golden-capture-config/v1"
        assert Path(config["run_dir"]).resolve() == path.parent.resolve()
        checkpoint = config["checkpoint"]
        assert isinstance(checkpoint, dict)
        assert set(checkpoint) == {
            "id",
            "model_path",
            "revision",
            "config_digest",
        }
        expected_state = {
            "schema": "kernel-call-capture-state/v1",
            "spec_binding": config["spec_binding"],
            "operator_id": config["operator_id"],
            "tp_rank": config["tp_rank"],
            "tensor_parallel_size": config["tensor_parallel_size"],
            "checkpoint": checkpoint,
            "loaded_checkpoint": {
                "model_path": checkpoint["model_path"],
                "revision": checkpoint["revision"],
            },
            "capture_session_config": config["session_config"],
        }
        for field, expected in expected_state.items():
            assert value[field] == expected
        process_id = value["capture_process_id"]
        assert isinstance(process_id, int) and not isinstance(
            process_id, bool
        )
        assert process_id > 0
        count = value["saved_shape_count"]
        assert isinstance(count, int) and not isinstance(count, bool)
        assert count >= 0
        if count == 0:
            assert value["status"] == "ACTIVE"
            assert value["capture_closed"] is False
            assert value["repeated_call_count"] == 0
            assert value["skipped_call_count"] == 0
            assert value["samples"] == []
            assert value["skipped_signatures"] == []
        saved_shape_count += count
        state_records.append((path, value))
    except (
        AssertionError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        state_errors.append(
            {"path": str(path), "error": f"{type(error).__name__}: {error}"}
        )

formal_session_consumed = bool(
    saved_shape_count or sample_artifacts or state_errors
)
archived_zero_sample_states = []
if not formal_session_consumed and state_records:
    archive_root = launch / "zero-sample-capture-states"
    archive_root.mkdir(parents=True, exist_ok=False)
    for path, _ in state_records:
        target = archive_root / path.parent.name / path.name
        target.parent.mkdir()
        assert not target.exists()
        os.replace(path, target)
        archived_zero_sample_states.append(str(target))
spec_path = Path("migration-spec.md")
source = spec_path.read_text()
current_revision = int(field_value(source, "state_revision"))
assert current_revision >= 50
assert field_value(source, "status") == "ACTIVE"
assert field_value(source, "phase") == "CUDA_CAPTURE"
assert field_value(source, "execution_site") == "CUDA"
assert field_value(source, "capture_session_id") == os.environ["SESSION_RUN"]
assert field_value(source, "session_status") == "ACTIVE"
target_revision = current_revision + 1
failure_path = session / f"failure-{target_revision}.json"
assert not failure_path.exists()
command = os.environ["FAILED_COMMAND"].strip()
summary = os.environ["FAILURE_SUMMARY"].strip()
assert command and summary and "\n" not in command and "\n" not in summary
exit_code = int(os.environ["FAILED_EXIT_CODE"])
start = json.loads((session / "session-start.json").read_text())

failure = {
    "tool": "migration-agent",
    "action": "record_revision_6_cuda_failure",
    "spec_binding": start["spec_binding"],
    "session_id": start["session_id"],
    "reservation_id": start["reservation_id"],
    "failed_command": command,
    "exit_code": exit_code,
    "error_summary": summary,
    "launch_evidence_dir": os.environ["LAUNCH_EVIDENCE_DIR"],
    "saved_shape_count": saved_shape_count,
    "sample_files_on_disk": len(sample_files),
    "sample_paths": [str(path) for path in sample_files],
    "sample_artifacts_on_disk": len(sample_artifacts),
    "sample_artifact_paths": [str(path) for path in sample_artifacts],
    "capture_state_errors": state_errors,
    "archived_zero_sample_states": archived_zero_sample_states,
    "formal_session_consumed": formal_session_consumed,
    "state_revision_before": current_revision,
    "state_revision_after": target_revision,
}
failure_path.write_text(
    json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)

if formal_session_consumed:
    updates = {
        "state_revision": str(target_revision),
        "status": "BLOCKED",
        "phase": "CUDA_CAPTURE",
        "execution_site": "CUDA",
        "last_completed_action": (
            "revision_6_cuda_capture_failed_after_sample"
        ),
        "last_run": os.environ["SESSION_RUN"],
        "next_action": "none",
        "session_status": "FAILED",
        "stop_reason": (
            "revision 6 正式 Session 已有 Golden Sample 或无法可靠读取"
            "采集状态，禁止重启或创建第二个 Session"
        ),
        "resume_requires_contract_revision": "true",
    }
    decision_text = (
        "revision 6 正式 CUDA Session 失败且已有样本证据，进入 "
        "BLOCKED / CUDA_CAPTURE，禁止重开 Session"
    )
else:
    updates = {
        "state_revision": str(target_revision),
        "status": "ACTIVE",
        "phase": "CUDA_CAPTURE",
        "execution_site": "CUDA",
        "last_completed_action": (
            "revision_6_cuda_capture_failed_before_sample"
        ),
        "last_run": os.environ["SESSION_RUN"],
        "next_action": (
            "修正模型加载或服务环境后，只在同一个 reservation、"
            "SESSION_RUN 和 evidence 分支继续正式 CUDA Capture；下一次"
            "启动必须使用新 state revision 对应的 launch 证据目录"
        ),
        "session_status": "ACTIVE",
        "stop_reason": "null",
        "resume_requires_contract_revision": "false",
    }
    decision_text = (
        "revision 6 CUDA 启动失败但 state 与磁盘均无 Golden Sample；"
        "保留证据并只允许同一 reservation 内重试"
    )
for name, value in updates.items():
    source = set_field(source, name, value)

decision = (
    f"| `{target_revision}` | {decision_text} | "
    f"`{failure_path}` |"
)
marker = "\n<!-- AGENT-WRITABLE WORKING STATE: END -->"
assert marker in source
source = source.replace(marker, "\n" + decision + "\n" + marker, 1)
temporary = spec_path.with_suffix(".md.next")
temporary.write_text(source)
os.replace(temporary, spec_path)
print("saved_shape_count=", saved_shape_count)
print("sample_files_on_disk=", len(sample_files))
print("sample_artifacts_on_disk=", len(sample_artifacts))
print("formal_session_consumed=", formal_session_consumed)
print("failure_result=", failure_path)
PY

git add migration-spec.md "$SESSION_RUN"
git diff --cached --check
git commit -m "evidence: record revision 6 CUDA capture failure"
git push origin "$EVIDENCE_BRANCH"
```

- 尚未产生任何 Golden Sample：模型加载或服务环境失败不算新的 Capture Session。
  修正后只允许在同一个 reservation、`SESSION_RUN` 和 evidence 分支继续；不得新建 Run 或 evidence 分支。
- 已有 Golden Sample：唯一 Session 已消耗。保留全部状态和日志，不得重启模型、
  不得补 shape，也不得创建第二个 Session。

零样本路径把 Working State 保持为 `ACTIVE / CUDA_CAPTURE`；已有样本或采集状态
无法可靠读取时，Working State 必须写成 `- status: BLOCKED`、
`- phase: CUDA_CAPTURE` 并停止。每次失败使用新的
`$SESSION_RUN/failure-<state_revision>.json`，提交到当前 `$EVIDENCE_BRANCH`；
禁止覆盖或删除失败证据后重新占位。

# Ticket 16：在 P800 执行 Kernel Baseline

> 本页只执行 baseline，不修改 SGLang-Kunlun。它从已校验 bundle 读取三个
> Golden Sample，调用现有 `kunlun_ops.swiglu`，再用固定
> `torch.testing.assert_close` 比较 CUDA expected。baseline 不计入五轮修复。

## 1. 准备两个干净仓库

以下命令都在 P800 机器执行。先进入 `model-adaptation-agents` 仓库根目录，并把
现有 SGLang-Kunlun checkout 的绝对路径写入环境变量：

```bash
set -euo pipefail
cd /root/model-adaptation-agents
export SGLANG_KUNLUN_WORKTREE=/替换为实际的/SGLang-Kunlun/仓库根目录
test -n "${SGLANG_KUNLUN_WORKTREE:-}"
export SGLANG_PLATFORM=kunlun
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export PYTHONPATH="$SGLANG_KUNLUN_WORKTREE/python:$SGLANG_KUNLUN_WORKTREE/sglang-kunlun${PYTHONPATH:+:$PYTHONPATH}"

git fetch origin
export SOURCE_BRANCH=origin/chore/step3p7-p800-baseline
export EVIDENCE_BRANCH=evidence/p800-baseline-r5-001
export CHECK_RUN=runs/p800-baseline-check-r5-001
export REPLAY_RUN=runs/p800-baseline-replay-r5-001
export ASSESS_RUN=runs/p800-baseline-assessment-r5-001
export GOLDEN_RUN=runs/handoff-build-r5-001/bundle/runs/cuda-golden-r5-001
export OPERATOR_ID=sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul
export ALLOWED_PATH=sglang-kunlun/sglang_kunlun/hooks/layers/quantization/unquant.py
export KUNLUN_REVISION=546ad8c682392922792bbbfe53a8bf575545f118
export EXPECTED_MANIFEST_SHA256=c7886622083e8516e335df6946321858ef58dfdec8f33d268aed7140eb13a83a

test -z "$(git status --short)"
git merge-base --is-ancestor \
  e7f861af61242be1625277725fbe75f14c36a0fe \
  "$SOURCE_BRANCH"
REMOTE_EVIDENCE="$(git ls-remote --heads origin "refs/heads/$EVIDENCE_BRANCH")"
test -z "$REMOTE_EVIDENCE"
unset REMOTE_EVIDENCE
test ! -e "$CHECK_RUN"
test ! -e "$REPLAY_RUN"
test ! -e "$ASSESS_RUN"

git switch -c "$EVIDENCE_BRANCH" "$SOURCE_BRANCH"
test -z "$(git status --short)"
```

只替换 `SGLANG_KUNLUN_WORKTREE`。上述两个 Kunlun 变量和 `PYTHONPATH` 前缀不可
删除。其他可选 P800 环境变量由 Agent 按
`docs/p800-environment-and-repair.md` 选择，不能整表无条件导出；实际选择及原因
必须进入环境证据。对每个实际导出的可选变量，准备一个
`--p800-environment-reason '变量名=选择原因'` 参数；没有选择可选变量时不传。
不能换 Run 名重试。若 evidence 分支或任一 Run 已存在，停止并告诉 Agent。

## 2. 在读取 Tensor 前复核状态

这一步只读取 Markdown 和 JSON：

```bash
EXPECTED_MANIFEST_SHA256="$EXPECTED_MANIFEST_SHA256" \
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

spec = Path("migration-spec.md").read_text(encoding="utf-8")
verification = json.loads(
    Path("runs/handoff-verify-p800-r5-001/result.json").read_text(
        encoding="utf-8"
    )
)
review = json.loads(
    Path("runs/p800-handoff-review-r5-001/result.json").read_text(
        encoding="utf-8"
    )
)
adapter = json.loads(
    Path("runs/adapter-003/result.json").read_text(encoding="utf-8")
)
manifest_path = Path("runs/handoff-build-r5-001/bundle/manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

assert "- `state_revision`: `28`" in spec
assert "- `status`: `ACTIVE`" in spec
assert "- `phase`: `P800_REPAIR`" in spec
assert "- `execution_site`: `SOURCE`" in spec
assert "- `last_run`: `runs/adapter-003`" in spec
assert "- `bundle_status`: `VERIFIED_ON_P800`" in spec
assert "- `baseline_run`: `null`" in spec
assert "- `attempts_used`: `0`" in spec
assert verification["passed"] is True
assert verification["manifest_verified"] is True
assert verification["file_count"] == 13
assert (
    verification["manifest_sha256"]
    == os.environ["EXPECTED_MANIFEST_SHA256"]
)
assert review["passed"] is True
assert review["tensor_payloads_deserialized"] is False
assert review["spec_binding"] == verification["spec_binding"]
assert adapter["passed"] is True
assert adapter["active_operator"].endswith("._swiglu_silu_clamp_mul")
assert adapter["spec_binding"] == verification["spec_binding"]
for source in adapter["source_files"]:
    assert (
        hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
        == source["sha256"]
    ), source["path"]
assert len(manifest["files"]) == 13
assert (
    hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    == os.environ["EXPECTED_MANIFEST_SHA256"]
)
print("P800 baseline metadata precheck: PASS")
PY
```

检查失败时不要读取 Golden Tensor。

## 3. 检查 P800 环境和固定源码基线

先确认当前 Python 能直接调用 P800 Torch 与现有 `kunlun_ops.swiglu`：

```bash
python - <<'PY'
import torch
import kunlun_ops
import sglang_kunlun
import os
from pathlib import Path

assert torch.cuda.is_available(), "当前 Python 看不到 P800 设备"
assert torch.cuda.device_count() >= 1, torch.cuda.device_count()
assert callable(torch.testing.assert_close)
assert callable(kunlun_ops.swiglu)
assert os.environ["SGLANG_PLATFORM"] == "kunlun"
assert os.environ["SGLANG_IS_FLASHINFER_AVAILABLE"] == "False"
worktree = Path(os.environ["SGLANG_KUNLUN_WORKTREE"]).resolve()
sglang = __import__("sglang")
assert Path(sglang.__file__).resolve().is_relative_to(worktree / "python")
assert Path(sglang_kunlun.__file__).resolve().is_relative_to(
    worktree / "sglang-kunlun"
)
print("torch_version=", torch.__version__)
print("device_count=", torch.cuda.device_count())
print("sglang=", sglang.__file__)
print("sglang_kunlun=", sglang_kunlun.__file__)
print("kunlun_ops=", kunlun_ops.__file__)
PY
```

再确认 SGLang-Kunlun checkout 正好位于 Contract 固定提交，并封存 baseline 前的
干净工作区检查：

```bash
test "$(git -C "$SGLANG_KUNLUN_WORKTREE" rev-parse --show-toplevel)" = \
  "$(cd "$SGLANG_KUNLUN_WORKTREE" && pwd -P)"
test "$(git -C "$SGLANG_KUNLUN_WORKTREE" rev-parse HEAD)" = \
  "$KUNLUN_REVISION"
test -f "$SGLANG_KUNLUN_WORKTREE/$ALLOWED_PATH"

python model-adaptation/scripts/workspace_guard.py \
  --mode check-baseline \
  --spec migration-spec.md \
  --run-dir "$CHECK_RUN" \
  --worktree "$SGLANG_KUNLUN_WORKTREE" \
  --allowed-path "$ALLOWED_PATH"
```

`workspace_guard.py` 会拒绝 staged、tracked 或 untracked 修改；它不会替你清理。

## 4. 重放 bundle 中的三个 Golden Sample

这里开始首次读取 `.pt`。调用边界是现有 `kunlun_ops.swiglu`，不会启动完整模型，
不会保存 P800 actual Tensor：

```bash
python model-adaptation/scripts/replay_compare.py \
  --mode kernel-replay \
  --spec migration-spec.md \
  --run-dir "$REPLAY_RUN" \
  --scan-result runs/scan-006/result.json \
  --operator-id "$OPERATOR_ID" \
  --golden-run "$GOLDEN_RUN" \
  --execution-site p800 \
  --sglang-worktree "$SGLANG_KUNLUN_WORKTREE"
```

如果 Agent 例如设置了 `MODEL_PATH`，在末尾追加：

```bash
  --p800-environment-reason \
  'MODEL_PATH=加载 Contract 固定的 Step-3.7-Flash checkpoint'
```

每个已设置的候选变量都追加一次；工具会核对变量确实存在，并把实际值和原因一起
写入 `result.json`。不要给未设置的变量编造原因。

工具把数值不一致记录为有效的 `passed: false` 结果，而不是命令错误。退出码 `2`、
缺少 `result.json` 或 Run 不完整属于工具/环境失败，此时直接进入文末失败处理。

## 5. 判定 baseline，但不消耗修复轮数

```bash
python model-adaptation/scripts/workspace_guard.py \
  --mode assess-baseline \
  --spec migration-spec.md \
  --run-dir "$ASSESS_RUN" \
  --worktree "$SGLANG_KUNLUN_WORKTREE" \
  --allowed-path "$ALLOWED_PATH" \
  --replay-result "$REPLAY_RUN/result.json"
```

显式核对结果：

```bash
REPLAY_RUN="$REPLAY_RUN" ASSESS_RUN="$ASSESS_RUN" python - <<'PY'
import json
import os
from pathlib import Path

replay = json.loads(
    (Path(os.environ["REPLAY_RUN"]) / "result.json").read_text(
        encoding="utf-8"
    )
)
worker = json.loads(
    (Path(os.environ["REPLAY_RUN"]) / "worker-result.json").read_text(
        encoding="utf-8"
    )
)
assessment = json.loads(
    (Path(os.environ["ASSESS_RUN"]) / "result.json").read_text(
        encoding="utf-8"
    )
)

assert replay["action"] == "kernel-replay"
assert replay["execution_site"] == "p800"
assert replay["invocation_target"] == "kunlun_ops.swiglu"
assert replay["launch_environment"]["SGLANG_PLATFORM"] == "kunlun"
assert (
    replay["launch_environment"]["SGLANG_IS_FLASHINFER_AVAILABLE"]
    == "False"
)
assert replay["launch_environment"]["PYTHONPATH_PREFIX"] == [
    str(Path(os.environ["SGLANG_KUNLUN_WORKTREE"]) / "python"),
    str(Path(os.environ["SGLANG_KUNLUN_WORKTREE"]) / "sglang-kunlun"),
]
assert all(
    set(record) == {"value", "reason"} and record["reason"]
    for record in replay["launch_environment"]["selected_optional"].values()
)
assert replay["checked_shape_count"] == 3
assert replay["failed_shape_count"] in {0, 1, 2, 3}
assert replay["actual_tensors_saved"] is False
assert worker["passed"] is replay["passed"]
assert worker["failed_shape_count"] == replay["failed_shape_count"]
assert len(worker["errors"]) == replay["failed_shape_count"]
allowed_failure_types = {
    "AssertionError",
    "KernelReplayError",
    "RuntimeError",
    "TypeError",
}
assert all(
    error["type"] in allowed_failure_types
    for error in worker["errors"]
), worker["errors"]
assert assessment["passed"] is replay["passed"]
assert assessment["gap_observed"] is (not replay["passed"])
assert assessment["attempts_used"] == 0
assert assessment["workspace_clean"] is True

print(
    "P800 baseline gap_observed:",
    str(assessment["gap_observed"]).lower(),
)
print("failed_shape_count=", replay["failed_shape_count"])
for error in worker["errors"]:
    print("baseline_error=", json.dumps(error, ensure_ascii=False))
PY
```

两个结果都可能成立：

- `gap_observed=true`：至少一个样本执行或精度失败，Agent 才能提出 attempt 1
  假设。`RuntimeError/TypeError` 只有在现有 `kunlun_ops.swiglu` 调用本身失败时
  才会被 worker 记录；依赖导入、设备搬运或比较器异常会使工具退出码为 `2`。
- `gap_observed=false`：三个样本全部通过，首选静态候选不是真实 correctness gap；
  Agent 会按 Contract 进入 `BLOCKED`，不能伪造失败或重访 CUDA。

`gap_observed=true` 仍不会自动开始修复。Agent 会核对上传的
`worker-result.json.errors`，确认它是算子调用或精度失败，而不是环境问题。

## 6. 只上传三个 baseline Run

```bash
test -z "$(git -C "$SGLANG_KUNLUN_WORKTREE" status --short)"
test -z "$(git status --short -- migration-spec.md)"

git add "$CHECK_RUN" "$REPLAY_RUN" "$ASSESS_RUN"
git diff --cached --check
STAGED_OUTSIDE_BASELINE="$(
  git diff --cached --name-only |
    grep -Ev "^(${CHECK_RUN}|${REPLAY_RUN}|${ASSESS_RUN})/" || true
)"
test -z "$STAGED_OUTSIDE_BASELINE"
unset STAGED_OUTSIDE_BASELINE

git commit -m "evidence: add P800 baseline replay"
git push -u origin "$EVIDENCE_BRANCH"
```

上传后停止，不修改 SGLang-Kunlun，不开始 attempt 1。告诉 Agent：

```text
evidence branch: evidence/p800-baseline-r5-001
P800 baseline gap_observed: true 或 false
```

Agent 会先核验三个 Run。只有 `gap_observed=true` 才会写入第一个
`active_hypothesis` 并给出 attempt 1 命令。

## 失败时怎么做

如果环境检查、`workspace_guard.py` 或 `replay_compare.py` 发生工具错误：

1. 不删除或复用已经创建的 Run；
2. 不修改、清理或切换 SGLang-Kunlun worktree；
3. 不创建另一个 evidence 分支或换 Run 名重试；
4. 回传失败命令、完整 stderr，以及两个仓库的 `git status --short`。

数值比较得到 `passed: false` 不是这里所说的工具错误，它是正常的
`gap_observed=true` baseline 结果，必须按第 6 节上传。

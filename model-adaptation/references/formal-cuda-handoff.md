# Ticket 15：用已接受的 Golden 构建 CUDA Handoff Bundle

> 本页是正式采集后的第二步，只构建和校验交接包。不要启动 SGLang、不要重新采集、
> 不要执行 CUDA 或 P800 kernel replay。PID `98141` 的模型路径失败不计入
> Capture Session；PID `114828` 产生的 `runs/cuda-golden-r5-001` 是已接受的
> 唯一正式 Golden。

## 1. 拉取已接受 Golden 的执行分支

以下命令都在 CUDA 机器的 `model-adaptation-agents` 仓库根目录执行：

```bash
set -euo pipefail
git fetch origin
git switch chore/step3p7-p800-baseline
git pull --ff-only origin chore/step3p7-p800-baseline
WORKTREE_STATUS="$(git status --short)"
test -z "$WORKTREE_STATUS"
unset WORKTREE_STATUS

export GOLDEN_RUN=runs/cuda-golden-r5-001
export SAMPLE_RECORD_RESULT=runs/cuda-golden-r5-001-sample-record/result.json
export BUNDLE_SPEC=runs/handoff-plan-r5-001/migration-spec.md
export BUILD_RUN=runs/handoff-build-r5-001
export VERIFY_RUN=runs/handoff-verify-cuda-r5-001
export EVIDENCE_BRANCH=evidence/cuda-handoff-r5-001
```

固定路径都必须存在，两个输出 Run 和远端 evidence 分支都必须尚未存在：

```bash
test -f migration-spec.md
test -f "$GOLDEN_RUN/capture-state.json"
test -f "$GOLDEN_RUN/sample-files.json"
test -f "$GOLDEN_RUN/self-replay/result.json"
test -f "$SAMPLE_RECORD_RESULT"
test -f "$BUNDLE_SPEC"
test ! -e "$BUILD_RUN"
test ! -e "$VERIFY_RUN"
REMOTE_EVIDENCE="$(git ls-remote --heads origin "refs/heads/$EVIDENCE_BRANCH")"
test -z "$REMOTE_EVIDENCE"
unset REMOTE_EVIDENCE
```

任一检查失败都停止，不要换 Run 名重做。

## 2. 只读核对接受决定、Golden 和候选 Spec

这一步只读取 JSON/Markdown，不加载 `.pt` Tensor：

```bash
python - <<'PY'
import json
from pathlib import Path

current = Path("migration-spec.md").read_text(encoding="utf-8")
candidate = Path("runs/handoff-plan-r5-001/migration-spec.md").read_text(
    encoding="utf-8"
)
review = json.loads(
    Path("runs/cuda-formal-review-r5-002/result.json").read_text()
)
capture = json.loads(
    Path("runs/cuda-golden-r5-001/capture-state.json").read_text()
)
record = json.loads(
    Path(
        "runs/cuda-golden-r5-001-sample-record/result.json"
    ).read_text()
)

begin = "<!-- CONTRACT-DATA: BEGIN -->"
end = "<!-- CONTRACT-DATA: END -->"
current_contract = current.split(begin, 1)[1].split(end, 1)[0]
candidate_contract = candidate.split(begin, 1)[1].split(end, 1)[0]

assert current_contract == candidate_contract
assert "- `state_revision`: `25`" in current
assert "- `status`: `ACTIVE`" in current
assert "- `phase`: `CUDA_CAPTURE`" in current
assert review["passed"] is True
assert review["accepted_golden_run"] == "runs/cuda-golden-r5-001"
assert (
    review["counting_rule"]["model_load_failure_counts_as_session"]
    is False
)
assert capture["status"] == "SEALED"
assert capture["capture_closed"] is True
assert capture["saved_shape_count"] == 3
assert capture["self_replay"]["passed"] is True
assert record["passed"] is True
assert record["sample_file_count"] == 3
assert Path(record["golden_run"]) == Path(
    "runs/cuda-golden-r5-001"
).resolve()
assert Path(record["sample_files_path"]) == Path(
    "runs/cuda-golden-r5-001/sample-files.json"
).resolve()
assert (
    record["sample_files_sha256"]
    == capture["self_replay"]["sample_files_sha256"]
)
assert "- `state_revision`: `26`" in candidate
assert "- `status`: `WAITING`" in candidate
assert "- `phase`: `HANDOFF`" in candidate

print("accepted Golden and candidate handoff Spec: PASS")
PY
```

检查失败时停止；不要修改当前 Spec。

## 3. 在独立 evidence 分支构建交接包

```bash
git switch -c "$EVIDENCE_BRANCH"

python model-adaptation/scripts/handoff_bundle.py \
  --mode build \
  --spec migration-spec.md \
  --run-dir "$BUILD_RUN" \
  --golden-run "$GOLDEN_RUN" \
  --bundle-spec "$BUNDLE_SPEC" \
  --sample-record-result "$SAMPLE_RECORD_RESULT"
```

构建会逐文件核对 Golden、样本摘要、CUDA self-replay 和 Contract 绑定，然后把候选
Spec、Golden 与完整 manifest 写入 `$BUILD_RUN/bundle`。它不会读取 Working State，
也不会加载模型。

## 4. 在 CUDA 端校验刚生成的 bundle

```bash
python model-adaptation/scripts/handoff_bundle.py \
  --mode verify \
  --spec migration-spec.md \
  --run-dir "$VERIFY_RUN" \
  --bundle-dir "$BUILD_RUN/bundle"
```

`verify` 会封存失败结果但仍正常退出，因此必须显式检查结果；只有下面全部通过才可
替换当前 Spec：

```bash
BUILD_RUN="$BUILD_RUN" VERIFY_RUN="$VERIFY_RUN" \
BUNDLE_SPEC="$BUNDLE_SPEC" python - <<'PY'
import json
import os
from pathlib import Path

build_run = Path(os.environ["BUILD_RUN"])
verify_run = Path(os.environ["VERIFY_RUN"])
bundle_spec = Path(os.environ["BUNDLE_SPEC"])
build = json.loads((build_run / "result.json").read_text())
verify = json.loads((verify_run / "result.json").read_text())

assert build["passed"] is True
assert verify["passed"] is True
assert verify["manifest_verified"] is True
assert verify["manifest_sha256"] == build["manifest_sha256"]
assert (
    (build_run / "bundle" / "migration-spec.md").read_bytes()
    == bundle_spec.read_bytes()
)
print("CUDA handoff bundle verify: PASS")
PY
```

## 5. 校验通过后再推进 Spec 并上传

先复制到临时文件，逐字节比较后再原子替换：

```bash
cp "$BUILD_RUN/bundle/migration-spec.md" migration-spec.md.next
cmp -s migration-spec.md.next "$BUNDLE_SPEC"
mv migration-spec.md.next migration-spec.md

grep -F -- '- `state_revision`: `26`' migration-spec.md
grep -F -- '- `status`: `WAITING`' migration-spec.md
grep -F -- '- `phase`: `HANDOFF`' migration-spec.md
```

然后只提交本次交接证据和状态推进：

```bash
git add migration-spec.md "$BUILD_RUN" "$VERIFY_RUN"
git diff --cached --check
git commit -m "evidence: add verified CUDA handoff bundle"
git push -u origin "$EVIDENCE_BRANCH"
```

上传后停止并告诉 Agent：

```text
evidence branch: evidence/cuda-handoff-r5-001
CUDA handoff bundle build/verify: PASS
```

此时不要在 CUDA 机器继续操作，也不要自行开始 P800 baseline。Agent 会核对上传内容，
再给出 P800 端“先复制、先校验 manifest、后读取 Tensor”的命令。

## 失败时怎么做

- 构建或校验失败：不要替换 `migration-spec.md`，不要重启模型，不要重新采集，也不要
  换新的 Run 名绕过失败。
- `git push` 失败：保留当前 evidence 分支和提交，只修复网络或权限后重试同一条
  push；不要重新 build/verify。
- 失败信息连同 `git status --short`、`$BUILD_RUN/result.json` 或
  `$VERIFY_RUN/result.json`（若已生成）回传给 Agent，由 Agent 决定下一动作。

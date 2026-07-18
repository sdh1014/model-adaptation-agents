# Ticket 15：在 P800 校验 Handoff Bundle

> 本页只把已由 CUDA 校验过的 bundle 复制到 P800，并在 P800 重新校验完整
> manifest。它不会调用 Tensor 反序列化接口，不会重放算子，不会修改
> `migration-spec.md`，也不会开始 baseline。

## 1. 从最新基线创建唯一 P800 evidence 分支

以下命令都在 P800 机器的 `model-adaptation-agents` 仓库根目录执行：

```bash
set -euo pipefail
git fetch origin

export SOURCE_BRANCH=origin/chore/step3p7-p800-baseline
export EVIDENCE_BRANCH=evidence/p800-handoff-verify-r5-001
export BUNDLE_DIR=runs/handoff-build-r5-001/bundle
export VERIFY_RUN=runs/handoff-verify-p800-r5-001
export EXPECTED_MANIFEST_SHA256="c7886622083e8516e335df6946321858ef58dfdec8f33d268aed7140eb13a83a"

WORKTREE_STATUS="$(git status --short)"
test -z "$WORKTREE_STATUS"
unset WORKTREE_STATUS

git merge-base --is-ancestor \
  a7042d2ef406726e6cf3a754ef8d9409b434987e \
  "$SOURCE_BRANCH"
REMOTE_EVIDENCE="$(git ls-remote --heads origin "refs/heads/$EVIDENCE_BRANCH")"
test -z "$REMOTE_EVIDENCE"
unset REMOTE_EVIDENCE
test ! -e "$VERIFY_RUN"

git switch -c "$EVIDENCE_BRANCH" "$SOURCE_BRANCH"
```

任一检查失败都停止，不要删除已有 Run，也不要换名字重新执行。

## 2. 读取状态和 manifest 元数据

这一步只读取 Markdown/JSON，不读取 `.pt` 内容：

```bash
BUNDLE_DIR="$BUNDLE_DIR" \
EXPECTED_MANIFEST_SHA256="$EXPECTED_MANIFEST_SHA256" \
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

spec = Path("migration-spec.md").read_text(encoding="utf-8")
bundle = Path(os.environ["BUNDLE_DIR"])
manifest_path = bundle / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

assert "- `state_revision`: `26`" in spec
assert "- `status`: `WAITING`" in spec
assert "- `phase`: `HANDOFF`" in spec
assert "- `bundle_status`: `VALID`" in spec
assert "- `p800_verification_run`: `null`" in spec
assert manifest["schema"] == "handoff-manifest/v1"
assert manifest["operator_id"].endswith("._swiglu_silu_clamp_mul")
assert manifest["golden_run"] == "runs/cuda-golden-r5-001"
assert len(manifest["files"]) == 13
assert (
    hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    == os.environ["EXPECTED_MANIFEST_SHA256"]
)
assert (
    Path("migration-spec.md").read_bytes()
    == (bundle / "migration-spec.md").read_bytes()
)

print("P800 handoff metadata precheck: PASS")
PY
```

失败时不要运行下一步。

## 3. 在 P800 校验完整 bundle

`verify` 会读取每个文件的原始字节来计算大小和 SHA-256，但不会反序列化 Tensor：

```bash
python model-adaptation/scripts/handoff_bundle.py \
  --mode verify \
  --spec migration-spec.md \
  --run-dir "$VERIFY_RUN" \
  --bundle-dir "$BUNDLE_DIR"
```

工具即使发现 manifest 错误也会把失败写进 Run 后正常退出，因此必须显式检查结果：

```bash
VERIFY_RUN="$VERIFY_RUN" \
EXPECTED_MANIFEST_SHA256="$EXPECTED_MANIFEST_SHA256" \
python - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(
    (Path(os.environ["VERIFY_RUN"]) / "result.json").read_text(
        encoding="utf-8"
    )
)
assert result["passed"] is True, result
assert result["manifest_verified"] is True, result
assert result["manifest_sha256"] == os.environ["EXPECTED_MANIFEST_SHA256"]
assert result["file_count"] == 13
assert result["spec_binding"] == {
    "contract_data_sha256": (
        "f683ea62427279da3c899a3ef0afcfa1d16298b7424154664c7e3fac9b13d938"
    ),
    "contract_revision": 5,
    "spec_id": "step3p7-flash-p800-demo",
}
print("P800 handoff manifest verify: PASS")
PY
```

只有看到两条 `PASS` 才继续。

## 4. 只上传 P800 verification Run

不要修改或提交 `migration-spec.md`。本次提交只包含新生成的 verification Run：

```bash
git add "$VERIFY_RUN"
git diff --cached --check
STAGED_OUTSIDE_VERIFY="$(
  git diff --cached --name-only | grep -v "^${VERIFY_RUN}/" || true
)"
test -z "$STAGED_OUTSIDE_VERIFY"
unset STAGED_OUTSIDE_VERIFY
git commit -m "evidence: verify handoff bundle on P800"
git push -u origin "$EVIDENCE_BRANCH"
```

上传后停止并告诉 Agent：

```text
evidence branch: evidence/p800-handoff-verify-r5-001
P800 handoff manifest verify: PASS
```

Agent 核验后才会把 Working State 从 `WAITING / HANDOFF` 推进到
`ACTIVE / P800_REPAIR`，并给出 baseline replay 命令。当前不要加载 Golden
Tensor，不要执行 baseline，不要修改 SGLang-Kunlun。

## 失败时怎么做

如果第 3 步的显式检查失败：

1. 不读取 Golden Tensor，不执行 baseline，不修改 Spec。
2. 仍在同一个 `$EVIDENCE_BRANCH` 中只提交 `$VERIFY_RUN`，提交消息使用
   `evidence: record failed P800 handoff verification`。
3. 推送该分支并回传 `result.json` 中的 `error`；不要删除失败 Run 或换名字重试。

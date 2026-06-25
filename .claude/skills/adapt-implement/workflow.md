# 适配实现流程

## 0. 总体流程

```text
解析目标
→ 校验分析与评估 revision
→ 选择一个工作项
→ 复核仓库与执行环境
→ 读取历史尝试
→ 按能力加载知识
→ 提出一个可证伪假设
→ 最小修改
→ 范围检查与局部验收
→ 成功 / 新假设 / blocked / decision_required
→ 更新报告并停止
```

任何时候都不得自动进入下一个工作项。

---

## 1. 解析参数

第一个参数：

```text
<model-id>/<target-id>
```

可选参数：

- `--item <WI-ID>`：指定工作项；
- `--max-attempts <N>`：本次最多验证的不同假设，默认 3，取值 1～5；
- `--decision "<开发者决定>"`：恢复 `decision_required` 工作项，并记录本次批准的边界；
- `--check-command "<command>"`：只有 assessment 的最小验收命令不可直接使用时才覆盖，必须在 run 中说明原因。

示例：

```text
/adapt-implement minimax-m3/vllm-kunlun
/adapt-implement minimax-m3/vllm-kunlun --item WI-003
/adapt-implement minimax-m3/vllm-kunlun \
  --item WI-003 \
  --decision "允许在指定 vLLM 上游模型文件中增加最小兼容修改，不改变公共 API"
```

---

## 2. 读取并固定输入

读取：

```text
tasks/<model-id>/model.yaml
tasks/<model-id>/model-analysis.md
tasks/<model-id>/context.md                         # 存在时
tasks/<model-id>/targets/<target-id>/target.yaml
tasks/<model-id>/targets/<target-id>/assessment.md
tasks/<model-id>/targets/<target-id>/implementation.md
tasks/<model-id>/targets/<target-id>/blockers/<WI-ID>.md    # 存在时
```

记录：

- model-analysis revision；
- assessment revision；
- assessment target revision；
- engine、hardware、`target_repo`、可选 `upstream_repo`；
- 当前 HEAD、branch、dirty 状态；
- 当前 Python 路径与关键包 import 来源；
- 用户约束和已批准决定。

若模型分析 revision 新于 assessment，停止并输出：

```text
/adapt-assess <model-id>/<target-id>
```

若 assessment 为 `blocked`、`inconclusive` 或环境 blocked，不进入代码修改。

---

## 3. 选择工作项

按 `requirements.md` 的选择规则选择一个工作项。

检查：

- 所有依赖工作项为 `passed` 或 `not_applicable`；
- 工作项有明确 capability；
- 有直接证据；
- 明确指定 `target_repo` 或 `upstream_repo` 作为本项唯一可编辑仓库；
- 有候选修改范围；
- 有最小验收命令或可明确构造的验收；
- 当前状态允许执行。

若工作项过大或需要同时修改两个仓库，标记 `blocked`，要求重新运行 `/adapt-assess` 拆分，不在实现阶段自行形成多个隐含任务。

从 `target.yaml` 解析当前工作项指定的仓库路径，后续记为 `<editable-repo>`。未声明或指向其他路径时停止。

---

## 4. 创建本次 run

执行：

```bash
python scripts/implementation/create_run.py \
  --model-id <model-id> \
  --target-id <target-id> \
  --item-id <WI-ID> \
  --target-repo <editable-repo> \
  --repository-role <target_repo|upstream_repo> \
  --runs-root runs
```

命令输出本次 `<run-dir>`。

保存当前工作项原文到：

```text
<run-dir>/work-item.md
```

执行修改前快照：

```bash
python scripts/implementation/snapshot_repo.py \
  --target-repo <editable-repo> \
  --run-dir <run-dir> \
  --phase before \
  --base-ref <assessment-target-revision>
```

---

## 5. 复核仓库和执行上下文

### 5.1 首次实现

如果当前工作项在该可编辑仓库中还没有 implement run，而仓库存在 dirty 修改：

- 不假设这些修改属于当前任务；
- 保存 `repo-before.json` 和 patch；
- 标记 `blocked` 或 `decision_required`；
- 要求开发者清理、迁移到专用 clone，或通过 `--decision` 明确接受当前工作区。

### 5.2 后续实现

比较当前 `patch_sha256` 与 `implementation.md` 的 `last_observed_patch_sha256`。

不一致且没有对应 run 时，分类为 `repository_drift`。停止修改，避免覆盖开发者或其他会话的变更。

### 5.3 执行环境

先保存轻量执行上下文：

```bash
python scripts/implementation/collect_context.py \
  --engine <engine> \
  --target-repo <editable-repo> \
  --output <run-dir>/execution-context.json
```

检查当前 Python、关键模块 import 来源和版本是否与 assessment 一致。只有当前局部验收依赖 P800 时，才执行：

```bash
bash scripts/hardware/p800/preflight.sh \
  --run-dir <run-dir>/p800-preflight \
  --target-repo <editable-repo> \
  --engine <engine> \
  --required-devices <required-device-count>
```

本阶段不进行完整环境勘测，也不安装或升级依赖。若 Python、import 来源、关键版本或设备可见性发生实质变化，标记 `environment_blocker`，停止修改并重新执行 `/adapt-assess`。

---

## 6. 读取历史尝试

执行：

```bash
python scripts/implementation/history.py \
  --runs-root runs/<model-id>/<target-id> \
  --item-id <WI-ID> \
  --output <run-dir>/history.json
```

至少阅读最近 3 个相关 run；若失败签名重复，则继续阅读最早一次相同签名的 run。

在 `<run-dir>/history-summary.md` 写明：

- 已尝试假设；
- 已修改区域；
- 已运行命令；
- 失败签名及出现次数；
- 已排除原因；
- 尚未验证的线索。

不得只读 `implementation.md` 摘要而忽略原始 stderr 和命令。

---

## 7. 加载知识

先读取通用 implementation 知识和目标引擎 `implement.md`，再根据 capability 读取适配知识。

失败前不读取 pitfalls。获得失败签名后，按 `blocker-policy.md` 分类，再读取与当前签名匹配的 pitfalls。

若知识文档版本或路径与当前 commit 不一致，以目标仓实际代码为准，并在 attempt 中注明知识适用边界。

---

## 8. 建立本轮假设

先创建下一 attempt：

```bash
python scripts/implementation/create_attempt.py \
  --run-dir <run-dir> \
  --max-attempts <N>
```

命令输出 `<attempt-dir>`。在其中创建 `attempt.md`，并按 [attempt-template.md](attempt-template.md) 写入：

- 当前观察；
- 当前假设；
- 直接证据；
- 与历史尝试的差异；
- 允许修改范围；
- 预期观察；
- 否证条件；
- 计划验收命令。

只有完成这些内容后才能编辑代码。

在编辑前为当前 attempt 记录独立快照：

```bash
python scripts/implementation/snapshot_repo.py \
  --target-repo <editable-repo> \
  --run-dir <run-dir>/attempts/<NN> \
  --phase before \
  --base-ref <assessment-target-revision>
```

该快照用于区分“此前工作项的累计修改”和“本 attempt 新产生的修改”。

---

## 9. 修改代码

原则：

- 只做支持当前假设的最小修改；
- 优先复用已有引擎扩展点；
- 硬件专用与模型专用修改尽量分离；
- 发现新的独立 capability 时停止并返回 assessment；
- 不为当前工作项顺带进行格式化、重命名或无关清理。

修改后在 attempt 中记录：

- 实际修改文件；
- 关键语义变化；
- 与原计划的偏差；
- 是否出现新依赖。

---

## 10. 范围检查和验收

根据工作项候选范围构造一个或多个 `--allow` glob。

执行：

```bash
bash scripts/engines/<engine>/check_implementation.sh \
  --target-repo <editable-repo> \
  --run-dir <run-dir>/attempts/<NN>/verification \
  --base-ref <assessment-target-revision> \
  --before-snapshot <run-dir>/attempts/<NN>/repo-before.json \
  --allow '<glob-1>' \
  --allow '<glob-2>' \
  --timeout-seconds <seconds> \
  -- <最小验收命令及参数>
```

脚本会：

1. 在验收前保存 Git 状态和累计 patch；
2. 在命令执行前检查当前代码修改是否越界；
3. 在可编辑仓库工作目录运行指定命令；
4. 保存 stdout、stderr、退出码和超时清理结果；
5. 在命令执行后再次保存仓库状态并检查测试是否产生越界文件；
6. 写入 `verification/summary.json`。

必须使用明确的工作项验收命令。没有命令时不得把语法检查写成通过。

---

## 11. 成功处理

验收和范围检查均通过时：

1. 执行 `snapshot_repo.py --phase after`；
2. 更新 attempt 结果为 `passed`；
3. 写 `<run-dir>/outcome.json`；
4. 将工作项状态改为 `passed`；
5. 更新 attempts、latest_run、证据和 patch hash；
6. 若所有工作项均通过，将 implementation 总状态改为 `passed`；
7. 不自动 commit，不处理下一工作项。

建议输出下一条命令，但不执行：

```text
/adapt-implement <model-id>/<target-id>
```

或全部完成后：

```text
/adapt-validate <model-id>/<target-id>
```

---

## 12. 失败处理

### 12.1 提取失败签名

```bash
python scripts/implementation/failure_signature.py \
  --command-result <verification>/command/result.json \
  --stdout <verification>/command/stdout.log \
  --stderr <verification>/command/stderr.log \
  --output <attempt-dir>/failure-signature.json
```

失败签名只是稳定定位符，不是根因。

### 12.2 分类

按 [blocker-policy.md](blocker-policy.md) 分类：

- `implementation_defect`：仍可提出新的局部假设；
- `model_fact_gap`：返回 `/model-analyze --update`；
- `assessment_gap`：返回 `/adapt-assess`；
- `environment_blocker`：停止代码修改；
- `repository_drift`：等待开发者确认；
- `scope_expansion`：返回 assessment；
- `test_oracle_gap`：需要补充验收或人工决定；
- `dependency_or_upstream_boundary`：生成开发者决策单；
- `transient_execution`：只有存在瞬态证据时允许一次受控重试。

### 12.3 处理失败 patch

进入下一假设前，在当前 `attempt.md` 中记录：

```text
patch_disposition: retain | amend | remove
```

- `retain`：必须有独立证据说明该修改仍然正确；
- `amend`：说明保留部分和下一轮要修正的部分；
- `remove`：只用定向编辑撤销本 attempt 引入的修改，并检查工作区是否恢复到 attempt 的 `repo-before.json` 对应状态。

不允许用 `git reset --hard`、`git clean` 或强制 checkout 处理失败 patch。无法区分本 attempt 与外部修改时，分类为 `repository_drift`。

### 12.4 是否继续

继续下一假设必须满足：

- 有新增证据；
- 假设与之前不同；
- 上一 attempt 的 patch 已明确处置；
- 修改仍在当前工作项和单一仓库范围；
- 尚未达到停止条件。

否则进入 blocked 或 decision_required。

---

## 13. Blocked 处理

`blocked` 适用于解除路径明确且不需要权衡的情况。

更新 `implementation.md`：

- blocker 分类；
- 缺失输入或环境条件；
- 直接证据；
- 唯一解除动作；
- 恢复命令。

示例：

```text
/model-analyze <model-id> --update "确认 grouped routing 中 top-k 前后的归一化顺序"
```

或：

```text
/adapt-assess <model-id>/<target-id>
```

停止当前 Skill。

---

## 14. Developer decision 处理

达到 `decision_required` 条件时：

1. 停止进一步编辑；
2. 执行 after snapshot；
3. 按 [blocker-template.md](blocker-template.md) 创建或更新：

```text
tasks/<model-id>/targets/<target-id>/blockers/<WI-ID>.md
```

4. 只给出最多 3 个有证据的选项；
5. 标记一个推荐选项及理由；
6. 明确每个选项的影响范围和验证成本；
7. 将当前工作项状态改为 `decision_required`；
8. 输出恢复命令：

```text
/adapt-implement <model-id>/<target-id> \
  --item <WI-ID> \
  --decision "<开发者决定>"
```

没有开发者决定时，不继续修改。

---

## 15. 最终一致性检查

结束前确认：

- 只修改了允许位置；
- run 中保存了所有已执行命令；
- `implementation.md` 的 latest_run 指向本次 run；
- patch hash 与 after snapshot 一致；
- 失败事实和根因推断被明确区分；
- 没有覆盖历史 run；
- 没有自动进入下一阶段。

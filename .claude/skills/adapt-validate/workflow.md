# `adapt-validate` 工作流

## 0. 总体流程

```text
解析目标
→ 固定模型/代码/环境/Runbook 指纹
→ 从模型分析与实现项建立 required coverage
→ 读取验证知识和历史失败
→ 执行同一 Runbook 的 validate check
→ 读取结构化 case summary 与原始日志
→ 判断 passed / failed / blocked / partial
→ 分类失败并写 validation.md
→ 停止
```

---

## 1. 解析参数

第一个参数必须为：

```text
<model-id>/<target-id>
```

可选：

- `--check <name>`：默认 `validate`，映射到 `runbook/checks/<name>.sh`；
- `--against-running`：仅在开发者已经显式 `/model-run --serve` 时使用当前托管服务。

正式结果默认使用全新服务；`--against-running` 主要用于快速诊断，并需在报告中标记。

---

## 2. 读取任务与阶段输入

读取：

```text
tasks/<model-id>/model.yaml
tasks/<model-id>/model-analysis.md
tasks/<model-id>/context.md                         # 存在时
tasks/<model-id>/targets/<target-id>/target.yaml
tasks/<model-id>/targets/<target-id>/assessment.md
tasks/<model-id>/targets/<target-id>/implementation.md
tasks/<model-id>/targets/<target-id>/validation.md  # 存在时
tasks/<model-id>/targets/<target-id>/runbook/
```

若 assessment 是 `already_supported`，允许没有 implementation 结果；否则要求所有阻塞正式验证的工作项均已通过。

---

## 3. 建立验证覆盖表

从以下内容提取 required capability：

1. `model-analysis.md` 中 Attention、RoPE、MoE、量化、多模态、MTP、tokenizer 和生成行为；
2. `assessment.md` 中所有风险和已确认 gap；
3. `implementation.md` 中所有已修改 capability；
4. 开发者在 context 中声明的使用场景。

在执行前形成表：

| Case | Required | Oracle | 证据来源 | 对应 capability |
|---|---|---|---|---|

然后检查 `checks/validate.sh` 是否包含或调用了对应 case。无法确认覆盖时，执行结果最多为 `partial`。

---

## 4. 加载知识

始终读取：

```text
knowledge/common/validation/validation-levels.md
knowledge/common/validation/parity.md
knowledge/common/validation/failure-classification.md
knowledge/engines/<engine>/validate.md
```

按模型能力读取：

- batch/长上下文：`scenario-validation.md`；
- 目标硬件和并行：P800 distributed/operator 文档；
- 已知失败签名出现后，再读取 engine/hardware/model pitfalls。

知识文档只能提供候选解释，当前代码与日志证据优先。

---

## 5. 读取历史

查找：

```text
runs/<model-id>/<target-id>/*-validate/
```

至少比较：

- validation summary；
- 失败 case；
- failure signature；
- model/target/runbook hashes；
- 上次建议动作。

相同输入指纹和相同失败签名已重复两次时，不再直接重跑，先进入 `decision_required`。

---

## 6. 执行验证

### 新服务

```bash
RUN_DIR="runs/<model-id>/<target-id>/$(date +%Y%m%d-%H%M%S)-validate"
python scripts/model_runtime.py run \
  <model-id>/<target-id> \
  --check <check-name> \
  --run-dir "$RUN_DIR"
```

### 已运行服务

```bash
RUN_DIR="runs/<model-id>/<target-id>/$(date +%Y%m%d-%H%M%S)-validate"
python scripts/model_runtime.py exec \
  <model-id>/<target-id> \
  --check <check-name> \
  --run-dir "$RUN_DIR"
```

不得直接 source `env.sh` 后自行启动另一个服务。

---

## 7. 读取结果

按顺序读取：

```text
<run-dir>/context.json
<run-dir>/result.json
<run-dir>/validation/summary.json
<run-dir>/validation/events.jsonl
<run-dir>/validation/cases/*
<run-dir>/logs/server.stderr.log
<run-dir>/logs/server.stdout.log
<run-dir>/logs/ready.log
<run-dir>/logs/check-<name>.stderr.log
<run-dir>/logs/check-<name>.stdout.log
```

判定优先级：

1. 服务启动、readiness 或 cleanup 失败；
2. 结构化 validation summary；
3. case 原始日志和比较结果；
4. 非结构化脚本退出码。

若 `summary.json` 不存在：

- 脚本失败：`failed` 或 `blocked`；
- 脚本成功：只能判定 `partial`，并说明缺少结构化 coverage。

---

## 8. 失败分类

按 [failure-routing.md](failure-routing.md) 分类，并引用具体证据。

不得将错误类型直接当作根因。必须区分：

```text
观察到的失败
候选解释
已验证事实
下一验证动作
```

---

## 9. 更新 `validation.md`

严格使用 [validation-template.md](validation-template.md)。

至少记录：

- 当前状态；
- model/assessment/implementation revision；
- 最新 run；
- required coverage；
- case 结果；
- 参考 oracle 和阈值；
- 已知偏差；
- 失败分类和下一动作；
- 是否允许进入 benchmark。

`validation.md` 只表示最新有效结论；历史事实保留在 `runs/`。

---

## 10. 结束输出

只给出一个主动作：

- `passed`：`/adapt-benchmark <target>`；
- `model_fact_gap`：精确的 `/model-analyze <model> --update "..."`；
- `assessment_gap`：`/adapt-assess <target>`；
- `implementation_defect`：`/adapt-implement <target> --item <ID>`，或要求 assess 新增工作项；
- `runbook_or_oracle_gap`：指出应编辑的单一文件；
- `decision_required`：指出 `validation.md` 中的决策段。

不自动执行下一命令。

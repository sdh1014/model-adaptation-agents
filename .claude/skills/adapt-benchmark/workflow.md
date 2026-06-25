# `adapt-benchmark` 工作流

## 0. 总体流程

```text
解析目标
→ 检查 validation 通过
→ 固定代码/环境/Runbook/workload 指纹
→ 读取 benchmark 知识和历史 baseline
→ 启动同一 Runbook 服务
→ 执行 checks/benchmark.sh
→ 汇总重复样本与目标
→ 可比时生成 baseline comparison
→ 写 benchmark.md
→ 停止
```

---

## 1. 参数

第一个参数：

```text
<model-id>/<target-id>
```

可选：

- `--check <name>`：默认 `benchmark`；
- `--baseline <run-dir>`：指定对比 run；
- `--against-running`：对当前托管服务执行，结果必须标记为诊断性或说明服务启动指纹。

正式 benchmark 默认启动全新服务。

---

## 2. 读取输入

读取：

```text
tasks/<model-id>/model.yaml
tasks/<model-id>/model-analysis.md
tasks/<model-id>/targets/<target-id>/target.yaml
tasks/<model-id>/targets/<target-id>/assessment.md
tasks/<model-id>/targets/<target-id>/implementation.md
tasks/<model-id>/targets/<target-id>/validation.md
tasks/<model-id>/targets/<target-id>/benchmark.md   # 存在时
tasks/<model-id>/targets/<target-id>/runbook/
```

`validation.md` 不是 `passed` 或 `benchmark_ready` 不是 true 时立即 blocked。

---

## 3. 加载知识

始终读取：

```text
knowledge/common/benchmark/methodology.md
knowledge/common/benchmark/metrics.md
knowledge/common/benchmark/reproducibility.md
knowledge/engines/<engine>/benchmark.md
```

根据 workload 和硬件读取 P800 distributed/operator 知识。只有出现异常时读取 pitfalls。

---

## 4. 检查 benchmark 定义

检查：

```text
runbook/checks/<check-name>.sh
```

确认脚本固定了 workload，并且：

- 使用 `benchmark_case`，或能够生成等价结构化 summary；
- warmup 与 repeat 明确；
- 输出路径使用 `$RUN_DIR`；
- 不在脚本中修改源码或安装依赖；
- 不在脚本中启动第二套模型服务。

未配置时只指出应编辑该文件，不猜测参数。

---

## 5. 执行

### 新服务

```bash
RUN_DIR="runs/<model-id>/<target-id>/$(date +%Y%m%d-%H%M%S)-benchmark"
python scripts/model_runtime.py run \
  <model-id>/<target-id> \
  --check <check-name> \
  --run-dir "$RUN_DIR"
```

### 当前托管服务

```bash
RUN_DIR="runs/<model-id>/<target-id>/$(date +%Y%m%d-%H%M%S)-benchmark"
python scripts/model_runtime.py exec \
  <model-id>/<target-id> \
  --check <check-name> \
  --run-dir "$RUN_DIR"
```

---

## 6. 读取结果

按顺序读取：

```text
<run-dir>/context.json
<run-dir>/result.json
<run-dir>/benchmark/summary.json
<run-dir>/benchmark/events.jsonl
<run-dir>/benchmark/cases/*
<run-dir>/logs/server.*
<run-dir>/logs/check-<name>.*
```

若 benchmark 脚本返回 0 但没有 `summary.json`：

- 允许保存原始结果；
- 阶段状态最多为 `partial`；
- 不自动提取或猜测指标。

---

## 7. Baseline

优先级：

1. 用户通过 `--baseline` 指定；
2. 否则查找最近一次 `status: passed` 且 workload 指纹一致的 benchmark；
3. 没有可比 baseline 时不进行 regression 判定。

比较命令：

```bash
python scripts/benchmark/compare.py \
  --current <run-dir>/benchmark/summary.json \
  --baseline <baseline>/benchmark/summary.json \
  --output <run-dir>/benchmark/comparison.json
```

只有显式提供 tolerance 时才输出 regression pass/fail；否则只输出变化。

---

## 8. 更新 `benchmark.md`

严格使用 [benchmark-template.md](benchmark-template.md)。

必须区分：

```text
status       压测是否有效完成
target_met   是否达到脚本中显式目标
regression   与可比 baseline 的结果；无阈值时只描述
```

报告中保存：

- 模型/代码/环境/Runbook 指纹；
- workload；
- 各 case 样本数；
- 聚合指标；
- 失败样本；
- baseline 可比性；
- 目标是否满足。

---

## 9. 结束

只输出当前结论：

- benchmark 有效且目标满足；
- benchmark 有效但目标未满足；
- benchmark blocked/failed/partial 及一个主修复动作。

性能未达标时不自动调整参数。后续需要时应新增独立 `adapt-optimize` 阶段。

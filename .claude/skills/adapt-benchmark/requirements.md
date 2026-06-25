# `adapt-benchmark` 要求

## 1. 阶段目标

本阶段在正确性已确认的固定实现上，生成可重复、可比较的性能事实。

必须回答：

1. 本次 workload、环境和代码是否固定；
2. 压测是否有效执行；
3. 关键性能指标及其统计分布；
4. 是否达到显式目标；
5. 与可比 baseline 相比发生了什么变化；
6. 结果不可比较或不稳定的原因是什么。

本阶段不解释为代码优化方案，不修改目标代码。

---

## 2. 前置条件

默认必须满足：

- `validation.md` 状态为 `passed`；
- `benchmark_ready: true`；
- 当前模型、代码 patch、运行环境和 Runbook 未发生未记录漂移；
- `runbook/checks/benchmark.sh` 已配置；
- workload 的输入长度、输出长度、并发、请求数和数据集已固定；
- 有足够资源完成测试。

若正确性未通过，状态为 `blocked`，不得用性能结果替代正确性结论。

---

## 3. Workload 契约

每个 benchmark case 至少固定：

- case 名称；
- 数据集或输入生成方式；
- input/output token 长度；
- 请求数；
- 并发或 request rate；
- sampling 参数；
- warmup 次数；
- 正式 repeat 次数；
- TP/DP/EP 等并行配置；
- 是否启用 prefix cache、speculative、quantization 等特性。

不能比较 workload 不同的两次结果。

---

## 4. 指标

推荐指标：

- request throughput；
- input token throughput；
- output token throughput；
- TTFT：median/p90/p99；
- TPOT：median/p90/p99；
- ITL：median/p90/p99；
- end-to-end latency；
- completed/failed requests；
- peak device memory；
- 服务启动时间。

不是所有引擎或 workload 都能提供全部指标。缺失项应标记，不得估算。

---

## 5. 结构化 Runbook 契约

推荐：

```bash
source "$CONTROL_ROOT/scripts/benchmark/lib.sh"
benchmark_init
benchmark_case <name> <required|optional> --warmup <N> --repeat <N> -- <command...>
benchmark_expect <case> <metric> <higher|lower> <threshold> [statistic] [unit]
benchmark_finish
```

`benchmark_case` 会为每次正式样本设置：

```text
BENCHMARK_CASE
BENCHMARK_ITERATION
BENCHMARK_SAMPLE_DIR
BENCHMARK_SAMPLE_FILE
```

测试命令必须将一个 JSON 对象写入 `$BENCHMARK_SAMPLE_FILE`。可以直接将现有 benchmark 工具的 JSON 输出写入该路径，或使用 `scripts/benchmark/extract_json.py` 做字段映射。

---

## 6. 样本与统计

- warmup 不计入正式统计；
- required case 默认建议至少 3 个正式样本；
- 每个样本保留原始 JSON 和 stdout/stderr；
- 聚合至少包含 count、mean、median、min、max、stddev、p90、p95、p99；
- 失败样本不得静默删除；
- 若 required repeat 未完成，执行状态不能为 `passed`；
- 明显异常值只能在报告中解释，不能自动剔除。

---

## 7. 阶段状态与性能目标分离

### `status: passed`

- 所有 required case 完成指定正式 repeat；
- 每个正式样本包含有效指标；
- 服务和 cleanup 正常；
- 环境与 workload 指纹完整。

### `status: failed`

- required case 命令失败；
- 样本文件缺失或无有效数值；
- 服务/cleanup 失败；
- 指标明显不可信且无法解释。

### `status: blocked`

- 正确性未通过；
- benchmark Runbook 未配置；
- 环境/设备/数据集缺失；
- workload 未固定。

### `status: partial`

- required case 通过，但 optional case 未完成；
- 使用非结构化脚本，无法生成完整统计；
- 只有部分指标可用。

### `target_met`

- `true`：所有显式期望满足；
- `false`：至少一个显式期望未满足；
- `null`：未设置性能目标。

`target_met: false` 不改变 `status: passed`。

---

## 8. Baseline 比较

可比较的 baseline 必须具有相同：

- 模型和 revision；
- workload；
- 并行和服务参数；
- 硬件数量和型号；
- dtype/quantization；
- benchmark 工具和主要版本。

若不同，只能并列展示，不能计算正式 regression 结论。

没有显式 tolerance 时，只报告变化百分比，不自动宣布 regression。

---

## 9. 修改边界

允许：

- 读取任务、验证报告、知识和历史 benchmark；
- 执行 Runbook；
- 保存样本和汇总；
- 更新 `benchmark.md`。

禁止：

- 修改目标代码；
- 为获得更好数值自动调整启动参数；
- 混合 warmup 和正式样本；
- 将不同 workload 混为同一 baseline；
- 只保留最好的一次结果；
- 自动进入性能优化。

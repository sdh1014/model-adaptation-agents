---
status: passed | failed | blocked | partial
target_met: true | false | null
latest_run: runs/<model>/<target>/<timestamp>-benchmark
validation_run: runs/<model>/<target>/<timestamp>-validate
benchmarked_at: <ISO-8601>
baseline_run: <path-or-null>
---

# Benchmark

## 结论

一句话说明压测是否有效、目标是否达到及可比边界。

## 输入指纹

- Model revision:
- Target revision / patch:
- Engine / version:
- Hardware / device count:
- Runtime Python/import:
- Runbook hashes:
- Benchmark tool/version:

## Workloads

| Case | Dataset | Input/Output | Concurrency/Rate | Warmup | Repeats | Status |
|---|---|---|---|---|---|---|

## Metrics

| Case | Metric | Unit | Median | Mean | P90 | P99 | Min | Max |
|---|---|---|---:|---:|---:|---:|---:|---:|

## Performance Targets

| Case | Metric/Statistic | Direction | Threshold | Actual | Met |
|---|---|---|---:|---:|---|

## Baseline Comparison

- Comparable: yes | no
- Baseline:
- Tolerance:

| Case | Metric | Baseline | Current | Change |
|---|---|---:|---:|---:|

## Failed or Missing Samples

无，或列出失败样本及证据。

## 结果边界

说明未测场景、异常波动和不能外推的结论。

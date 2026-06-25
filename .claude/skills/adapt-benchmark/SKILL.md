---
name: adapt-benchmark
description: 在正确性验证通过后，复用同一目标 Runbook 执行可复现的性能测试，汇总 TTFT、TPOT、吞吐、并发和资源指标，并将执行状态与性能目标是否达成分开报告。
argument-hint: "<model-id>/<target-id> [--check benchmark] [--baseline <run-dir>] [--against-running]"
disable-model-invocation: true
---

# 适配性能测试

对 `$ARGUMENTS` 指定的模型目标执行 benchmark。

## 执行前必须读取

1. [requirements.md](requirements.md)：前置条件、指标和可复现要求；
2. [workflow.md](workflow.md)：执行、汇总和比较流程；
3. [benchmark-template.md](benchmark-template.md)：`benchmark.md` 输出格式。

## 唯一运行定义

必须复用：

```text
tasks/<model-id>/targets/<target-id>/runbook/
├── env.sh
├── start.sh
├── ready.sh
├── stop.sh
└── checks/benchmark.sh
```

不得复制或重建另一套服务启动参数。

## 核心规则

- 默认要求最新 `validation.md` 为 `passed` 且 `benchmark_ready: true`；
- benchmark 只测量，不修改代码，不做性能优化；
- workload、warmup、repeat 和指标提取全部保留在 `checks/benchmark.sh`，开发者可直接粘贴现有命令；
- 推荐使用 `scripts/benchmark/lib.sh` 记录具名 case、重复样本和期望值；
- `status` 表示压测是否有效执行；`target_met` 单独表示是否达到性能目标；
- 性能未达标不能被写成 benchmark 执行失败；
- 结果必须包含环境、代码、模型、Runbook 和 workload 指纹；
- 完成后停止，不自动修改参数或进入优化。

## 输出

更新：

```text
tasks/<model-id>/targets/<target-id>/benchmark.md
```

创建：

```text
runs/<model-id>/<target-id>/<timestamp>-benchmark/
```

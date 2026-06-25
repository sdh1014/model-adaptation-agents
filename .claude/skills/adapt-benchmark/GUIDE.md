# adapt-benchmark 指南

默认要求最新 `validation.md` 为 passed。

执行前确认 workload、warmup、请求数、输入/输出长度、request rate/concurrency、TP 和 repeats 可复现。

`checks/benchmark.sh` 将 JSON/JSONL 写到 `$RUN_DIR/benchmark/`。执行器会解析常见 throughput、TTFT、TPOT、ITL、E2E latency 和内存字段。

状态：

```text
passed   测量有效且至少含核心吞吐和延迟
partial  命令成功但指标或 workload 不完整
failed   服务、请求或结果执行失败
blocked  验证、Runbook、环境或数据集不足
```

`target_met` 单独记录。未达标只生成开发者决策，不在本阶段修改代码。

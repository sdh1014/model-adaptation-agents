---
name: adapt-benchmark
description: 在正确性通过后复用目标 Runbook 执行正式性能测试，解析常见指标并生成 benchmark.md。
argument-hint: "<model-id>/<target-id> [--against-running] [--allow-unvalidated]"
disable-model-invocation: true
---

# 性能测试

执行前读取 [GUIDE.md](GUIDE.md)、`knowledge/benchmark.md`、对应引擎知识和最新 validation。

唯一执行入口：

```text
runbook/checks/benchmark.sh
```

执行：

```bash
python scripts/evaluate.py benchmark <model>/<target>
```

写结论前读取 `templates/reports/benchmark.md`。

边界：不修改代码、参数、权重或 Runbook；不自动优化；执行状态与性能目标必须分离。

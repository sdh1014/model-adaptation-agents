---
name: adapt-validate
description: 复用目标 Runbook 执行正式正确性验证，检查 required case 覆盖并生成 validation.md。
argument-hint: "<model-id>/<target-id> [--against-running] [--allow-incomplete]"
disable-model-invocation: true
---

# 正确性验证

执行前读取 [GUIDE.md](GUIDE.md)、`knowledge/validation.md`、对应引擎知识和最近实现/运行证据。

唯一执行入口：

```text
runbook/checks/validate.sh
```

执行：

```bash
python scripts/evaluate.py validate <model>/<target>
```

写结论前读取 `templates/reports/validation.md`。

边界：不修改代码、权重、Runbook 或知识；不把服务启动成功当作正确性通过；不自动进入 benchmark。

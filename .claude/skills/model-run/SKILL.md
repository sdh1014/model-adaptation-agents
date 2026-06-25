---
name: model-run
description: 使用目标 Runbook 启动、调用、检查和停止模型服务；支持手工 smoke、validate 与 benchmark 调试。
argument-hint: "<model-id>/<target-id> [--init | --check NAME | --serve | --against-running NAME | --status | --stop]"
disable-model-invocation: true
---

# 模型运行

执行前读取 [GUIDE.md](GUIDE.md) 和对应引擎知识。

唯一运行定义：

```text
tasks/<model>/targets/<target>/runbook/
├── env.sh
├── env.local.sh          # 可选、本地秘密
├── start.sh
├── ready.sh
├── stop.sh
└── checks/{smoke,validate,benchmark}.sh
```

默认执行 smoke。`--check validate/benchmark` 是手工调试，不生成正式阶段结论。

边界：不修改代码、权重或 Runbook；不判断正确性和性能目标；默认运行必须清理服务。

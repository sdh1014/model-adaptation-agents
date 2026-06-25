# 包说明

本包同时实现：

- `/adapt-validate`
- `/adapt-benchmark`
- 两阶段共用的 Runbook 执行契约
- Validation/Benchmark 结构化 Shell 辅助工具
- 更新后的 `/model-run` 与 `scripts/model_runtime.py`

开发者只需要维护：

```text
tasks/<model>/targets/<target>/runbook/
├── env.sh
├── start.sh
├── ready.sh
├── stop.sh
└── checks/
    ├── smoke.sh
    ├── validate.sh
    └── benchmark.sh
```

环境变量、完整启动命令和测试命令都可以直接粘贴到对应 Shell 文件中。

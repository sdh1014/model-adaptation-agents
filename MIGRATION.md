# 从旧版 model-run 迁移

## 删除的概念

新版不再需要：

```text
.claude/skills/model-runtime/
scripts/model_runtime/internal_cli.py
scripts/model_runtime/adapter_loader.py
scripts/model_runtime/plan.py
caller / purpose / parent-run / lease 参数
```

## 配置迁移

旧 `target.yaml` 中的：

```text
runtime.commands
runtime.env
完整启动参数
```

迁移到：

```text
runbook/env.sh
runbook/start.sh
```

旧 Smoke 或验证命令迁移到：

```text
runbook/checks/smoke.sh
runbook/checks/validate.sh
```

`target.yaml` 继续保存目标仓、引擎、硬件和少量默认信息，不再承担复杂命令配置。

# Runtime Runbook 规则

1. 每个模型目标的唯一运行定义位于 `tasks/<model>/targets/<target>/runbook/`。
2. 环境变量写入 `env.sh`，完整服务命令写入 `start.sh`，测试命令写入 `checks/*.sh`。
3. 禁止在 Skill、`target.yaml`、阶段报告或多个脚本中复制同一套启动命令。
4. `/model-run` 是人工入口；其他阶段直接调用 `python scripts/model_runtime.py`。
5. `adapt-validate` 固定复用 `checks/validate.sh`，不得单独拼装服务启动参数。
6. 默认运行必须清理服务；只有显式 `serve` 操作可以保留进程。
7. Runbook 属于开发者配置，Skill 不得在失败后自动改写它。
8. `start.sh` 必须保持前台运行，禁止 `nohup`、后台 `&` 和 daemonize。
9. 日志与执行结果写入 `runs/`；不得将密钥或完整环境变量转存到运行记录。

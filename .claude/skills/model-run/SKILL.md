---
name: model-run
description: 使用目标目录中的可执行 Runbook 启动、调用、检查和停止模型服务；复杂环境变量、完整启动命令与测试命令均保留为可直接粘贴的 Shell 脚本。
argument-hint: "<model-id>/<target-id> [--init | --check <name> | --serve | --against-running <name> | --status | --stop]"
disable-model-invocation: true
---

# 模型运行

对 `$ARGUMENTS` 指定的模型目标执行运行操作。

## 执行前读取

1. [runbook-contract.md](runbook-contract.md)：Runbook 文件和进程约束。
2. [workflow.md](workflow.md)：参数解析、命令映射和失败处理。
3. 需要查看示例时读取 [usage.md](usage.md)。

## 唯一运行定义

当前目标的运行配置固定放在：

```text
tasks/<model-id>/targets/<target-id>/runbook/
├── env.sh
├── start.sh
├── ready.sh
├── stop.sh
└── checks/
    ├── smoke.sh
    ├── validate.sh
    └── benchmark.sh
```

不得把完整启动参数复制到 `target.yaml`、Skill 或其他阶段。`model-run` 和后续
`adapt-validate` 必须复用该 Runbook。

## 用户操作

- 默认：启动新服务，执行 `checks/smoke.sh`，无论结果如何都停止服务。
- `--init`：创建 Runbook 模板；创建后停止，等待开发者粘贴命令。
- `--check <name>`：启动新服务并执行 `checks/<name>.sh`，随后停止；可直接使用 `validate` 或 `benchmark`。
- `--serve`：启动托管的持久服务。
- `--against-running <name>`：在当前托管服务上执行命名检查。
- `--status`：查看托管服务。
- `--stop`：停止托管服务。

## 强制边界

- 不修改目标框架源码。
- 不修改模型权重。
- 不自动改写开发者的 Runbook。
- 不把“返回了文本”解释为正确性通过。
- 失败时读取日志并报告稳定错误签名，但不在本 Skill 中修复代码。
- 默认运行必须完成进程清理；只有显式 `--serve` 可以保留服务。
- 完成后停止，不自动进入其他 Skill。

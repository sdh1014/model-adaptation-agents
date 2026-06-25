# 可执行 Runbook

## 目的

推理服务启动通常包含大量环境变量、路径、设备配置和多行参数。把这些内容拆进 YAML 会产生转义、字段同步和维护成本。Runbook 直接保留 Shell 形式，使现有命令可以原样迁移。

## 单一事实来源

每个模型目标只保留一个 Runbook：

```text
tasks/<model>/targets/<target>/runbook/
```

以下阶段必须复用它：

- `model-run`：Smoke、人工调试、持久服务；
- `adapt-assess`：需要实际 baseline 时；
- `adapt-implement`：工作项完成后的局部运行检查；
- `adapt-validate`：启动目标服务并运行 `checks/validate.sh`；
- `adapt-benchmark`：后续可增加 `checks/benchmark.sh`。

## 责任划分

```text
env.sh          定义环境
start.sh        定义服务如何启动
ready.sh        定义何时可用
checks/*.sh     定义服务启动后执行什么
model_runtime.py 定义生命周期、日志和清理
```

生命周期代码不理解 vLLM、SGLang 或模型参数；Runbook 不负责进程管理。

## 修改原则

- 改启动方式，只改 `start.sh`；
- 改环境变量，只改 `env.sh`；
- 改验证命令，只改 `checks/validate.sh`；
- 不在 Skill 文档、`target.yaml` 和多个阶段中复制同一命令。

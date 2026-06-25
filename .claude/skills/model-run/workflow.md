# `/model-run` 工作流

## 1. 解析目标

第一个参数必须为：

```text
<model-id>/<target-id>
```

目标目录：

```text
tasks/<model-id>/targets/<target-id>/
```

## 2. 参数映射

### 初始化

```text
/model-run <target> --init
```

执行：

```bash
python scripts/model_runtime.py init <target>
```

创建模板后停止，不继续启动服务。

### 默认 Smoke

```text
/model-run <target>
```

执行：

```bash
python scripts/model_runtime.py run <target> --check smoke
```

### 自定义检查

```text
/model-run <target> --check validate
```

执行：

```bash
python scripts/model_runtime.py run <target> --check validate
```

### 持久服务

```text
/model-run <target> --serve
```

执行：

```bash
python scripts/model_runtime.py serve <target>
```

### 对当前服务执行检查

```text
/model-run <target> --against-running smoke
```

执行：

```bash
python scripts/model_runtime.py exec <target> --check smoke
```

### 状态与停止

```bash
python scripts/model_runtime.py status <target>
python scripts/model_runtime.py stop <target>
```

## 3. Runbook 不存在

不要猜测启动命令。执行初始化：

```bash
python scripts/model_runtime.py init <target>
```

然后明确列出需要编辑的三个主要文件：

```text
env.sh
start.sh
checks/smoke.sh
```

本次调用结束。

## 4. 执行结果

读取 CLI 输出和 `result.json`，只报告：

- `passed`、`failed`、`blocked`；
- 失败阶段：prepare、launch、readiness、check、cleanup；
- 运行目录；
- 主要日志；
- 持久服务的 endpoint 或 PID 状态。

## 5. 失败分析

按顺序查看：

1. `server.stderr.log`；
2. `server.stdout.log`；
3. `ready.log`；
4. 当前检查的 stderr；
5. 当前检查的 stdout。

提取第一处稳定错误签名。必要时再读取：

```text
knowledge/engines/<engine>/run.md
knowledge/engines/<engine>/pitfalls.md
knowledge/hardware/p800/pitfalls.md
```

仅给出分类和证据：

```text
runbook_configuration
environment_or_import
model_load
runtime_operator
readiness
request_or_check
cleanup
unknown
```

本 Skill 不修改代码，也不自动修改 Runbook。

## 6. 与其他阶段共享

其他 Skill 不调用 `/model-run` 命令，而是直接调用唯一运行器：

```bash
python scripts/model_runtime.py run <target> \
  --check <name> \
  --run-dir <阶段运行目录>/runtime
```

`adapt-validate` 固定使用 `checks/validate.sh`，`adapt-benchmark` 固定使用 `checks/benchmark.sh`。

因此启动命令、环境变量和服务生命周期只有一份定义。

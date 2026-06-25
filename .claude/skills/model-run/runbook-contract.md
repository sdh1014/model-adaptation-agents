# Runbook 契约

## 1. 设计目的

Runbook 是“可执行配置”。开发者无需把几十个参数拆成 YAML 字段，可以直接粘贴已经验证过的 Shell 命令。

```text
env.sh              环境变量与公共参数
start.sh            完整启动命令
ready.sh            单次就绪检查
stop.sh             可选优雅停止命令
checks/smoke.sh     最小模型调用
checks/validate.sh  正确性验证入口
checks/benchmark.sh 性能测试入口
checks/<name>.sh    其他自定义检查
```

## 2. `env.sh`

允许：

```bash
FOO=bar
export BAR=baz
export LD_LIBRARY_PATH=/path:${LD_LIBRARY_PATH:-}
```

禁止：

- 安装或卸载软件；
- 修改代码或权重；
- 启动后台进程；
- 执行长时间探测。

运行器使用 `set -a` 加载该文件，因此普通赋值也会导出给后续脚本。

## 3. `start.sh`

可以粘贴多行环境初始化、`cd`、启动参数和 here-document。

必须满足：

- 服务进程保持前台；
- 不使用 `nohup`；
- 不在末尾添加 `&`；
- 不自行 daemonize；
- 返回前不能丢失服务进程的进程组；
- 推荐最后使用 `exec`。

运行器通过独立进程组管理整棵服务进程树。

## 4. `ready.sh`

每次只进行一次就绪探测：

```text
exit 0     服务已就绪
exit != 0  尚未就绪
```

不要在脚本内部写无限循环。重试和超时由运行器处理。

## 5. `checks/*.sh`

检查脚本可包含任意数量的测试命令。统一约定：

```text
exit 0   通过
exit 64  尚未配置或缺少必要输入
exit 65  已执行但覆盖不完整（partial）
其他值   执行失败
```

脚本可使用 `$RUN_DIR` 保存响应、logits、token、指标等证据。

## 6. 自动注入变量

路径类变量不可由 `env.sh` 覆盖：

```text
CONTROL_ROOT
MODEL_ID
TARGET_ID
MODEL_TASK_DIR
TARGET_TASK_DIR
RUNBOOK_DIR
RUN_DIR
MODEL_CONFIG
TARGET_CONFIG
```

从 `model.yaml` 和 `target.yaml` 提取的默认值可由 `env.sh` 覆盖：

```text
MODEL_NAME
MODEL_PATH
MODEL_REVISION
ENGINE
HARDWARE
TARGET_REPO
UPSTREAM_REPO
RUNTIME_PYTHON
TENSOR_PARALLEL_SIZE
```

常用运行变量由 `env.sh` 定义：

```text
MODEL_HOST
MODEL_PORT
MODEL_BASE_URL
MODEL_STARTUP_TIMEOUT
MODEL_CHECK_TIMEOUT
MODEL_SHUTDOWN_TIMEOUT
MODEL_READY_INTERVAL
MODEL_READY_PROBE_TIMEOUT
```

## 7. 运行证据

每次执行生成独立目录：

```text
runs/<model>/<target>/<timestamp>-run-<check>/
├── context.json
├── process.json
├── result.json
└── logs/
    ├── server.stdout.log
    ├── server.stderr.log
    ├── ready.log
    ├── check-<name>.stdout.log
    ├── check-<name>.stderr.log
    ├── stop.stdout.log
    └── stop.stderr.log
```

`context.json` 只记录非敏感运行参数和 Runbook 文件哈希，不复制可能包含密钥的 `env.sh`。

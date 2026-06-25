# Model Run：Runbook 版

## 最小心智模型

```text
/model-run
    ↓
scripts/model_runtime.py
    ↓
tasks/<model>/targets/<target>/runbook/
```

使用者只需维护一个目录：

```text
runbook/
├── env.sh
├── start.sh
├── ready.sh
├── stop.sh
└── checks/
    ├── smoke.sh
    └── validate.sh
```

## 为什么采用 Shell Runbook

模型服务命令通常已经以 Shell 形式存在，并包含：

- 大量 `export`；
- 多行启动参数；
- 环境初始化和 `cd`；
- 管道、here-document、curl；
- 一组连续测试命令。

保留 Shell 形式比把内容拆成 YAML 字段更直接，也减少两份配置不一致的问题。

## 快速开始

```text
/model-run minimax-m3/vllm-kunlun --init
```

然后直接编辑：

```text
tasks/minimax-m3/targets/vllm-kunlun/runbook/env.sh
tasks/minimax-m3/targets/vllm-kunlun/runbook/start.sh
tasks/minimax-m3/targets/vllm-kunlun/runbook/checks/smoke.sh
```

执行：

```text
/model-run minimax-m3/vllm-kunlun
```

## 验证复用

正确性测试全部放入：

```text
runbook/checks/validate.sh
```

手动执行：

```text
/model-run minimax-m3/vllm-kunlun --check validate
```

后续 `adapt-validate` 使用相同入口：

```bash
python scripts/model_runtime.py run \
  minimax-m3/vllm-kunlun \
  --check validate \
  --run-dir runs/.../runtime
```

## 持久服务

```text
/model-run minimax-m3/vllm-kunlun --serve
/model-run minimax-m3/vllm-kunlun --status
/model-run minimax-m3/vllm-kunlun --against-running smoke
/model-run minimax-m3/vllm-kunlun --stop
```

## Docker

控制仓、目标仓和模型目录继续使用 bind mount。Runbook 中写容器内路径：

```text
CONTROL_ROOT=/workspace/control
TARGET_REPO=/workspace/targets/...
MODEL_PATH=/models/...
```

运行器、Claude Code 和推理引擎在同一个开发容器中执行，不需要 Docker socket 或宿主机守护进程。

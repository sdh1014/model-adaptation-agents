# 使用示例

## 首次创建 Runbook

```text
/model-run minimax-m3/vllm-kunlun --init
```

编辑：

```text
tasks/minimax-m3/targets/vllm-kunlun/runbook/env.sh
tasks/minimax-m3/targets/vllm-kunlun/runbook/start.sh
tasks/minimax-m3/targets/vllm-kunlun/runbook/checks/smoke.sh
```

## 默认 Smoke

```text
/model-run minimax-m3/vllm-kunlun
```

## 执行验证入口

```text
/model-run minimax-m3/vllm-kunlun --check validate
```

## 启动持久服务

```text
/model-run minimax-m3/vllm-kunlun --serve
```

## 在持久服务上执行 Smoke

```text
/model-run minimax-m3/vllm-kunlun --against-running smoke
```

## 查看和停止

```text
/model-run minimax-m3/vllm-kunlun --status
/model-run minimax-m3/vllm-kunlun --stop
```

## 执行 Benchmark 检查

```text
/model-run <model>/<target> --check benchmark
```

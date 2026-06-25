# SGLang-Kunlun

## 仓库形态

目标可能是完整 fork、独立平台插件或上游仓加本地 patch。先从当前 commit 判断，不预设形态。

## 评估边界

分别检查模型 registry、weight loader、model runner、Attention/KV Cache、MoE、量化、并行、平台发现和本地运行时。

## 实现优先级

```text
复用 SGLang 原生模型
→ 使用当前版本扩展接口
→ Kunlun backend/plugin 最小 override
→ 必要时请求修改核心
```

## 运行与验证

Runbook `start.sh` 保存当前版本实际可用的完整启动命令。模型、TP 和服务参数名可能随版本变化，必须依据当前 `--help`。

## Benchmark

优先使用当前版本的 `python -m sglang.bench_serving`，输出 JSONL 到 `$RUN_DIR/benchmark/`。

# Docker 工作区

推荐宿主机布局：

```text
/srv/model-adaptation/
├── control/model-adaptation-agents/
├── workspaces/<model>/<target>/
├── models/
└── cache/
```

容器内使用相同绝对路径或稳定映射：

```text
/workspace/control
/workspace/workspaces
/models
/cache
```

原则：

- 控制仓和目标仓通过读写 bind mount 持久化；
- 模型目录只读挂载；
- Claude Code、推理服务和测试都在同一个开发容器中运行；
- 每个模型目标使用独立 clone 或独立 worktree；
- 不挂载 Docker socket；
- 容器进程尽量使用宿主机相同 UID/GID。

启动 Claude Code：

```bash
cd /workspace/control
claude \
  --add-dir /workspace/workspaces/minimax-m3/vllm-kunlun \
  --add-dir /models/MiniMax-M3
```

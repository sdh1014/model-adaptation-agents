# Docker 开发工作区

## 推荐结论

**宿主机保存唯一代码副本，开发容器通过 bind mount 访问。**

控制仓库、目标仓库、任务文件和运行证据都必须在宿主机持久化；不要把唯一代码或 runs 放在容器可写层。

## 宿主机目录

推荐：

```text
/srv/model-adaptation/
├── control/                              # model-adaptation-agents
├── targets/
│   ├── minimax-m3-vllm-kunlun/
│   │   ├── engine/                       # vLLM-Kunlun 独立 clone
│   │   └── upstream/                     # vLLM 独立 clone，可选
│   └── minimax-m3-sglang-kunlun/
│       ├── engine/                       # SGLang-Kunlun 插件或 fork
│       └── upstream/                     # SGLang，可选
├── models/                               # 模型权重
├── cache/                                # HF、编译等缓存
└── home/                                 # 容器用户 HOME
```

容器内固定映射：

```text
/workspace/control
/workspace/targets
/models
/cache
/workspace/.home
```

`/workspace/control` 是 Claude Code 的启动目录。

## 为什么每个模型目标使用独立 clone

每个 `<model>/<target>` 使用独立 clone 和独立分支，主要为了：

- 避免不同模型共享 dirty 工作区；
- 让 HEAD、patch 和 run 一一对应；
- 避免一个任务的失败修改污染另一个任务；
- 避免 Git worktree 中记录的宿主机路径在容器路径下失效；
- 让开发者可以直接删除和重新创建单个任务工作区。

分支建议：

```text
adapt/<model-id>
```

`adapt-implement` 不自动提交。工作项通过后，由开发者按团队规范创建 checkpoint commit。

## 一个工作项只修改一个仓库

`target.yaml` 可以声明两个仓库：

```yaml
engine: vllm-kunlun
hardware: p800

target_repo: /workspace/targets/minimax-m3-vllm-kunlun/engine
upstream_repo: /workspace/targets/minimax-m3-vllm-kunlun/upstream
```

每个 `implementation.md` 工作项必须指定：

```text
可编辑仓库：target_repo
```

或：

```text
可编辑仓库：upstream_repo
```

一个工作项不得同时修改两个仓库。确需跨仓修改时，由 `adapt-assess` 拆成两个有依赖关系的工作项。

## Bind mount 权限

- `control`：读写，Skill 需要写入 `tasks/` 和 `runs/`；
- `targets`：读写，`adapt-implement` 修改源码；
- `models`：只读；
- `cache`：读写；
- 不因 Claude Code 新增 Docker socket 挂载；
- 不因 Claude Code 新增 `privileged: true`；P800 的 devices、runtime、IPC 和共享内存沿用现有开发容器配置。

示例见 `docker-compose.workspace.example.yml`。它只描述工作区挂载，需要合并到已有 P800 服务中。

## Claude Code 访问目录

从控制仓启动：

```bash
cd /workspace/control
claude --add-dir /workspace/targets --add-dir /models
```

也可把 `.claude/settings.local.example.json` 复制为 `.claude/settings.local.json`。目标模型目录虽然加入可访问范围，但文件系统挂载为只读。

## UID/GID

Linux 宿主机建议让容器用户与宿主机用户 UID/GID 一致：

```bash
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
export ADAPT_ROOT=/srv/model-adaptation
```

优先修正 UID/GID，不设置全局 `safe.directory=*`。确有共享仓需求时，只把明确仓库加入 Git safe directory。

## 路径规则

任务文件只写容器内稳定路径：

```yaml
model:
  path: /models/MiniMax-M3

target_repo: /workspace/targets/minimax-m3-vllm-kunlun/engine
upstream_repo: /workspace/targets/minimax-m3-vllm-kunlun/upstream
```

不要在任务文件中混用宿主机路径与容器路径。
